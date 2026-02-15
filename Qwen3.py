import torch
from torch import nn
import torch.nn.functional as F
import os
from transformers import Qwen3Config
from dataclasses import dataclass

from layers.rms_norm import RMSNorm
from layers.rope import RoPE
from layers.silu import SiluAndMul


@dataclass
class Qwen3AttentionConfig:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    max_position: int = 4096 * 32
    head_dim: int | None = None
    rms_norm_eps: float = 1e-6
    qkv_bias: bool = False
    rope_theta: float = 10000
    rope_scaling: tuple | None = None


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        config
    ):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim or config.hidden_size // self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.q_size)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_size)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_size)
        self.o_proj = nn.Linear(self.q_size, config.hidden_size)
        
        self.rotary_emb = RoPE(
            self.head_dim,
            max_position=config.max_position,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape

        # (batch, seq_len, hidden_size) -> (batch, seq_len, dim)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # (batch, seq_len, dim) -> (batch, seq_len, head_num, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rotary_emb(q, positions)
        k = self.rotary_emb(k, positions)
        
        # (batch, seq_len, head_num, head_dim) -> (batch, head_num, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # GQA -- softmax(mask(QK^T))
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=attn_weights.device), diagonal=1)
        causal_mask = causal_mask.bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.q_size)
        output = self.o_proj(attn_output)

        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ):    
        super().__init__()
        attention_config = Qwen3AttentionConfig(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, 'head_dim', None),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None)
        )
        self.self_attn = Qwen3Attention(attention_config)
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ):    
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
        Qwen3ForCausalLM = Qwen3Model + LM-head
    """

    def __init__(
        self,
        config: Qwen3Config
    ):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ):
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ):
        return self.lm_head(hidden_states)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        """
        Load model from pretrained path
        """
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(pretrained_model_path)
        model = cls(config)
        
        # 加载权重
        from glob import glob
        from safetensors import safe_open
        
        weight_files = glob(os.path.join(pretrained_model_path, "*.safetensors"))
        
        if weight_files:
            # 从safetensors加载权重
            state_dict = {}
            for file in weight_files:
                with safe_open(file, "pt", "cpu") as f:
                    for weight_name in f.keys():
                        # 直接使用权重文件中的键名，不需要移除'model.'前缀
                        state_dict[weight_name] = f.get_tensor(weight_name)
            
            # 加载权重到模型
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded {len(state_dict)} weights from safetensors")
        else:
            # 尝试加载pytorch权重
            weight_files = glob(os.path.join(pretrained_model_path, "*.bin"))
            if weight_files:
                hf_state_dict = {}
                for file_path in weight_files:
                    hf_state_dict.update(torch.load(file_path, map_location="cpu"))
                
                # 加载权重到模型
                model.load_state_dict(hf_state_dict, strict=False)
                print(f"Loaded {len(hf_state_dict)} weights from pytorch bin")
        
        return model
