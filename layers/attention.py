import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, qkv_bias: bool):
        super().__init__()
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.qkv_bias = qkv_bias

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(self.q_size, self.q_size, bias=qkv_bias)
        self.k_proj = nn.Linear(self.kv_size, self.kv_size, bias=qkv_bias)
        self.v_proj = nn.Linear(self.kv_size, self.kv_size, bias=qkv_bias)
        self.o_proj = nn.Linear(self.q_size, self.q_size, bias=qkv_bias)

    def forward(self, q, k, v, attn_mask=None):
        batch_size, seq_len, _ = q.size()
        
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # 重新排列维度以便进行注意力计算
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        assert self.num_kv_heads == self.num_heads, f"num_kv_heads({self.num_kv_heads}) must be equal to num_heads({self.num_heads})"
        
        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # 添加causal attention mask
    # def 