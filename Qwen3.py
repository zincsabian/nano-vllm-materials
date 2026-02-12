import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config
from dataclasses import dataclass


@dataclass
class Qwen3AttentionConfig:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    max_posision: int = 4096 * 32
    head_dim: int | None = None
    rms_norm_eps = 1e-6
    qkv_bias: bool = False
    rope_theta: float = 10000
    rope_scaling: tuple | None = None


class Qwen3Attention(nn.Module):

    def _process_attention(self, config: Qwen3AttentionConfig):
        pass

    def _process_norm(self):
        pass

    def _process_rope(self):
        pass

    def __init__(
        self,
        config: Qwen3AttentionConfig
    ):
        super().__init__()
        pass

    def forward(self, positions, hidden_states):
        pass


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        pass

    def forward(self, x):
        pass


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ):
        pass

    def forward(self, positions, hidden_states, redidual):
        pass


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ):
        pass


    def forward(self, input_tokens, positions):
        pass


class Qwen3ForCausalLM(nn.Module):
    """
        Qwen3ForCausalLM = Qwen3Model + LM-head
    """

    def __init__(
        self,
        config: Qwen3Config
    ):
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ):
        pass

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ):
        pass
