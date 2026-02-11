import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        max_posision,
        head_dim,
        rms_norm_eps = 1e-6,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ):
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
