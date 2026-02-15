import torch
from torch import nn


class RoPE(nn.Module):
    def __init__(self, head_dim, max_position=4096*32, base=10000, rope_scaling=None):
        super().__init__()
        self.head_dim = head_dim
        self.max_position = max_position
        self.base = base
        self.rope_scaling = rope_scaling
        
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)

        self.positions = [i for i in range(0, max_position)]

    def get_cos_sin(self, positions):
        freqs = torch.einsum('bs,j->bsj', positions, self.inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        return cos, sin

    def forward(self, x, positions):
        cos, sin = self.get_cos_sin(positions)
        return self.apply_rotary_emb(x, cos, sin)

    def apply_rotary_emb(self, x, cos, sin):
        x1, x2 = torch.chunk(x.float(), 2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat((y1, y2), dim=-1).to(x.dtype)
