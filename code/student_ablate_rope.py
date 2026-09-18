"""Iteration 2 ablation: learned absolute positions, RMSNorm, and SwiGLU.

This module is self-contained so both iteration checkpoints remain reproducible.
Only the positional mechanism differs from the iteration-1 student model.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class Block(nn.Module):
    def __init__(self, width, heads, hidden):
        super().__init__()
        self.heads = heads
        self.norm1 = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.gate = nn.Linear(width, hidden)
        self.up = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        h = self.norm2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width, heads = config['width'], config['heads']
        if width % heads:
            raise ValueError('Width must be divisible by attention heads.')
        hidden = 3 * width
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList(
            [Block(width, heads, hidden) for _ in range(config['depth'])]
        )
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        if ids.shape[1] > self.context:
            raise ValueError('Input exceeds the configured context.')
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.token(ids) + self.pos(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return GPT(config)
