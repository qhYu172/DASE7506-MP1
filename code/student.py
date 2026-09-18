"""Causal learned-position/RMSNorm/GELU decoder with exact memory."""
import os
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
        self.fc = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        h = self.norm2(x)
        return x + self.down(F.gelu(self.fc(h)))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width, heads = config['width'], config['heads']
        if width % heads:
            raise ValueError('Width must be divisible by attention heads.')
        hidden = 4 * width
        self.temperature = 1.10
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
        length = ids.shape[1]
        if length > self.context:
            raise ValueError('Input exceeds the configured context.')
        positions = torch.arange(length, device=ids.device)
        x = self.token(ids) + self.pos(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    @torch.inference_mode()
    def predict_log_probs(self, ids):
        # Match the longest available suffix of one, two, or three observed
        # tokens. A source at j < t supplies only its already-observed successor
        # ids[j+1]. Every cache tensor is local to this call.
        base = F.softmax(self(ids).float() / self.temperature, dim=-1)
        batch, length = ids.shape
        if length < 3:
            return base.log()
        previous = torch.cat((ids.new_full((batch, 1), -1), ids[:, :-1]), dim=1)
        second_previous = torch.cat((ids.new_full((batch, 2), -1), ids[:, :-2]), dim=1)
        past = torch.ones((length, length), dtype=torch.bool, device=ids.device).tril(-1)
        token_matches = (ids[:, :, None] == ids[:, None, :]) & past[None]
        pair_matches = token_matches & (previous[:, :, None] == previous[:, None, :])
        triple_matches = pair_matches & (
            second_previous[:, :, None] == second_previous[:, None, :]
        )
        has_triple = triple_matches.any(dim=-1, keepdim=True)
        has_pair = pair_matches.any(dim=-1, keepdim=True)
        matches = torch.where(has_triple, triple_matches,
                              torch.where(has_pair, pair_matches, token_matches))
        positions = torch.arange(length, device=ids.device)
        distance = positions[:, None] - positions[None, :]
        recency = torch.exp(-distance.clamp_min(0).float() / 512.0)
        weights = matches.to(base.dtype) * recency[None]
        weight_sum = weights.sum(dim=-1, keepdim=True)
        successors = torch.cat((ids[:, 1:], ids.new_zeros((batch, 1))), dim=1)
        cache = torch.zeros_like(base)
        cache.scatter_add_(2, successors[:, None, :].expand(-1, length, -1), weights)
        cache = cache / weight_sum.clamp_min(1e-12)
        mix = torch.where(has_triple, 0.6,
                          torch.where(has_pair, 0.35,
                                      torch.where(weight_sum > 0, 0.1, 0.0)))
        return ((1.0 - mix) * base + mix * cache).log()


def build_model(config):
    if config['width'] == 320 and config['depth'] == 8:
        torch.set_num_threads(min(32, os.cpu_count() or 4))
    return GPT(config)
