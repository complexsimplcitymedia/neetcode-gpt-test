import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_k is None or self.cache_v is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            self.cache_k = torch.cat([self.cache_k, new_k], dim=1)
            self.cache_v = torch.cat([self.cache_v, new_v], dim=1)
        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        if kv_cache is None:
            kv_cache = KVCache()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        full_k, full_v = kv_cache.update(k, v)

        model_dim = q.shape[-1]
        scores = (q @ full_k.transpose(1, 2)) / (model_dim ** 0.5)

        q_len = q.shape[1]
        total_len = full_k.shape[1]
        prev_len = total_len - q_len

        # Causal mask accounting for offset of cached tokens
        q_indices = torch.arange(q_len, device=x.device).unsqueeze(1) + prev_len
        k_indices = torch.arange(total_len, device=x.device).unsqueeze(0)
        mask = q_indices < k_indices
        scores = scores.masked_fill(mask, float('-inf'))

        attention_weights = nn.functional.softmax(scores, dim=-1)
        output = attention_weights @ full_v

        return torch.round(output * 10000) / 10000, kv_cache