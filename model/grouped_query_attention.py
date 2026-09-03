import torch
import torch.nn as nn
from torchtyping import TensorType


class GroupedQueryAttention(nn.Module):

  def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
    super().__init__()
    torch.manual_seed(0)
    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = model_dim // num_heads

    self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
    self.k_proj = nn.Linear(
        model_dim, num_kv_heads * self.head_dim, bias=False
    )
    self.v_proj = nn.Linear(
        model_dim, num_kv_heads * self.head_dim, bias=False
    )
    self.output_proj = nn.Linear(
        num_heads * self.head_dim, model_dim, bias=False
    )

  def forward(self, x: TensorType[float]) -> TensorType[float]:
    B, T, D = x.shape

    # 1. Project x into Q, K, V
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    # 2. Reshape into heads: (B, num_heads, T, head_dim)
    q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
    v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

    # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
    num_queries_per_kv = self.num_heads // self.num_kv_heads
    k = torch.repeat_interleave(k, repeats=num_queries_per_kv, dim=1)
    v = torch.repeat_interleave(v, repeats=num_queries_per_kv, dim=1)

    # 4. Scaled dot-product attention with causal mask
    scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

    causal_mask = torch.tril(torch.ones(T, T, device=x.device)) == 0
    scores = scores.masked_fill(causal_mask, float("-inf"))

    attn_weights = nn.functional.softmax(scores, dim=-1)
    out = attn_weights @ v

    # 5. Concatenate heads and apply output projection
    out = out.transpose(1, 2).contiguous().view(B, T, -1)
    output = self.output_proj(out)

    # 6. Return rounded output (decimals=4)
    return torch.round(output * 10000) / 10000