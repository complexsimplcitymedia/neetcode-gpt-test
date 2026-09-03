import torch
import torch.nn as nn
from torchtyping import TensorType

class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.attn = MultiHeadedSelfAttention(model_dim, model_dim, num_heads)
        self.ffn = VanillaNeuralNetwork(model_dim)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        x = embedded + self.attn(self.ln1(embedded))
        x = x + self.ffn(self.ln2(x))
        return torch.round(x * 10000) / 10000

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.head_size = attention_dim // num_heads
        self.heads = nn.ModuleList([
            SingleHeadAttention(embedding_dim, self.head_size)
            for _ in range(num_heads)
        ])
        self.w_o = nn.Linear(attention_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        head_outputs = [head(embedded) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=2)
        return self.w_o(concatenated)

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        k = self.key_gen(embedded)
        q = self.query_gen(embedded)
        v = self.value_gen(embedded)
        scores = q @ torch.transpose(k, 1, 2)
        head_size = k.shape[2]
        scores = scores / (head_size ** 0.5)
        context_length = k.shape[1]
        lower_triangular = torch.tril(torch.ones(context_length, context_length, device=embedded.device))
        mask = lower_triangular == 0
        scores = scores.masked_fill(mask, float('-inf'))
        scores = nn.functional.softmax(scores, dim=2)
        return scores @ v

class VanillaNeuralNetwork(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.up_projection = nn.Linear(model_dim, model_dim * 4)
        self.relu = nn.ReLU()
        self.down_projection = nn.Linear(model_dim * 4, model_dim)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        return self.down_projection(self.relu(self.up_projection(x)))