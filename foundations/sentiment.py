import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.embedding = nn.Embedding(vocabulary_size, 16)
        self.linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # 1. Embed input token IDs: (batch_size, seq_len) -> (batch_size, seq_len, 16)
        embedded = self.embedding(x)
        
        # 2. Average across sequence dimension: (batch_size, 16)
        mean_embedded = torch.mean(embedded, dim=1)
        
        # 3. Linear layer + Sigmoid: (batch_size, 1)
        out = self.sigmoid(self.linear(mean_embedded))
        
        # 4. Round to 4 decimal places
        return torch.round(out, decimals=4)