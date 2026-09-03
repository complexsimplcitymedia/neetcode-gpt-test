import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        loss = None

        for epoch in range(epochs):
            torch.manual_seed(epoch)
            
            # Sample random start indices
            idx = torch.randint(0, len(data) - context_length, (batch_size,))
            
            # Construct input X and target Y (shifted right by 1)
            X = torch.stack([data[i : i + context_length] for i in idx])
            Y = torch.stack([data[i + 1 : i + context_length + 1] for i in idx])
            
            # Forward pass
            logits = model(X)  # shape: (B, T, C)
            
            # Flatten for cross-entropy
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), Y.view(B * T))
            
            # Backward pass & update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(), 4)