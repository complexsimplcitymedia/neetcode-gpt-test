import torch
import math
from typing import List

class Solution:
    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        W = torch.randn(fan_out, fan_in) * std
        W = torch.round(W * 10000) / 10000
        return W.tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        W = torch.randn(fan_out, fan_in) * std
        W = torch.round(W * 10000) / 10000
        return W.tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)
        
        # 1. Build all layer weight matrices first: shape (out_dim, in_dim)
        weights = []
        fan_in = input_dim
        fan_out = hidden_dim
        
        for _ in range(num_layers):
            if init_type == 'xavier':
                std = math.sqrt(2.0 / (fan_in + fan_out))
            elif init_type == 'kaiming':
                std = math.sqrt(2.0 / fan_in)
            else:  # 'random'
                std = 1.0
                
            W = torch.randn(fan_out, fan_in) * std
            weights.append(W)
            fan_in = hidden_dim
            
        # 2. Sample a single random input vector
        x = torch.randn(input_dim)
        stds = []
        
        # 3. Forward pass: x @ W.T + ReLU at each layer
        for W in weights:
            x = torch.relu(x @ W.T)
            stds.append(round(x.std().item(), 2))
            
        return stds