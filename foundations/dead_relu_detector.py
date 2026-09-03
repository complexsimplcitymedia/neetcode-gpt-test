import torch
import torch.nn as nn
from typing import List

class Solution:
    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fractions = []
        
        with torch.no_grad():
            out = x
            for layer in model:
                out = layer(out)
                if isinstance(layer, nn.ReLU):
                    # A neuron is dead if its output is 0 across all samples in batch (dim=0)
                    # (out == 0).all(dim=0) returns a boolean tensor per neuron
                    dead_mask = (out == 0).all(dim=0)
                    fraction = dead_mask.float().mean().item()
                    dead_fractions.append(round(fraction, 4))
                    
        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        if not dead_fractions:
            return 'healthy'

        # 1. Any layer has dead fraction > 0.5
        if any(f > 0.5 for f in dead_fractions):
            return 'use_leaky_relu'

        # 2. First layer has dead fraction > 0.3
        if dead_fractions[0] > 0.3:
            return 'reinitialize'

        # 3. Strictly increases with depth AND last layer > 0.1
        if (len(dead_fractions) > 1 and 
            all(dead_fractions[i] < dead_fractions[i + 1] for i in range(len(dead_fractions) - 1)) and 
            dead_fractions[-1] > 0.1):
            return 'reduce_learning_rate'

        # 4 & 5. Max dead fraction < 0.1 or otherwise healthy
        return 'healthy'