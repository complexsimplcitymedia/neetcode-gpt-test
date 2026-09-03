import torch
import torch.nn as nn
from typing import List, Dict

class Solution:
    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        stats = []
        curr = x

        with torch.no_grad():
            for layer in model:
                curr = layer(curr)
                if isinstance(layer, nn.Linear):
                    mean_val = curr.mean().item()
                    std_val = curr.std().item()
                    # A neuron is dead if its output is <= 0 for all samples in the batch
                    # curr shape: (batch_size, num_neurons)
                    dead_neurons = (curr <= 0).all(dim=0)
                    dead_fraction = dead_neurons.float().mean().item()

                    stats.append({
                        'mean': round(mean_val, 4),
                        'std': round(std_val, 4),
                        'dead_fraction': round(dead_fraction, 4)
                    })

        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        model.zero_grad()
        
        criterion = nn.MSELoss()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()

        stats = []
        for layer in model:
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                mean_val = grad.mean().item()
                std_val = grad.std().item()
                norm_val = torch.norm(grad).item()

                stats.append({
                    'mean': round(mean_val, 4),
                    'std': round(std_val, 4),
                    'norm': round(norm_val, 4)
                })

        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Check 1: dead_neurons if any layer has dead_fraction > 0.5
        for s in activation_stats:
            if s['dead_fraction'] > 0.5:
                return 'dead_neurons'

        # Check 2: exploding_gradients if any layer gradient norm > 1000
        for s in gradient_stats:
            if s['norm'] > 1000:
                return 'exploding_gradients'

        # Check 3: vanishing_gradients if last layer gradient norm < 1e-5
        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'

        # Check 4: Check activation std for all layers
        for s in activation_stats:
            if s['std'] < 0.1:
                return 'vanishing_gradients'
            if s['std'] > 10.0:
                return 'exploding_gradients'

        # Check 5: healthy if none of the above
        return 'healthy'