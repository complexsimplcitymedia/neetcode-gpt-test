import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # Pre-activation value: z = w · x + b
        z = np.dot(x, w) + b
        
        # Apply selected activation function
        if activation == "sigmoid":
            out = 1 / (1 + np.exp(-z))
        elif activation == "relu":
            out = max(0.0, float(z))
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        return round(float(out), 5)