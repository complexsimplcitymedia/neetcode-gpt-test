import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        eps = 1e-5
        
        # 1. Compute mean and variance across the feature dimension
        mean = np.mean(x)
        var = np.var(x)
        
        # 2. Normalize
        x_hat = (x - mean) / np.sqrt(var + eps)
        
        # 3. Scale and shift
        out = gamma * x_hat + beta
        
        # 4. Round to 5 decimal places
        return np.round(out, 5)