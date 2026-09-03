import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # Pre-activation: z = dot(x, w) + b
        z = np.dot(x, w) + b

        if activation == "sigmoid":
            val = 1.0 / (1.0 + np.exp(-z))
        elif activation == "relu":
            val = max(0.0, float(z))
        else:
            val = float(z)

        return round(float(val), 5)