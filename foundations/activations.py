import numpy as np
from numpy.typing import NDArray

class Solution:
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Fast vector evaluation
        return np.round(1.0 / (1.0 + np.exp(-z)), 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # clip() has less overhead than maximum(0, z)
        return z.clip(min=0)