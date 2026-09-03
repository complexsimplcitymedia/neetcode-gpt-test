import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # 1 / (1 + e^(-z))
        sig = 1 / (1 + np.exp(-z))
        return np.round(sig, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # max(0, z) element-wise
        return np.maximum(0, z)