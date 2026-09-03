import numpy as np
from numpy.typing import NDArray

class Solution:
    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Subtract max for numerical stability to prevent overflow
        shifted_z = z - np.max(z)
        exp_z = np.exp(shifted_z)
        out = exp_z / np.sum(exp_z)
        return np.round(out, 4)