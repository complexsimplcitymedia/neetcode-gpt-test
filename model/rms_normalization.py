import numpy as np
from typing import List

class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        x_arr = np.array(x, dtype=np.float64)
        gamma_arr = np.array(gamma, dtype=np.float64)
        
        # 1. Compute Root Mean Square: sqrt(mean(x^2) + eps)
        rms = np.sqrt(np.mean(x_arr ** 2) + eps)
        
        # 2. Normalize and scale by gamma
        out = (x_arr / rms) * gamma_arr
        
        # 3. Round each value to 4 decimal places and return as a list
        return np.round(out, 4).tolist()