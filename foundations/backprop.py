import numpy as np
from numpy.typing import NDArray
from typing import Tuple

class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # 1. Forward pass
        z = np.dot(w, x) + b
        y_hat = 1 / (1 + np.exp(-z)) # Sigmoid activation function
        
        # 2. Compute the common gradient term (derivative of loss with respect to z)
        # dL/dz = (y_hat - y_true) * y_hat * (1 - y_hat)
        dL_dz = (y_hat - y_true) * y_hat * (1 - y_hat)
        
        # 3. Compute gradients with respect to weights and bias
        dL_dw = dL_dz * x
        dL_db = dL_dz
        
        # 4. Round to 5 decimal places and return as a tuple
        return (np.round(dL_dw, 5), round(float(dL_db), 5))