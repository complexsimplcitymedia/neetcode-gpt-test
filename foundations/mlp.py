import numpy as np
from numpy.typing import NDArray
from typing import List

class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        h = x
        num_layers = len(weights)
        
        for i in range(num_layers):
            # 1. Compute the linear transformation: h @ W + b
            h = np.dot(h, weights[i]) + biases[i]
            
            # 2. Apply ReLU activation for all layers EXCEPT the last one
            if i < num_layers - 1:
                h = np.maximum(0, h)
                
        # 3. Round the final output to 5 decimal places and return
        return np.round(h, 5)