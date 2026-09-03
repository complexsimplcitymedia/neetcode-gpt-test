import numpy as np
from numpy.typing import NDArray
from typing import Tuple

class Solution:
    def train(
        self, 
        X: NDArray[np.float64], 
        y: NDArray[np.float64], 
        epochs: int, 
        lr: float
    ) -> Tuple[NDArray[np.float64], float]:
        n_samples, n_features = X.shape
        
        # 1. Initialize weights as zeros and bias as 0.0
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0
        
        # 2. Gradient descent loop
        for _ in range(epochs):
            # Forward pass: y_hat = X @ w + b
            y_hat = X @ w + b
            
            # Error term: (y_hat - y)
            error = y_hat - y
            
            # Gradients
            dw = (2.0 / n_samples) * (X.T @ error)
            db = (2.0 / n_samples) * np.sum(error)
            
            # Update weights and bias
            w -= lr * dw
            b -= lr * db
            
        # 3. Round all values to 5 decimal places
        return (np.round(w, 5), round(float(b), 5))