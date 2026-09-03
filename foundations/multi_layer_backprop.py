import numpy as np
from typing import List

class Solution:
    def forward_and_backward(self, x: List[float], W1: List[List[float]], b1: List[float], W2: List[List[float]], b2: List[float], y_true: List[float]) -> dict:
        # Convert inputs to NumPy arrays for easier matrix math
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        
        # --- 1. Forward Pass ---
        # Layer 1
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1) # ReLU activation
        
        # Layer 2
        z2 = W2 @ a1 + b2
        
        # Loss calculation (Mean Squared Error)
        n = len(y_true)
        loss = np.mean((z2 - y_true) ** 2)
        
        # --- 2. Backward Pass ---
        # Output layer gradient
        dz2 = 2 * (z2 - y_true) / n
        
        # Layer 2 gradients
        dW2 = np.outer(dz2, a1)
        db2 = dz2
        
        # Gradient through ReLU
        da1 = dz2 @ W2
        dz1 = da1 * (z1 > 0) # ReLU derivative mask
        
        # Layer 1 gradients
        dW1 = np.outer(dz1, x)
        db1 = dz1
        
        # --- 3. Format and Return ---
        # Round values to 4 decimal places and convert arrays back to nested lists
        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }