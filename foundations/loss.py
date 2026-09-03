import numpy as np
from numpy.typing import NDArray

class Solution:
    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        eps = 1e-7
        # Clip probabilities to prevent log(0)
        p = np.clip(y_pred, eps, 1.0 - eps)
        loss = -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        return round(float(loss), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        eps = 1e-7
        # Clip probabilities to prevent log(0)
        p = np.clip(y_pred, eps, 1.0 - eps)
        # Sum over classes (axis=-1), then average across samples
        loss = -np.mean(np.sum(y_true * np.log(p), axis=-1))
        return round(float(loss), 4)