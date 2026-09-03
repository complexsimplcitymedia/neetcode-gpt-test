import numpy as np
from typing import Tuple, List

class Solution:
    def batch_norm(
        self,
        x: List[List[float]],
        gamma: List[float],
        beta: List[float],
        running_mean: List[float],
        running_var: List[float],
        momentum: float,
        eps: float,
        training: bool
    ) -> Tuple[List[List[float]], List[float], List[float]]:
        # Convert inputs to NumPy arrays
        x_np = np.array(x, dtype=np.float64)
        gamma_np = np.array(gamma, dtype=np.float64)
        beta_np = np.array(beta, dtype=np.float64)
        running_mean_np = np.array(running_mean, dtype=np.float64)
        running_var_np = np.array(running_var, dtype=np.float64)

        if training:
            # 1. Compute batch mean and variance across the batch dimension (axis=0)
            mean = np.mean(x_np, axis=0)
            var = np.var(x_np, axis=0)

            # 2. Update running statistics
            running_mean_np = (1.0 - momentum) * running_mean_np + momentum * mean
            running_var_np = (1.0 - momentum) * running_var_np + momentum * var
        else:
            # Inference mode uses tracked running statistics
            mean = running_mean_np
            var = running_var_np

        # 3. Normalize across batch
        x_hat = (x_np - mean) / np.sqrt(var + eps)

        # 4. Scale and shift
        y = gamma_np * x_hat + beta_np

        # 5. Round to 4 decimal places and convert back to Python lists
        y_rounded = np.round(y, 4).tolist()
        mean_rounded = np.round(running_mean_np, 4).tolist()
        var_rounded = np.round(running_var_np, 4).tolist()

        return (y_rounded, mean_rounded, var_rounded)