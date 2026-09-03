import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self, 
        X: NDArray[np.float64], 
        Y: NDArray[np.float64], 
        num_iterations: int, 
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        weights = np.array(initial_weights, dtype=np.float64)
        N = len(X)
        num_weights = len(weights)

        for _ in range(num_iterations):
            # 1. Forward pass: compute current model predictions
            predictions = self.get_model_prediction(X, weights)

            # 2. Compute gradients for each weight
            gradients = np.zeros(num_weights, dtype=np.float64)
            for j in range(num_weights):
                gradients[j] = self.get_derivative(predictions, Y, N, X, j)

            # 3. Update weights using gradient descent
            weights -= self.learning_rate * gradients

        # 4. Round final weights to 5 decimal places
        return np.round(weights, 5)