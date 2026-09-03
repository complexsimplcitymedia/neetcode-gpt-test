class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        x = init
        
        for _ in range(iterations):
            # Compute derivative: f'(x) = 2x
            grad = 2 * x
            # Apply gradient descent update: x = x - learning_rate * grad
            x = x - learning_rate * grad
            
        return round(x, 5)