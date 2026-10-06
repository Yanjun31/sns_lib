import numpy as np


class Division:
    def __init__(self, epsilon: float = 0.05):
        self.in_dim = 2
        self.out_dim = 1
        self.epsilon = epsilon

    def __call__(
        self,
        x: np.ndarray,
    ) -> float:
        return x[0] / (self.epsilon + x[1])
