import numpy as np


class Matyas:
    def __init__(
        self,
    ):
        self.in_dim = 2
        self.out_dim = 1

    def __call__(self, x: np.ndarray) -> float:
        return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]