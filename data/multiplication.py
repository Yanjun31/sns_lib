import numpy as np


class Multiplication:
    def __init__(
        self,
    ):
        self.in_dim = 2
        self.out_dim = 1

    def __call__(self, x: np.ndarray) -> float:
        return x[0] * x[1]