import numpy as np


class BilinearRatio:
    def __init__(
        self,
    ):
        self.in_dim = 3
        self.out_dim = 1

    def __call__(self, x: np.ndarray) -> float:
        return x[0] * x[1] / (x[2] + 0.1)