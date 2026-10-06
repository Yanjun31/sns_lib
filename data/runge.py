import numpy as np


class Runge:
    def __init__(
        self,
    ):
        self.in_dim = 1
        self.out_dim = 1

    def __call__(self, x: np.ndarray) -> float:
        return 1 / (1 + 25 * x[0] ** 2)