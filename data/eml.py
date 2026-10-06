import numpy as np


class EML:
    def __init__(
        self,
    ):
        self.in_dim = 2
        self.out_dim = 1

    def __call__(self, x: np.ndarray) -> float:
        return np.exp(x[0]) - np.log(x[1])