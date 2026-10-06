from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    def __init__(
        self,
        func,
        num_samples: int,
        low: Iterator[float],
        high: Iterator[float],
        noise_level: float,
        normalize: bool = True,
    ):
        self.num_samples = num_samples
        self.low = tuple(low)
        self.high = tuple(high)
        self.noise_level = noise_level
        self.normalize = normalize
        self.data, self.x_min, self.x_max, self.y_min, self.y_max = self._generate_data(
            func
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = self.data["labels"][index]
        data_val = self.data["features"][index]
        return data_val, target

    def _generate_data(
        self,
        func,
    ) -> tuple[
        dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        raw_x = np.random.uniform(self.low, self.high, (self.num_samples, func.in_dim))
        noise = np.random.normal(0, self.noise_level, (self.num_samples, func.out_dim))
        raw_y = (
            np.apply_along_axis(func, -1, raw_x).reshape(self.num_samples, func.out_dim)
            + noise
        )
        x, y, x_min, x_max, y_min, y_max = self._preprocess_data(raw_x, raw_y)
        return (
            {"features": torch.Tensor(x), "labels": torch.Tensor(y)},
            torch.Tensor(x_min),
            torch.Tensor(x_max),
            torch.Tensor(y_min),
            torch.Tensor(y_max),
        )

    def _preprocess_data(
        self,
        raw_x: np.ndarray,
        raw_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_min, x_max = np.min(raw_x, 0), np.max(raw_x, 0)
        y_min, y_max = np.min(raw_y, 0), np.max(raw_y, 0)

        # Apply preprocessing
        if self.normalize:
            x = (raw_x - x_min) / (x_max - x_min)
            # x = raw_x
            y = (raw_y - y_min) / (y_max - y_min)
            # y = raw_y
        else:
            x = raw_x
            y = raw_y
        return x, y, x_min, x_max, y_min, y_max