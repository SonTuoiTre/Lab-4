from typing import Tuple
import numpy as np

class Normalize:
    def __init__(self, mean: Tuple[float, float, float],
                       std:  Tuple[float, float, float]) -> None:
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std  = np.asarray(std , dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # img: H x W x 3, uint8
        x = img.astype(np.float32) / 255.0        # → 0-1
        x = x.transpose(2, 0, 1)                  # C H W
        x = (x - self.mean) / self.std
        return x.transpose(1, 2, 0)               # back to H W C
