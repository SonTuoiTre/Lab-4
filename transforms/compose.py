from typing import Sequence, Callable
import numpy as np

class Compose:


    def __init__(self, transforms: Sequence[Callable[[np.ndarray], np.ndarray]]):
        self.transforms = list(transforms)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            img = t(img)
        return img