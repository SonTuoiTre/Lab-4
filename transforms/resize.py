from typing import Tuple
import numpy as np
from PIL import Image

class Resize:


    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        im = Image.fromarray(img.astype(np.uint8), mode="RGB")
        im = im.resize(self.size, Image.BILINEAR)
        return np.array(im)
