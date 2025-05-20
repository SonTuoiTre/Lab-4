import numpy as np

class RandomFlip:


    def __init__(self, horizontal: bool = True, vertical: bool = False,
                 p: float = 0.5) -> None:
        assert horizontal or vertical, "At least one flip direction must be True."
        self.h = horizontal
        self.v = vertical
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # img: H x W x 3
        if self.h and np.random.rand() < self.p:
            img = np.flip(img, axis=1)            # left-right
        if self.v and np.random.rand() < self.p:
            img = np.flip(img, axis=0)            # top-bottom
        return img.copy()  # return contiguous array
