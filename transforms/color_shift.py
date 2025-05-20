import numpy as np

class ColorShift:


    def __init__(self, shift: tuple[int, int, int], p: float = 0.5):
        self.r_shift, self.g_shift, self.b_shift = shift
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() >= self.p:
            return img
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        r = np.roll(r, self.r_shift, axis=0)
        g = np.roll(g, self.g_shift, axis=1)
        b = np.roll(b, self.b_shift, axis=0)
        return np.stack((r, g, b), axis=-1)
