import numpy as np

def wrap(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi
