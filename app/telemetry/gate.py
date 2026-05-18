import numpy as np
from pyglm import glm

class Gate:

    def __init__(self, corners: np.ndarray) -> None:
        assert corners.shape == (4, 2)
        self.corners = [glm.vec2(corner) for corner in corners]
