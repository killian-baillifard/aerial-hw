import numpy as np
from pyglm import glm
from app.io import Measurement
from app.telemetry.camera import FOV_Y, HEIGHT as CAM_HEIGHT
from app.telemetry.ray import Ray

class Gate:

    HEIGHT = 0.40   # m

    def __init__(self, corners: np.ndarray, measurement: Measurement) -> None:

        # Save constructor inputs
        assert corners.shape == (4, 2)
        self.corners = [glm.vec2(corner) for corner in corners]
        self.measurement = measurement
        
        # Find opposite corners
        max_i = 0
        max_d = 0
        for i in range(1, 4):
            d = glm.distance(self.corners[0], self.corners[i])
            if d > max_d:
                max_d = d
                max_i = i

        diag_1 = [0, max_i]
        diag_2 = [1, 2, 3]
        diag_2.remove(max_i)

        # Compute max gate height in pixels
        side_1 = [self.corners[diag_1[0]], self.corners[diag_2[0]]]
        side_2 = [self.corners[diag_1[1]], self.corners[diag_2[1]]]
        height_1_px = glm.distance(side_1[0], side_1[1])
        height_2_px = glm.distance(side_2[0], side_2[1])
        max_height_py: float = np.max([height_1_px, height_2_px])

        # Approximate gate distance
        projected_height: float = max_height_py / np.cos(self.measurement.rotation.x)
        self.distance: float = Gate.HEIGHT / (2 * np.tan(FOV_Y / 2) * (projected_height / CAM_HEIGHT))

        # Compute gate center on screen
        self.center = glm.vec2(0.0, 0.0)
        for corner in self.corners:
            self.center += corner
        self.center /= 4
    
        # Compute gate position
        ray = Ray(self.center, self.measurement.position, self.measurement.rotation)
        origin, direction = ray.cast()
        self.position = origin + direction * self.distance
