import numpy as np
from pyglm import glm

class Measurement:

    def __init__(self, timestamp: float = 0.0, position: glm.vec3 = glm.vec3(0.0), rotation: glm.vec3 = glm.vec3(0.0), battery: float = 0.0) -> None:
        self.timestamp: float = timestamp
        self.position: glm.vec3 = position
        self.rotation: glm.vec3 = rotation
        self.battery: float = battery

    def __str__(self):
        return str(self.timestamp) + ' ' + str(self.position) + ' ' + str(self.rotation) + ' ' + str(self.battery)

    def to_array(self) -> np.ndarray:
        return np.array([
            self.timestamp,
            self.position.x,
            self.position.y,
            self.position.z,
            self.rotation.x,
            self.rotation.y,
            self.rotation.z,
            self.battery
        ])
