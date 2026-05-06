import numpy as np
from app import wrap
from app.telemetry.measurement import Measurement
from app.inputs.setpoint import Setpoint
from pyglm import glm
from abc import ABC, abstractmethod

class Input(ABC):

    def __init__(self) -> None:
        self.yaw: float = 0.0
        self.position: glm.vec3 = glm.vec3(0.0, 0.0, 0.0)
        self.capture: bool = False

    def reset(self) -> None:
        self.yaw = 0.0
        self.position = glm.vec3(0.0, 0.0, 0.0)
        self.capture = False

    def to_setpoint(self, measurement: Measurement) -> Setpoint:
        xy = glm.rotateZ(glm.vec3(self.position.x, self.position.y, 0.0), measurement.rotation.z).xy
        xy = measurement.position.xy + xy
        z = np.max([0.0, measurement.position.z + self.position.z])
        yaw = wrap(measurement.rotation.z + self.yaw)
        return Setpoint(glm.vec3(xy.x, xy.y, z), yaw)

    @abstractmethod
    def update(self) -> None:
        pass
