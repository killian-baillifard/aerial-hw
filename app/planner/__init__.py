from pyglm import glm
from cv2.typing import MatLike
from app.telemetry.measurement import Measurement
from app.inputs.setpoint import Setpoint

class Planner:

    TOLERANCE = 0.01

    def __init__(self) -> None:
        self.waypoints = [
            glm.vec3(0.0, 0.0, 1.0),
            glm.vec3(-1.0, 0.0, 1.0),
            glm.vec3(-1.0, -0.5, 1.0),
            glm.vec3(-1.0, 0.5, 1.0),
            glm.vec3(-1.0, 0.0, 1.0),
            glm.vec3(-1.0, 0.0, 0.2)
        ]
        self.i = 0

    def update(self, measurement: Measurement, frame: MatLike) -> Setpoint:
        if self.i < len(self.waypoints):
            setpoint = Setpoint(self.waypoints[self.i], 0.0)
            if glm.distance(self.waypoints[self.i], measurement.position) < Planner.TOLERANCE:
                self.i += 1
            return setpoint
        else:
            return Setpoint(self.waypoints[-1], 0.0)
