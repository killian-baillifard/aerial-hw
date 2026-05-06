from pyglm import glm
from cv2.typing import MatLike
from app.telemetry.measurement import Measurement
from app.inputs.setpoint import Setpoint

class Planner:

    def __init__(self) -> None:
        pass

    def update(measurement: Measurement, frame: MatLike) -> Setpoint:
        return Setpoint(glm.vec3(0.0, 0.0, 0.0), 0.0)
