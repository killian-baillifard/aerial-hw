from overrides import override
from pyglm import glm
from cv2.typing import MatLike
from app.io import Measurement
from app.io import Setpoint
from app.planner import Planner
from app.telemetry import TelemetryFlags

class ExamplePlanner(Planner):

    TOLERANCE = 0.05 # 5 cm

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

    @override
    def update(self, measurement: Measurement, frame: MatLike, flags: TelemetryFlags, dt: float) -> Setpoint:

        if self.i < len(self.waypoints):
            setpoint = Setpoint(self.waypoints[self.i], 0.0)
            if glm.distance(self.waypoints[self.i], measurement.position) < ExamplePlanner.TOLERANCE:
                self.i += 1
            return setpoint
        else:
            return Setpoint(self.waypoints[-1], 0.0)
