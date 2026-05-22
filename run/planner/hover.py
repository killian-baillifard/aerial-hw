"""
Built-in hover planner – no frame or YOLO needed.

Holds (0, 0, target_z) and returns to home on reload.
Overrides LAND_HEIGHT / MIN_HEIGHT / TOLERANCE to show the pattern.
"""

from pyglm import glm

from run import Flags
from run.planner import Planner
from run.telemetry import Measurement, Setpoint

class HoverPlanner(Planner):

    # Override defaults if desired
    LAND_HEIGHT = 0.10   # m
    TOLERANCE   = 0.12   # m
    MIN_HEIGHT  = 0.05   # m

    def __init__(self, target_z: float = 0.5) -> None:
        super().__init__()
        self.target_z = target_z
        self.waypoints.append(Setpoint(glm.vec3(0.0, 0.0, target_z), 0.0))

    # ------------------------------------------------------------------ Planner API
    
    def update(
        self,
        measurement: Measurement,
        frame,
        flags: Flags,
        dt: float,
    ) -> Setpoint:
        target = self.waypoints[-1]
        sp, _reached = Planner.reach(target, measurement, speed=0.25)
        # enforce floor
        sp = Setpoint(glm.vec3(sp.position.xy, self.safe_z(sp.position.z)), sp.yaw)
        return sp
