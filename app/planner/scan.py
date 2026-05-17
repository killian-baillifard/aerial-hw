import numpy as np
from pyglm import glm
from cv2.typing import MatLike
from overrides import overrides
from app import wrap
from app.io import Measurement
from app.io import Setpoint
from app.planner import Planner
from app.telemetry import Telemetry

class ScanPlanner(Planner):

    def __init__(self):
        super().__init__()

    @overrides
    def reload(self) -> None:

        # Fill waypoints with all scan positions
        self.waypoints.clear()
        self.waypoints.append(ScanPlanner.HOME_SETPOINT)

    @overrides
    def update(self, measurement: Measurement, frame: MatLike, flags: Telemetry.Flags, dt: float) -> Setpoint:

        # Waypoint list empty, go back to home position
        if(len(self.waypoints) == 0):
            return Planner.HOME_SETPOINT
        
        # Until waypoint is reached, return interpolated setpoint
        setpoint, reached = Planner.reach(self.waypoints[0], measurement)
        if not reached:
            return setpoint
        
        # TODO find and go though gate
        # Call self.gate_found_event(gate) to draw it on HUD
        
        # When reached, call this function recursively to get next setpoint
        self.waypoints.pop(0)
        return self.update(measurement, frame, flags, dt)
