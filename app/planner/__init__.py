import numpy as np
from pyglm import glm
from abc import ABC, abstractmethod
from cv2.typing import MatLike
from app.io import Measurement
from app.io import Setpoint
from app import wrap
from app.telemetry import Telemetry

class Planner(ABC):

    HOME_SETPOINT   = Setpoint(glm.vec3(-1.0, 0.0, 1.0), 0.0)
    APPROACH_DIST   = 0.20          # m
    POS_TOL         = 0.15          # m
    YAW_TOL         = np.pi / 6    # radians

    def __init__(self):
        self.waypoints: list[Setpoint]  = []
        self.gates: list[Setpoint]      = []

    def reach(setpoint: Setpoint, measurement: Measurement) -> tuple[Setpoint, bool]:
        """
        Interpolate a new setpoint between current measurement and target setpoint

        Parameters
        ----------

        setpoint : Setpoint
            Target position and yaw
        measurement : Measurement
            Last measured position and yaw

        Returns
        -------

        interpolated : Setpoint
            Intermediate position for the PID to reach
        reached : bool
            True if the target position and yaw is reached
        """

        # Compute position error
        error: glm.vec3 = setpoint.position - measurement.position
        dist_xy = glm.length(error.xy)

        # Compute error direction
        target_heading: float = np.atan2(error.y, error.x)
        heading_error: float = np.abs(wrap(target_heading - measurement.rotation.z))

        # Align heading before moving
        if dist_xy > 1.0 and heading_error > Planner.YAW_TOL:
            return Setpoint(measurement.position, target_heading), False

        # Advance toward target
        direction   = glm.normalize(error.xy) if dist_xy > 1.0 else error.xy
        position    = glm.vec3(measurement.position.xy + direction, setpoint.position.z)
        loc_reached = dist_xy < Planner.POS_TOL
        if not loc_reached:
            return Setpoint(position, target_heading), False

        # Correct yaw once on target
        pos_reached = glm.length(error) < Planner.POS_TOL
        yaw_reached = np.abs(wrap(measurement.rotation.z - setpoint.yaw)) < Planner.YAW_TOL
        reached     = pos_reached and yaw_reached
        return setpoint, reached

    @abstractmethod
    def reload(self) -> None:
        """
        Reload planner to its initial state
        """
        pass

    @abstractmethod
    def update(self, measurement: Measurement, frame: MatLike, flags: Telemetry.Flags, dt: float) -> Setpoint:
        """
        Run one frame on the planner state machine

        Parameters
        ----------
        measurement : Measurement
            - timestamp
            - position (x, y, z)
            - rotation (x, y, z) = (roll, pitch, yaw)
            - battery (0% - 100%)
        frame : MatLike
            BGR frame (height, width, color) = (244, 324, 3)
        flags : TelemetryFlags
            Indicate if measurement and / or frame argument contains new measurements
        dt : float
            Time delta since last call

        Returns
        -------
        setpoint : Setpoint
            - position (x, y, z)
            - yaw
        """
        pass
