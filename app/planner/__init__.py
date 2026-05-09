from abc import ABC, abstractmethod
from cv2.typing import MatLike
from app.io import Measurement
from app.io import Setpoint
from app.telemetry import TelemetryFlags

class Planner(ABC):

    @abstractmethod
    def update(self, measurement: Measurement, frame: MatLike, flags: TelemetryFlags, dt: float) -> Setpoint:
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
