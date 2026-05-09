from abc import ABC, abstractmethod
from cv2.typing import MatLike
from app.io import Measurement
from app.io import Setpoint

class Planner(ABC):

    @abstractmethod
    def update(self, measurement: Measurement, frame: MatLike, dt: float) -> Setpoint:
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

        Returns
        -------
        setpoint : Setpoint
            - position (x, y, z)
            - yaw
        """
        pass
