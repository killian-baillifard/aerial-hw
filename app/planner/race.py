import os
import numpy as np
from pyglm import glm
from cv2.typing import MatLike
from overrides import overrides
from app import wrap
from app.io import Measurement
from app.io import Setpoint
from app.planner import Planner
from app.telemetry import Telemetry

ENABLE_TAS_REF_ANGLE = True

class Race(Planner):

    GATES_DIRECTORY = "gates"

    def __init__(self, speed: float = 1.0):
        super().__init__()
        self.speed = speed

    @overrides
    def reload(self) -> None:

        # Take first file found in gates directory
        gates_directory = os.path.join(Race.GATES_DIRECTORY)
        file_name = os.listdir(gates_directory)[0]
        file_path = os.path.join(gates_directory, file_name)

        # Load and parse gates coordinates
        # - Assume it's a csv with columns (Gate, x, y, z, theta, width, height)
        # - Assume first row is a header, pop it
        raw_csv_data = np.genfromtxt(file_path, delimiter=',', dtype=float, ndmin=2, filling_values=np.nan)
        if raw_csv_data.size == 0 and raw_csv_data.shape[0] > 1:
            # no sim data -> leave gates empty
            self.gates = []
            return
        raw_csv_data = raw_csv_data[1:]
        if not np.isnan(raw_csv_data[0]).any():
            positions: list[glm.vec3]   = [glm.vec3(col[1], col[2], col[3]) for col in raw_csv_data]
            if ENABLE_TAS_REF_ANGLE:
                yaws: list[float]       = [wrap(col[4] - np.pi) for col in raw_csv_data]
            else:
                yaws: list[float]       = [wrap(col[4]) for col in raw_csv_data]
            self.gates                  = [Setpoint(position, yaw) for position, yaw in zip(positions, yaws)]

            # Build trajectory starting from home position
            self.waypoints.clear()
            home_to_first_gate_direction = self.gates[0].position - Race.HOME_SETPOINT.position
            home_to_first_gate_yaw = np.atan2(home_to_first_gate_direction.y, home_to_first_gate_direction.x)
            self.waypoints.append(Setpoint(Race.HOME_SETPOINT.position, home_to_first_gate_yaw))

            # Append all gates twice (2 laps) with two points on either end of each gate
            for _ in range(2):
                for gate in self.gates:
                    normal = glm.vec3(
                        Race.APPROACH_DIST * np.cos(gate.yaw),
                        Race.APPROACH_DIST * np.sin(gate.yaw),
                        0.0
                    )
                    self.waypoints.append(Setpoint(gate.position - normal, gate.yaw))
                    self.waypoints.append(Setpoint(gate.position + normal, gate.yaw))

        # Return to home position at the end
        self.waypoints.append(Race.HOME_SETPOINT)

    @overrides
    def update(self, measurement: Measurement, frame: MatLike, flags: Telemetry.Flags, dt: float) -> Setpoint:

        # Waypoint list empty, go back to home position
        if(len(self.waypoints) == 0):
            return Race.HOME_SETPOINT
        
        # Until waypoint is reached, return interpolated setpoint
        setpoint, reached = Planner.reach(self.waypoints[0], measurement, self.speed)
        if not reached:
            return setpoint
        
        # When reached, call this function recursively to get next setpoint
        self.waypoints.pop(0)
        return self.update(measurement, frame, flags, dt)
