import os
import numpy as np
from pyglm import glm
from cv2.typing import MatLike
from overrides import overrides
from app import wrap
from app.io import Measurement
from app.io import Setpoint
from app.planner import Planner
from app.telemetry import TelemetryFlags

class RacePlanner(Planner):

    GATES_DIRECTORY = "gates"

    def __init__(self):
        super().__init__()

    @overrides
    def reload(self) -> None:

        # Take first file found in gates directory
        gates_directory = os.path.join(RacePlanner.GATES_DIRECTORY)
        file_name = os.listdir(gates_directory)[0]
        file_path = os.path.join(gates_directory, file_name)

        # Load and parse gates coordinates
        # - Assume it's a csv with columns (x, y, z, yaw)
        # - Assume first row is a header, pop it
        raw_csv_data: np.ndarray    = np.genfromtxt(file_path, delimiter=',')[1:]
        positions: list[glm.vec3]   = [glm.vec3(row[0], row[1], row[2]) for row in raw_csv_data]
        yaws: list[float]           = [wrap(row[3]) for row in raw_csv_data]
        self.gates                  = [Setpoint(position, yaw) for position, yaw in zip(positions, yaws)]

        # Build trajectory starting from home position
        self.waypoints.clear()
        home_to_first_gate_direction = self.gates[0].position - RacePlanner.HOME_SETPOINT.position
        home_to_first_gate_yaw = np.atan2(home_to_first_gate_direction.y, home_to_first_gate_direction.x)
        self.waypoints.append(Setpoint(RacePlanner.HOME_SETPOINT.position, home_to_first_gate_yaw))

        # Append all gates twice (2 laps) with two points on either end of each gate
        for _ in range(2):
            for gate in self.gates:
                normal = glm.vec3(
                    RacePlanner.APPROACH_DIST * np.cos(gate.yaw),
                    RacePlanner.APPROACH_DIST * np.sin(gate.yaw),
                    0.0
                )
                self.waypoints.append(Setpoint(gate.position - normal, gate.yaw))
                self.waypoints.append(Setpoint(gate.position + normal, gate.yaw))

        # Return to home position at the end
        self.waypoints.append(RacePlanner.HOME_SETPOINT)

    @overrides
    def update(self, measurement: Measurement, frame: MatLike, flags: TelemetryFlags, dt: float) -> Setpoint:

        # Waypoint list empty, go back to home position
        if(len(self.waypoints) == 0):
            return RacePlanner.HOME_SETPOINT
        
        # Until waypoint is reached, return interpolated setpoint
        setpoint, reached = Planner.reach(self.waypoints[0], measurement)
        if not reached:
            return setpoint
        
        # When reached, call this function recursively to get next setpoint
        self.waypoints.pop(0)
        return self.update(measurement, frame, flags, dt)
