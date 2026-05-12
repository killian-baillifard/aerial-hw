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

    HOME_SETPOINT   = Setpoint(glm.vec3(-1.0, 0.0, 1.0), 0.0)
    GATES_DIRECTORY = "gates"
    
    APPROACH_DIST   = 0.20          # m

    POS_TOL         = 0.05          # m
    YAW_TOL         = np.pi / 12    # radians

    POS_INTERP_DIST = 0.5           # m
    YAW_INTERP_DIST = np.pi / 8     # radians

    def __init__(self):
        self.waypoints: list[Setpoint]  = []
        self.gates: list[Setpoint]      = []

    def reload_gates(self) -> None:

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

        # Build trajectory with two points on either end of each gate
        self.waypoints.clear()
        for gate in self.gates:
            normal = glm.vec3(
                RacePlanner.APPROACH_DIST * np.cos(gate.yaw),
                RacePlanner.APPROACH_DIST * np.sin(gate.yaw),
                0.0
            )
            self.waypoints.append(Setpoint(gate.position - normal, gate.yaw))
            self.waypoints.append(Setpoint(gate.position + normal, gate.yaw))
        
    def reach_next_wp(self, measurement: Measurement) -> tuple[Setpoint, bool]:

        # Compute position error
        setpoint = self.waypoints[0]
        error: glm.vec3 = setpoint.position - measurement.position
        dist_xy = glm.length(error.xy)

        # Compute error direction
        target_heading: float = np.atan2(error.y, error.x)
        heading_error: float = np.abs(wrap(target_heading - measurement.rotation.z))

        # Align heading before moving
        if dist_xy > 1.0 and heading_error > RacePlanner.YAW_TOL:
            return Setpoint(measurement.position, target_heading), False

        # Advance toward target
        direction   = glm.normalize(error.xy) if dist_xy > 1.0 else error.xy
        position    = glm.vec3(measurement.position.xy + direction, setpoint.position.z)
        loc_reached = dist_xy < RacePlanner.POS_TOL
        if not loc_reached:
            return Setpoint(position, target_heading), False

        # Correct yaw once on target
        pos_reached = glm.length(error) < RacePlanner.POS_TOL
        yaw_reached = np.abs(wrap(measurement.rotation.z - setpoint.yaw)) < RacePlanner.YAW_TOL
        reached     = pos_reached and yaw_reached
        return setpoint, reached

    @overrides
    def update(self, measurement: Measurement, frame: MatLike, flags: TelemetryFlags, dt: float) -> Setpoint:

        # Waypoint list empty, go back to home position
        if(len(self.waypoints) == 0):
            return RacePlanner.HOME_SETPOINT
        
        # Until waypoint is reached, return interpolated setpoint
        setpoint, reached = self.reach_next_wp(measurement)
        if not reached:
            return setpoint
        
        # When reached, call this function recursively to get next setpoint
        else:
            self.waypoints.pop(0)
            return self.update(measurement, frame, flags, dt)
