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
from app.planner.racer_z_params import compute_z_shift

class RacerPlanner(Planner):

    GATES_DIRECTORY = "gates"

    @overrides
    def reload(self) -> None:

        # Take first file found in gates directory
        gates_directory = os.path.join(RacerPlanner.GATES_DIRECTORY)
        file_name = os.listdir(gates_directory)[0]
        file_path = os.path.join(gates_directory, file_name)

        # Load and parse gates coordinates
        # - Assume it's a csv with columns (Gate, x, y, z, theta, width, height)
        # - Assume first row is a header, pop it
        raw_csv_data: np.ndarray    = np.genfromtxt(file_path, delimiter=',')[1:]
        positions: list[glm.vec3]   = [glm.vec3(row[1], row[2], row[3]) for row in raw_csv_data]
        yaws: list[float]           = [wrap(row[4]) for row in raw_csv_data]
        self.gates                  = [Setpoint(position, yaw) for position, yaw in zip(positions, yaws)]

        # Build trajectory starting from home position
        self.waypoints.clear()
        home_to_first_gate_direction = self.gates[0].position - RacerPlanner.HOME_SETPOINT.position
        home_to_first_gate_yaw = np.atan2(home_to_first_gate_direction.y, home_to_first_gate_direction.x)
        self.waypoints.append(Setpoint(RacerPlanner.HOME_SETPOINT.position, home_to_first_gate_yaw))

        # Flatten gates for 2 laps, then build pre/center/post triplet per gate.
        # Pre and post z are offset by a small slope-proportional bias derived from
        # the height trend across each gate (prev -> next), matching the approach used
        # in my_assignment.py _build_race_gate_triplet.
        lapped_gates = self.gates * 2
        home_pos = RacerPlanner.HOME_SETPOINT.position

        for i, gate in enumerate(lapped_gates):
            prev_pos = lapped_gates[i - 1].position if i > 0          else home_pos
            next_pos = lapped_gates[i + 1].position if i < len(lapped_gates) - 1 else home_pos

            z_shift = compute_z_shift(
                [prev_pos.x, prev_pos.y, prev_pos.z],
                [next_pos.x, next_pos.y, next_pos.z]
            )

            normal = glm.vec3(
                RacerPlanner.APPROACH_DIST * np.cos(gate.yaw),
                RacerPlanner.APPROACH_DIST * np.sin(gate.yaw),
                0.0
            )
            pre_pos  = gate.position - normal
            post_pos = gate.position + normal

            self.waypoints.append(Setpoint(glm.vec3(pre_pos.x,  pre_pos.y,  gate.position.z - 0.5 * z_shift), gate.yaw))
            self.waypoints.append(Setpoint(gate.position, gate.yaw))
            self.waypoints.append(Setpoint(glm.vec3(post_pos.x, post_pos.y, gate.position.z + 0.5 * z_shift), gate.yaw))

        # Return to home position at the end
        self.waypoints.append(RacerPlanner.HOME_SETPOINT)

    @overrides
    def update(self, measurement: Measurement, frame: MatLike, flags: Telemetry.Flags, dt: float) -> Setpoint:

        # Waypoint list empty, go back to home position
        if(len(self.waypoints) == 0):
            return RacerPlanner.HOME_SETPOINT

        # Until waypoint is reached, return interpolated setpoint
        setpoint, reached = Planner.reach(self.waypoints[0], measurement)
        if not reached:
            return setpoint

        # When reached, call this function recursively to get next setpoint
        self.waypoints.pop(0)
        return self.update(measurement, frame, flags, dt)
