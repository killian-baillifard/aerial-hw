import os
import numpy as np
from queue import Queue

from gate_poses import load_gate_poses

DEBUG = True

APPROACH_DIST = 0.20  # [m] offset from gate center on each side of the gate

# Path to the gate-poses file written by the detection controller after Lap 1.
# Replace with the output path of detection_controller when that pipeline is complete.
MEASUREMENTS_NPY = os.path.join(
    os.path.dirname(__file__), '..',
    'saved_captures/2026-05-06-17-32-33_corner_measurements/measures.npy'
)

# Module-level singleton so main.py can call speed_controller.get_command()
_controller = None

def get_command(sensor_data, camera_data, dt):
    global _controller
    if _controller is None:
        _controller = SpeedController()
    return _controller.compute_command(sensor_data, camera_data, dt)


### Auxiliary functions

def _add_angles(lhs, rhs):
    """Adds two angles and wraps the result to the range [-pi, pi]."""
    result = lhs + rhs
    while result > np.pi:
        result -= 2 * np.pi
    while result < -np.pi:
        result += 2 * np.pi
    return result


### SpeedController

class SpeedController:
    def __init__(self):
        self.current_sensor_data = None
        self.starting_position = None

        # State machine
        self.state = "takeoff"  # takeoff, race, land

        # Load gate poses produced by detection controller (Lap 1)
        # Columns: [x, y, z, yaw, measured_width]  — drone coordinate frame
        self.gate_poses = load_gate_poses(MEASUREMENTS_NPY)

        # Pre-compute trajectory: 2 waypoints per gate (approach + exit).
        # To change traversal order, reorder self.gate_poses rows before calling
        # _build_trajectory, e.g.: self.gate_poses = self.gate_poses[[0,2,1,3,4]]
        self.trajectory = self._build_trajectory()

        # Waypoint follower
        self.position_tolerance = 0.05   # [m]
        self.yaw_tolerance = np.radians(5)  # [rad]
        self.current_setpoint = None
        self.setpoint_queue = Queue()

    # ------------------------------------------------------------------
    # Trajectory
    # ------------------------------------------------------------------

    def _build_trajectory(self) -> list:
        """
        Two waypoints per gate, in gate order:
          1. approach point — APPROACH_DIST before the gate (against the gate normal)
          2. exit point     — APPROACH_DIST after  the gate (along  the gate normal)

        Straight-line segments between gates are implicit: the waypoint follower
        flies directly from the exit of gate N to the approach of gate N+1.
        All coordinates are in drone frame (same as gate_poses output).
        """
        waypoints = []
        for x, y, z, yaw, _ in self.gate_poses:
            dx, dy = np.cos(yaw), np.sin(yaw)
            waypoints.append([x - APPROACH_DIST * dx, y - APPROACH_DIST * dy, z, yaw])
            waypoints.append([x + APPROACH_DIST * dx, y + APPROACH_DIST * dy, z, yaw])
        return waypoints

    # ------------------------------------------------------------------
    # Waypoint follower
    # ------------------------------------------------------------------

    def set_control_command(self):
        if self.current_setpoint is None:
            if self.setpoint_queue.empty():
                # Queue exhausted — hover in place
                return [
                    self.current_sensor_data['x_global'],
                    self.current_sensor_data['y_global'],
                    self.current_sensor_data['z_global'],
                    self.current_sensor_data['yaw'],
                ]
            self.current_setpoint = self.setpoint_queue.get()

        ctrl_cmd = self.current_setpoint[:4]

        dist = np.linalg.norm(
            np.array(self.current_setpoint[:3]) - np.array([
                self.current_sensor_data['x_global'],
                self.current_sensor_data['y_global'],
                self.current_sensor_data['z_global'],
            ])
        )
        dyaw = abs(_add_angles(self.current_setpoint[3], -self.current_sensor_data['yaw']))

        if dist < self.position_tolerance and dyaw < self.yaw_tolerance:
            self.current_setpoint = self.setpoint_queue.get() if not self.setpoint_queue.empty() else None

        return ctrl_cmd

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def compute_command(self, sensor_data, camera_data, dt):
        """Return [x, y, z, yaw] setpoint each tick."""
        self.current_sensor_data = sensor_data

        ### TAKEOFF ###
        if self.state == "takeoff":
            if self.starting_position is None:
                self.starting_position = [
                    sensor_data['x_global'],
                    sensor_data['y_global'],
                    sensor_data['z_global'],
                ]
            if sensor_data['z_global'] < 0.5:
                # Return directly — bypass waypoint queue until airborne
                return [sensor_data['x_global'], sensor_data['y_global'], 0.5, 0.0]
            self.state = "race"

        ### RACE ###
        elif self.state == "race":
            # Load trajectory into queue once on first tick after takeoff
            if self.setpoint_queue.empty() and self.current_setpoint is None:
                # Start from home, fly through all gates, return home
                self.setpoint_queue.put([
                    self.starting_position[0],
                    self.starting_position[1],
                    0.5,
                    0.0,
                ])
                for _ in range(2):
                    for wp in self.trajectory:
                        self.setpoint_queue.put(wp)
                self.setpoint_queue.put([
                    self.starting_position[0],
                    self.starting_position[1],
                    0.5,
                    0.0,
                ])

        ### LAND ###
        elif self.state == "land":
            pass

        return self.set_control_command()
