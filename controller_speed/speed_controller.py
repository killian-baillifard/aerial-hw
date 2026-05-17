import os
import threading
import numpy as np
from queue import Queue

from csv_gates import load_gates_csv

DEBUG = True

APPROACH_DIST    = 0.20   # [m] offset from gate center on each side of the gate
LAND_HOVER_Z     = 0.30   # [m] height above home pad before starting descent
LAND_DESCENT_RATE = 0.20  # [m/s] controlled descent speed

# Path to competition-provided gate CSV (Gate,x,y,z,theta,width,height).
GATES_CSV = os.path.join(os.path.dirname(__file__), 'gates_info.csv')

# Home pad position in drone/Lighthouse frame — used to resolve first gate yaw.
HOME_XY = (0.0, 0.0)

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

        # State machine: takeoff -> hover -> race -> land
        self.state = "takeoff"
        self._trajectory_loaded = False

        # Keyboard trigger: press ENTER in the terminal to start the race
        self._race_trigger = threading.Event()
        threading.Thread(target=self._keyboard_listener, daemon=True).start()

        # Load gate poses from competition CSV.
        # Columns: [x, y, z, yaw, width, height] — drone/Lighthouse coordinate frame
        self.gate_poses = load_gates_csv(GATES_CSV, home_xy=HOME_XY)

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
    # Keyboard listener
    # ------------------------------------------------------------------

    def _keyboard_listener(self):
        input("\n[SpeedController] Hovering — press ENTER to start race...\n")
        self._race_trigger.set()
        print("[SpeedController] Race started!")

    # ------------------------------------------------------------------
    # Trajectory
    # ------------------------------------------------------------------

    def _build_trajectory(self) -> list:
        """
        Two waypoints per gate, in gate order:
          1. approach point — APPROACH_DIST before the gate (against gate normal)
          2. exit point     — APPROACH_DIST after  the gate (along  gate normal)

        All coordinates in drone/Lighthouse frame.
        """
        waypoints = []
        for x, y, z, yaw, *_ in self.gate_poses:
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
                return [sensor_data['x_global'], sensor_data['y_global'], 0.5, 0.0]
            self.state = "hover"

        ### HOVER — hold position, wait for ENTER ###
        elif self.state == "hover":
            if self._race_trigger.is_set():
                self.state = "race"
            else:
                return [
                    self.starting_position[0],
                    self.starting_position[1],
                    0.5,
                    0.0,
                ]

        ### RACE ###
        elif self.state == "race":
            if not self._trajectory_loaded:
                self._trajectory_loaded = True
                hx, hy = self.starting_position[0], self.starting_position[1]
                # Z of last gate exit — fly home at this height, then descend
                cruise_z = self.trajectory[-1][2]

                for _ in range(2):
                    for wp in self.trajectory:
                        self.setpoint_queue.put(wp)
                # Fly horizontally to home at cruise height (no descent during transit)
                self.setpoint_queue.put([hx, hy, cruise_z, 0.0])
                # Descend to low hover above home pad before landing
                self.setpoint_queue.put([hx, hy, LAND_HOVER_Z, 0.0])

            result = self.set_control_command()
            # All waypoints consumed — start landing
            if self.setpoint_queue.empty() and self.current_setpoint is None:
                self.state = "land"
            return result

        ### LAND — descend over home pad ###
        elif self.state == "land":
            hx, hy = self.starting_position[0], self.starting_position[1]
            z = sensor_data['z_global']
            if z > 0.05:
                return [hx, hy, max(0.0, z - LAND_DESCENT_RATE * dt), 0.0]
            return [hx, hy, 0.0, 0.0]

        return self.set_control_command()
