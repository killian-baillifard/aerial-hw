import cv2
import numpy as np
from queue import Queue
from ultralytics import YOLO

DEBUG = True

# Module-level singleton so main.py can call detection_controller.get_command()
_controller = None

def get_command(sensor_data, camera_data, dt):
    global _controller
    if _controller is None:
        _controller = DetectionController()
    return _controller.compute_command(sensor_data, camera_data, dt)

### Data structures

BL = 0
TL = 1
TR = 2
BR = 3

### Auxiliary functions

def _add_angles(lhs, rhs):
    """Adds two angles and wraps the result to the range [-pi, pi]"""
    result = lhs + rhs
    while result > np.pi:
        result -= 2 * np.pi
    while result < -np.pi:
        result += 2 * np.pi
    return result

### DetectionController

class DetectionController:
    def __init__(self):
        self.current_sensor_data = None
        self.starting_position = None

        # State Machine
        self.state = "takeoff" # takeoff, search, pass_gate, land

        # Detection
        self.model = YOLO('detection_model/models/yolov8s_v2_r1/weights/best.pt')

        # Control Command
        self.position_tolerance = 0.05 # [m]
        self.yaw_tolerance = np.radians(5) # [rad]
        self.current_setpoint = None
        self.setpoint_queue = Queue()

        pass

    def detect_gates(self, camera_data):

        # Run inference on the camera data
        # Note: iou=0.7 allows bounding boxes to heavily overlap without being filtered out
        results = self.model.predict(source=camera_data, conf=0.5, iou=0.7)

        # Extract coordinates and confidences
        keypoints = results.keypoints
        if keypoints is not None and keypoints.xy.numel() > 0:
            for i in range(len(keypoints.xy)):
                coords = keypoints.xy[i].cpu().numpy()  # Array of 4 (x,y) corners
                confs = keypoints.conf[i].cpu().numpy() # Array of 4 confidences

                print(f"\n--- Gate {i+1} ---")
                corner_names = ["Bottom-Left", "Top-Left", "Top-Right", "Bottom-Right"]

                for j in range(4):
                    pred_x, pred_y = int(coords[j][0]), int(coords[j][1])
                    print(f"{corner_names[j]}: ({pred_x}, {pred_y}), Conf: {confs[j]:.2f}")

        pass

    def set_control_command(self):

        if self.current_setpoint is None:
            if self.setpoint_queue.empty():
                # If no setpoint is set, just hover in place
                return [self.current_sensor_data['x_global'], self.current_sensor_data['y_global'], self.current_sensor_data['z_global'], self.current_sensor_data['yaw']]
            else:
                # Otherwise, get next waypoint from queue
                self.current_setpoint = self.setpoint_queue.get()

        # Set control command to current setpoint
        ctrl_cmd = [self.current_setpoint[0], self.current_setpoint[1], self.current_setpoint[2], self.current_setpoint[3]]

        # Check if we have reached the current waypoint, if so, get next waypoint from queue
        dist_xyz_from_setpoint = np.linalg.norm(np.array(self.current_setpoint[:3]) - np.array([self.current_sensor_data['x_global'], self.current_sensor_data['y_global'], self.current_sensor_data['z_global']]))
        diff_yaw_from_setpoint = abs(_add_angles(self.current_setpoint[3], -self.current_sensor_data['yaw']))
        reached_position = dist_xyz_from_setpoint < self.position_tolerance
        reached_yaw = diff_yaw_from_setpoint < self.yaw_tolerance

        if reached_position and reached_yaw:
            self.current_setpoint = self.setpoint_queue.get() if not self.setpoint_queue.empty() else None

        return ctrl_cmd

    def compute_command(self, sensor_data, camera_data, dt):
        """Process camera data and sensor data to compute control command"""
        self.current_sensor_data = sensor_data

        ### TAKEOFF ###
        if self.state == "takeoff":

            # set starting position for landing
            if self.starting_position is None:
                self.starting_position = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]

            # Takeoff logic
            if sensor_data['z_global'] < 0.5:
                control_command = [sensor_data['x_global'], sensor_data['y_global'], 0.5, 0.0]
            else:
                self.state = "search"
                control_command = [sensor_data['x_global'], sensor_data['y_global'], 0.5, 0.0]

        ### SEARCH ###
        elif self.state == "search":

            pass

        ### PASS GATE ###
        elif self.state == "pass_gate":

            pass

        ### LAND ###
        elif self.state == "land":

            pass



        control_command = self.set_control_command()

        return control_command
