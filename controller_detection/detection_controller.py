# TODO
### Critical (needed for minimal working version)
# [Oskar]  Add DetectionController.triangulate(candidate) wrapper 
#          → calls candidate.triangulate(), stores result
# [Oskar]  Add DetectionController.validate_gate_pos(corners_world) 
#          → calls validate_gate_pos(), returns Gate or None
# [Oskar]  Implement validate_gate_pos(): check planarity, aspect ratio, 
#          min/max side length of triangulated corners
# []       Define SEARCH waypoint pattern (e.g. expanding square or 
#          fixed sweep trajectory at constant altitude)
# [Oskar]  Implement SEARCH → PASS_GATE transition once ≥1 gate confirmed
#          (select nearest / highest-confidence gate as target)
#          gate selection logic if multiple candidates are confirmed
# [Oskar]  Implement PASS_GATE waypoints: compute gate center + normal 
#          from corners_world, set approach → through → exit waypoints

### Validation & geometry
# [Oskar]  Validate gate position / zone in world frame (expected arena bounds)
# [Oskar]  cornerSubPix refinement after YOLO detection for sub-pixel accuracy

### Debug & tuning
# [Oskar]  Implement DEBUG visualization: draw tracked corners, candidate 
#          bounding boxes, state label, and triangulated gate overlay on frame
# Tune: matching_threshold, forget_time, redetect_timeout, 
#       track_error_threshold, position_tolerance
# Consider adapting search pattern based on candidate distribution 
#      (e.g. turn toward cluster of unconfirmed candidates)

### Search pattern
# Define search pattern logic in SEARCH state to explore the environment effectively
# Search pattern in dependence of observations? (e.g. turn toward cluster of unconfirmed candidates)

### Optional / polish
# Classical vision fallback if YOLO confidence is low
# Final confirmation sweep before committing to PASS_GATE 
#      (re-detect gate from close range to verify position)
# Gate ordering / sequencing if multiple gates are confirmed

### Robustness / fallback
# Implement fallback if no gate detected after N seconds in SEARCH 
#      (e.g. spiral outward, change altitude)
# Add try/except around YOLO model load with clear error message

import cv2
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Optional
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

class DroneState(Enum):
    TAKEOFF = auto()
    SEARCH = auto()
    PASS_GATE = auto()
    LAND = auto()  

class CornerID(Enum):
    BL = auto()  # bottom-left
    TL = auto()  # top-left
    TR = auto()  # top-right
    BR = auto()  # bottom-right

CORNER_ID_MAP = {
    0: CornerID.BL,
    1: CornerID.TL,
    2: CornerID.TR,
    3: CornerID.BR,
}

# Output of detect/track — pure vision, no pose
@dataclass
class CornerDetection:
    corner_id: CornerID
    uv: np.ndarray
    conf: float

# One corner detection in one frame
@dataclass
class CornerObservation:
    corner_id: CornerID
    uv: np.ndarray         # 2D pixel coordinates
    conf: float            # confidence of the detection
    timestamp: float      # seconds
    P: np.ndarray          # 3x4 projection matrix at that frame
    drone_xyz: np.ndarray  # drone position in world frame at that frame
    drone_rpy: np.ndarray  # drone orientation (roll, pitch, yaw) in world frame at that frame

class CandidateState(Enum):
    TRACKING = auto()      # being actively detected/tracked, accumulating observations
    TRIANGULATING = auto() # lost from view, attempt triangulation + validation
    CONFIRMED = auto()     # passed geometric validation, promoted to Gate
    REJECTED = auto()      # failed validation, discard

# A candidate gate being tracked across frames
@dataclass
class GateCandidate:
    # corner_id → list of observations across frames
    observations: Dict[CornerID, list[CornerObservation]] = field(default_factory=dict)
    last_tracked_pts: Optional[Dict[CornerID, np.ndarray]] = field(default=None, repr=False)
    last_tracked_timestamp: Optional[float] = field(default=None, repr=False)
    state: CandidateState = CandidateState.TRACKING
    corners_world: Optional[Dict[CornerID, np.ndarray]] = field(default=None)  # filled after triangulation
    _last_seen: Optional[CornerObservation] = field(default=None, repr=False, init=False)
    
    def add(self, obs: CornerObservation):
        self.observations.setdefault(obs.corner_id, []).append(obs)
        if self._last_seen is None or obs.timestamp > self._last_seen.timestamp:
            self._last_seen = obs

    @property
    def last_seen(self) -> Optional[CornerObservation]:
        return self._last_seen

    def triangulate(self) -> Optional[Dict[CornerID, np.ndarray]]:
        """Returns world positions for corners with enough observations."""
        result = {}
        for corner_id, obs_list in self.observations.items():
            if len(obs_list) < 2:
                continue  # need at least 2 views
            pts  = np.array([o.uv for o in obs_list], dtype=np.float64).T  # (2, N)
            Ps   = [o.P for o in obs_list]
            result[corner_id] = self._triangulate_dlt(pts, Ps)
        return result if result else None

    def _triangulate_dlt(self, pts: np.ndarray, Ps: list[np.ndarray]) -> np.ndarray:
        """Multi-view DLT triangulation. pts: (2, N), Ps: list of N 3x4 matrices."""
        A = []
        for uv, P in zip(pts.T, Ps):
            u, v = uv
            A.append(u * P[2] - P[0])
            A.append(v * P[2] - P[1])
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        return (X[:3] / X[3])

# Final output — only what you need after validation
@dataclass
class Gate:
    corners_world: Dict[CornerID, np.ndarray]  # 3D world positions

### Auxiliary functions

def _add_angles(lhs, rhs):
    """Adds two angles and wraps the result to the range [-pi, pi]"""
    result = lhs + rhs
    while result > np.pi:
        result -= 2 * np.pi
    while result < -np.pi:
        result += 2 * np.pi
    return result

def _euler2rotmat(euler_angles):
    """
    Inputs:
        euler_angles: A list of 3 Euler angles [roll, pitch, yaw] in radians
    Outputs:
        R: A 3x3 numpy array that represents the rotation matrix of the euler angles
    """

    R = np.eye(3)

    roll = euler_angles[0]
    pitch = euler_angles[1]
    yaw = euler_angles[2]

    R_roll = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]], dtype=np.float64)

    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]], dtype=np.float64)

    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]], dtype=np.float64)

    R = R_yaw @ R_pitch @ R_roll

    return R

### DetectionController

class DetectionController:
    def __init__(self):
        self.starting_position = None
        self.current_drone_xyz = None
        self.current_drone_rpy = None
        self.current_timestamp = float(0.0) # seconds

        # State Machine
        self.state: DroneState = DroneState.TAKEOFF

        # Detection
        self.forget_time: float = 1.0 # seconds
        self.matching_threshold: float = 20.0 # pixels, for associating detections to candidates
        self.model = YOLO('detection_model/models/yolov8s_v2_r1/weights/best.pt')
        self.gate_candidates: dict[int, GateCandidate] = {} # dict of GateCandidate objects
        self._next_candidate_id: int = 0
        self.detected_gates: dict[int, GateCandidate] = {} # dict of confirmed Gate objects

        # Tracking
        self.redetect_timeout: float = 0.2 # seconds
        self.track_error_threshold = 10.0 # pixels
        self.last_detection_timestamp: float = float(0.0)
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_frame_timestamp: Optional[float] = None

        # Camera
        self.camera_rotation = np.array([
            [0, -1,  0], 
            [0,  0, -1], 
            [1,  0,  0]
        ]) # rotation from camera to body frame (Zcam = Xdrone, Xcam = -Ydrone, Ycam = -Zdrone)
        self.camera_translation = np.array([0.03, 0.0, -0.01]) # translation from body to camera frame (x forward, y left, z up)
        self.K = np.array([
            [161.01392228,   0.0, 150.0],
            [  0.0, 161.01392228, 150.0],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float64)
        self.focal_length = 161.013922282 # focal length in pixels (calculated from FOV and image size)

        # Control Command
        self.position_tolerance = 0.05 # [m]
        self.yaw_tolerance = np.radians(5) # [rad]
        self.set_path = False # whtether waypoints are already set or not, to prevent re-setting them every frame
        self.current_setpoint = None
        self.setpoint_queue = Queue()

        pass

    def _get_drone_pose(self, sensor_data):
        # Extract drone position and orientation from sensor data
        drone_xyz = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])
        drone_rpy = np.array([sensor_data['roll'], sensor_data['pitch'], sensor_data['yaw']])

        return drone_xyz, drone_rpy

    def _compute_world_to_camera_projection(self, xyz, rpy):
        # drone_position: (3,) in world frame
        # drone_orientation: (3,) roll, pitch, yaw in radians

        R_cb = self.camera_rotation
        t_bc = self.camera_translation

        R_wb = _euler2rotmat(rpy)
        R_bw = R_wb.T
        R_cw = R_cb @ R_bw

        t_wb = np.array(xyz)
        t_wc = R_wb @ t_bc + t_wb
        t_cw = -R_cw @ t_wc # get world origin in camera frame

        P = self.K @ np.hstack((R_cw, t_cw.reshape(3,1)))

        return P

    def detect(self, camera_data: np.ndarray) -> list[list[CornerDetection]]:
        """
        Returns one list of CornerDetections per gate candidate found.
        Runs YOLO → cornerSubPix → optional classical fallback.
        No pose involved.
        """

        detected_gates = self.yolo_detect(camera_data)

        # cornerSubPix

        # classical vision pipeline

        return detected_gates

    def yolo_detect(self, camera_data) -> list[list[CornerDetection]]:

        # Run inference on the camera data
        # Note: iou=0.7 allows bounding boxes to heavily overlap without being filtered out
        results = self.model.predict(source=camera_data, conf=0.5, iou=0.7)

        # Extract coordinates and confidences
        detected_gates = []
        keypoints = results.keypoints
        if keypoints is not None and keypoints.xy.numel() > 0:
            for i in range(len(keypoints.xy)):
                corners = []

                coords = keypoints.xy[i].cpu().numpy()  # Array of 4 (x,y) corners
                confs = keypoints.conf[i].cpu().numpy() # Array of 4 confidences

                for j in range(4):
                    pred_x, pred_y = int(coords[j][0]), int(coords[j][1])
                    corners.append(CornerDetection(
                        corner_id=CORNER_ID_MAP[j],
                        uv=np.array([pred_x, pred_y]),
                        conf=confs[j]
                    ))

                detected_gates.append(corners)

        return detected_gates

    def track(self, curr_frame: np.ndarray) -> int:
        """
        Returns the number of successfully tracked CornerDetections.
        Lost points are simply absent from the dict.
        """
        num_tracked = 0

        if self.prev_frame is None:
            return num_tracked

        for _, candidate in self.gate_candidates.items():
            if candidate.last_tracked_pts is None:
                continue

            # Skip if this candidate wasn't updated last frame — prev_frame is wrong reference
            if candidate.last_tracked_timestamp != self.prev_frame_timestamp:
                continue

            corner_ids = list(candidate.last_tracked_pts.keys())
            prev_pts = np.array(
                [candidate.last_tracked_pts[cid] for cid in corner_ids],
                dtype=np.float32
            ).reshape(-1, 1, 2)  # LK expects (N, 1, 2)

            new_pts, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, curr_frame, prev_pts, None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )

            # Tracking successful - update candidate
            tracked_pts = {}
            for i, cid in enumerate(corner_ids):
                if status[i, 0] and error[i, 0] < self.track_error_threshold:
                    tracked_pts[cid] = new_pts[i, 0]  # note: new_pts[i,0] not new_pts[i] — shape (2,) not (1,2)

            if tracked_pts:
                candidate.last_tracked_pts = tracked_pts
                candidate.last_tracked_timestamp = self.current_timestamp
                num_tracked += 1

        return num_tracked

    def associate_and_update(self, detections, P: np.ndarray):
        """Associates new detections to existing candidates based on proximity in image space.
        Updates candidates with new detections or creates new candidates as needed."""
        
        for detection in detections if isinstance(detections, list) else []:
            best_id, best_dist = self._find_best_candidate(detection)
            if best_id is not None and best_dist < self.matching_threshold:
                candidate = self.gate_candidates[best_id]
            else:
                candidate = GateCandidate()
                self.gate_candidates[self._next_candidate_id] = candidate
                self._next_candidate_id += 1
            self._add_detections_to_candidate(candidate, detection, P)

    def _find_best_candidate(self, detection: list[CornerDetection]):
        det_pts = {cd.corner_id: cd.uv for cd in detection}
        best_id, best_dist = None, float('inf')

        for cand_id, candidate in self.gate_candidates.items():
            if not candidate.last_tracked_pts:
                continue
            # Only compare corners that are present in both detection and candidate
            common = det_pts.keys() & candidate.last_tracked_pts.keys()
            if not common:
                continue
            dist = np.mean([np.linalg.norm(det_pts[k] - candidate.last_tracked_pts[k]) for k in common])
            if dist < best_dist:
                best_dist, best_id = dist, cand_id

        return best_id, best_dist

    def _add_detections_to_candidate(self, candidate: GateCandidate, detection: list[CornerDetection], P: np.ndarray):
        """Fuses pose into detections and stores as observations. Updates snapshot."""
        new_snapshot = {}
        for cd in detection:
            obs = CornerObservation(
                corner_id=cd.corner_id,
                uv=cd.uv,
                conf=cd.conf,
                timestamp=self.current_timestamp,
                P=P,
                drone_xyz=self.current_drone_xyz,
                drone_rpy=self.current_drone_rpy,
            )
            candidate.add(obs)
            new_snapshot[cd.corner_id] = cd.uv
        candidate.last_tracked_pts = new_snapshot
        candidate.last_tracked_timestamp = self.current_timestamp

    def validate_gate_pos(self, world_gate_pos: Dict[CornerID, np.ndarray]) -> bool:
        """
        Runs geometric validation on the candidate's triangulated corners.
        """
        return False # placeholder

    def _clear_waypoints(self) -> None:
        """Clears current setpoint and queued waypoints."""
        self.current_setpoint = None
        while not self.setpoint_queue.empty():
            self.setpoint_queue.get()

    def _no_waypoints(self) -> bool:
        """Returns True if there are no current or queued waypoints."""
        return self.current_setpoint is None and self.setpoint_queue.empty()

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
        self.current_timestamp += dt
        self.current_drone_xyz, self.current_drone_rpy = self._get_drone_pose(sensor_data)  # your FC/odometry

        ### TAKEOFF ###
        if self.state == DroneState.TAKEOFF:

            # set starting position for landing
            if self.starting_position is None:
                self.starting_position = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]

            # Takeoff logic
            if not self.set_path:
                self._clear_waypoints()
                self.setpoint_queue.put([sensor_data['x_global'], sensor_data['y_global'], 1, 0.0]) # takeoff point
                self.set_path = True

            if sensor_data['z_global'] > 0.9:
                self.state = DroneState.SEARCH
                self.set_path = False # reset for next state

        ### SEARCH ###
        elif self.state == DroneState.SEARCH:

            # 1) Update candidate states based on timeouts and triangulation results
            delete_ids = []
            for i, candidate in self.gate_candidates.items():
                if candidate.last_seen is None:
                    continue
                age = self.current_timestamp - candidate.last_seen.timestamp

                if candidate.state == CandidateState.TRACKING and age > self.forget_time:
                    candidate.state = CandidateState.TRIANGULATING

                if candidate.state == CandidateState.TRIANGULATING:
                    candidate.corners_world = candidate.triangulate()
                    gate = self.validate_gate_pos(candidate.corners_world)
                    candidate.state = CandidateState.CONFIRMED if gate else CandidateState.REJECTED

                if candidate.state == CandidateState.CONFIRMED:
                    self.detected_gates[i] = candidate
                    delete_ids.append(i) # remove from candidates, now in detected_gates

                ### EXIT SEARCH STATE POSSIBLE ###

                if candidate.state == CandidateState.REJECTED:
                    delete_ids.append(i)

            for i in delete_ids:
                del self.gate_candidates[i]

            # 2) Run detection/tracking to update gate candidates
            if self.current_timestamp - self.last_detection_timestamp > self.redetect_timeout:
                # Redetect
                detections = self.detect(camera_data)
                P = self._compute_world_to_camera_projection(self.current_drone_xyz, self.current_drone_rpy)
                self.associate_and_update(detections, P)
                self.last_detection_timestamp = self.current_timestamp
            else:
                # Track
                self.track(camera_data)

            # 3) (Optionaly) Adapt search pattern or hover point based on candidate distribution
            if not self.set_path:
                self._clear_waypoints()
                # SET WAYPOINTS
                self.set_path = True

            # 4) Update prev_frame for tracking
            self.prev_frame = camera_data.copy()
            self.prev_frame_timestamp = self.current_timestamp

            pass

        ### PASS GATE ###
        elif self.state == DroneState.PASS_GATE:

            if not self.set_path:
                self._clear_waypoints()
                # SET WAYPOINTS
                self.set_path = True

            ### further passing gate logic ###

            if self._no_waypoints():
                print("Passed gate, no more waypoints.")
                self.state = DroneState.LAND
                self.set_path = False # reset for next state

            pass

        ### LAND ###
        elif self.state == DroneState.LAND:

            if not self.set_path:
                self._clear_waypoints()
                self.setpoint_queue.put([self.starting_position[0], self.starting_position[1], 1.0, 0.0]) # hover above starting point
                self.setpoint_queue.put([self.starting_position[0], self.starting_position[1], self.starting_position[2], 0.0]) # landing point
                self.set_path = True

            if self._no_waypoints():
                print("Landed, no more waypoints.")

            pass


        control_command = self.set_control_command()

        return control_command
