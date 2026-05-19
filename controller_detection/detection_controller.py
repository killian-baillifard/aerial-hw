### Search pattern
# TODO:1 Define SEARCH waypoint pattern (e.g. expanding square or fixed sweep trajectory at constant altitude). How can we do this effectively?
# TODO:15 Define exit condition for search state
# (TODO:8) Which detection do we select as final gate (retriangulation)? If multiple gates are confirmed are we going straight fo all of them?
# ((TODO:4)) Consider adapting search pattern based on candidate distribution (e.g. turn toward cluster of unconfirmed candidates)
# ((TODO:9)) In PASS_GATE, assert we are passing through the gate we intended (Final confirmation sweep). Adapt path if necessary.

### Camera
# TODO:19 Intrinsic validation / calibration

### Debug & tuning
# (TODO:3) Tune parameters: e.g. matching_threshold, forget_time, redetect_timeout, track_error_threshold, position_tolerance

### Validation & geometry
# (TODO:5) check get_zone function for correctness
# ((TODO:6)) understand validation function
# ((TODO:7)) check for out of bounds flight commands

### Robustness / fallback
# TODO:21 handle asserts
# (TODO:16) Remove detected gates after pass through
# TODO:18 Triangulation: add minimum baseline check!!!
# (TODO:17) Triangulation: consider MAX N points per corner
# (TODO:18) Cap number of observations per corner to avoid unbounded memory growth
# (TODO:14) Continuous refinement of gate position / detection (however once a gate is confirmed, we have to handle changes more carefully to avoid oscillation)
# ((TODO:10)) Implement fallback if no gate detected after N seconds in SEARCH (e.g. spiral outward, change altitude)
# ((TODO:11)) Add try/except around YOLO model load with clear error message
# ((TODO:13)) Classical vision fallback if YOLO confidence is low
# ((TODO:12)) Recalculate missing corners
# ((TODO:20)) Doing dtection with yolo on some frames twice at the moment

import cv2
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Optional
from queue import Queue
from ultralytics import YOLO
import hashlib
import colorsys

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
    triangulation_confidence = 0.9
    # corner_id → list of observations across frames
    observations: Dict[CornerID, list[CornerObservation]] = field(default_factory=dict)
    last_tracked_pts: Optional[Dict[CornerID, np.ndarray]] = field(default=None, repr=False)
    last_tracked_timestamp: Optional[float] = field(default=None, repr=False)
    state: CandidateState = CandidateState.TRACKING
    corners_world: Optional[Dict[CornerID, np.ndarray]] = field(default=None)  # filled after triangulation
    val_conf: Optional[float] = field(default=None) # confidence of the validated position
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
            obs_list_filtered = [obs for obs in obs_list if obs.conf >= self.triangulation_confidence]
            pts  = np.array([o.uv for o in obs_list_filtered], dtype=np.float64).T  # (2, N)
            Ps   = [o.P for o in obs_list_filtered]
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

### Auxiliary functions

def _add_angles(lhs, rhs):
    """Adds two angles and wraps the result to the range [-pi, pi]"""
    return (lhs + rhs + np.pi) % (2 * np.pi) - np.pi

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

def _hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCV uses BGR

def _generate_color_dict(data):
    color_dict = {}

    for key in data.keys():
        # 1. Convert the key to a deterministic string and encode to bytes
        key_bytes = str(key).encode('utf-8')
        
        # 2. Generate a fixed-seed PRF hash using MD5
        hash_hex = hashlib.md5(key_bytes).hexdigest()
        hash_int = int(hash_hex, 16)
        
        # 3. Map the hash integer to a unique hue on a 360-degree color wheel (0.0 to 1.0)
        hue = (hash_int % 360) / 360.0
        
        # Keep saturation and value constant for highly visible colors
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)

        # Convert RGB floats to hex color
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )

        color_dict[key] = _hex_to_bgr(hex_color)

    return color_dict

### DetectionController

class DetectionController:
    def __init__(self):
        self.starting_position = None
        self.current_drone_xyz = None
        self.current_drone_rpy = None
        self.current_timestamp = float(0.0) # milliseconds

        # State Machine
        self.state: DroneState = DroneState.TAKEOFF

        # Detection
        self.forget_time: float = 1000.0 # ms
        self.matching_threshold: float = 100.0 # pixels, for associating detections to candidates
        self.yolo_conf_threshold: float = 0.5 # confidence threshold for YOLO detections
        self.model = YOLO('detection_model/models/yolov8n_v2rgb_r1/weights/best.pt')
        self.gate_candidates: dict[int, GateCandidate] = {} # dict of GateCandidate objects
        self._next_candidate_id: int = 0
        self.detected_gates: dict[int, GateCandidate] = {} # dict of confirmed Gate objects

        # Tracking
        self.redetect_timeout: float = 0.0 # ms
        self.track_error_threshold = 20.0 # pixels
        self.last_detection_timestamp: float = float(0.0)
        self.prev_frame: Optional[np.ndarray] = None # Gray-scale image of previous frame for LK tracking
        self.prev_frame_timestamp: Optional[float] = None

        # Camera
        self.camera_rotation = np.array([
            [0, -1,  0], 
            [0,  0, -1], 
            [1,  0,  0]
        ]) # rotation from camera to body frame (Zcam = Xdrone, Xcam = -Ydrone, Ycam = -Zdrone)
        self.camera_translation = np.array([0.03, 0.0, -0.01]) # translation from body to camera frame (x forward, y left, z up)
        self.focal_length = 210 # focal length in pixels (calculated from FOV and image size)
        self.K = np.array([
            [self.focal_length,   0.0, 160.0],
            [  0.0, self.focal_length, 160.0],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float64)

        # Path through gate
        self.gate_approach_distance = 0.15 # [m] 
        self.gate_exit_distance = 0.10 # [m]

        # Closed loop control
        ### TODO:VINCENT ###
        # set your state variables here

        # Control Command
        self.position_tolerance = 0.05 # [m]
        self.yaw_tolerance = np.radians(5) # [rad]
        self.set_path = False # whether waypoints are already set or not, to prevent re-setting them every frame
        self.current_setpoint = None
        self.setpoint_queue = Queue()

        # World / arena parameters
        self.room_x = 4.05
        self.room_y = 2.87
        self.room_z = 3.00
        self._homepad_x = 0.875,
        self._origin_y = 1.43,
        self.wall_clearance = 0.04
        self.gate_nominal_sizes = {
            "50x40 cm": (0.50, 0.40),
            "40x40 cm": (0.40, 0.40),
            "29x40 cm": (0.29, 0.40),
        }

        # Tolerance bands for geometric validation
        self.val_planarity_thresh  = 0.20   # [m] max RMS deviation from best-fit plane, e.g. 8cm for a 40cm gate
        self.val_angle_thresh_deg  = 20.0   # [deg] max deviation from 90° at each corner, e.g. 15°
        self.val_side_ratio_thresh = 0.40   # max |w-h|/(w+h) asymmetry (gates are ~square to ~2:1)
        self.val_size_min  = 0.20           # [m] min plausible outer side length (height and width)
        self.val_size_max  = 0.70           # [m] max plausible outer side length (height and width)

        # Debugging
        self.debug_mode = DEBUG
        self.camera_data = None # for visualization in debug mode

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

        # Run YOLO detection and filter out low confidence corners
        detected_gates = self.yolo_detect(camera_data)
        detected_gates = [[cd for cd in det if cd.conf >= self.yolo_conf_threshold] for det in detected_gates]

        # ((TODO:13)) classical vision pipeline

        return detected_gates

    def yolo_detect(self, camera_data) -> list[list[CornerDetection]]:
        # Run inference on the camera data
        results = self.model.predict(source=camera_data, conf=0.5, iou=0.7, verbose=False)

        detected_gates = []
        iterable = results if isinstance(results, (list, tuple)) else [results]

        for r in iterable:
            keypoints = getattr(r, "keypoints", None)
            
            # Check if keypoints exist and are not empty using PyTorch's .numel()
            if keypoints is None or not hasattr(keypoints, "xy") or keypoints.xy.numel() == 0:
                continue

            # Safely move tensors to CPU and convert to numpy to avoid CUDA TypeErrors
            xy_all = keypoints.xy.cpu().numpy()
            conf_all = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None
        
            for i, det in enumerate(xy_all):
                # det is already a numpy array of shape (N_keypoints, 2)
                pts = det.reshape(-1, 2) 
                
                # Map confidences directly matching the shape of the detections
                if conf_all is None:
                    confs = np.ones(len(pts), dtype=float)
                else:
                    confs = conf_all[i]

                corners = []
                for j, p in enumerate(pts):
                    # Check for the (0, 0) default YOLO outputs for hidden keypoints
                    if p[0] == 0.0 and p[1] == 0.0:
                        continue

                    corner_id = CORNER_ID_MAP.get(j)
                    if corner_id is None:
                        continue
                    
                    conf_val = float(confs[j]) if j < len(confs) else 0.0
                    
                    corners.append(CornerDetection(
                        corner_id=corner_id,
                        uv=np.array([float(p[0]), float(p[1])]),
                        conf=conf_val
                    ))
                    
                # We also ensure we only append gates that actually have valid corners
                if corners:
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

            gray_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            new_pts, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, gray_frame, prev_pts, None,
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
        Updates candidates with new detection or creates new candidates as needed."""
        
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
        """
        Match detection with a recent candidate based on pixel distance.
        Recent means it has tracked points from the one of the last frames (not necessarily the immediately previous one, to allow for temporary occlusion or detection failure).
        """
        det_pts = {cd.corner_id: cd.uv for cd in detection}
        best_id, best_dist = None, float('inf')

        for cand_id, candidate in self.gate_candidates.items():
            if not candidate.last_tracked_pts:
                continue
            # Check recency
            if candidate.last_tracked_timestamp is None or (self.current_timestamp - candidate.last_tracked_timestamp) > self.forget_time:
                continue
            if candidate.last_tracked_timestamp == self.current_timestamp:
                continue  # already updated this frame with a different detection, skip to avoid double counting

            # Only compare corners that are present in both detection and candidate
            common = det_pts.keys() & candidate.last_tracked_pts.keys()
            if not common:
                continue
            dist = np.mean([np.linalg.norm(det_pts[k] - candidate.last_tracked_pts[k]) for k in common])
            if dist < best_dist:
                best_dist, best_id = dist, cand_id

        return best_id, best_dist

    def _add_detections_to_candidate(self, candidate: GateCandidate, detection: list[CornerDetection], P: np.ndarray):
        """Fuses pose into detections and stores as observations. Updates snapshot.
        Only add corner observations if we have a full gate detection (4 corners), but keep track of partial detections."""
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
            if len(detection):
                # Only add if we have a full gate detection
                candidate.add(obs)
            new_snapshot[cd.corner_id] = cd.uv
        candidate.last_tracked_pts = new_snapshot
        candidate.last_tracked_timestamp = self.current_timestamp

    def triangulate(self, candidate: GateCandidate) -> None:
        """Runs multi-view triangulation on the candidate's observations to estimate world positions of corners."""
        candidate.corners_world = candidate.triangulate()

        pass

    def validate_gate_pos(self, world_gate_pos: Dict[CornerID, np.ndarray]) -> tuple[bool, float]:
        """
        Geometric validation of triangulated gate corners.
        Returns (is_valid, confidence) where confidence ∈ [0, 1].

        Checks:
        1. All 4 corners present          (hard gate)
        2. Each corner within arena bounds (hard gate)
        3. Planarity                       (scored)
        4. Rectangularity                  (scored)
        5. Side lengths                    (scored)
        6. Gate roughly vertical           (scored)
        """
        if world_gate_pos is None:
            return False, 0.0

        # ── 1. Need all 4 corners (hard) ────────────────────────────────────────
        required = [CornerID.BL, CornerID.TL, CornerID.TR, CornerID.BR]
        if not all(c in world_gate_pos for c in required):
            return False, 0.0

        pts = np.array([world_gate_pos[c] for c in required], dtype=np.float64)

        # ── 2. Arena bounds (hard) ───────────────────────────────────────────────
        if not all(self._in_world_bounds(p) for p in pts):
            return False, 0.0

        scores = {}

        # ── 3. Planarity ─────────────────────────────────────────────────────────
        centroid  = pts.mean(axis=0)
        _, _, Vt  = np.linalg.svd(pts - centroid)
        normal    = Vt[-1]
        residuals = (pts - centroid) @ normal
        rms_planarity = float(np.sqrt(np.mean(residuals ** 2)))

        if rms_planarity > self.val_planarity_thresh:
            return False, 0.0

        # 1.0 when perfectly flat, 0.0 at threshold
        scores["planarity"] = 1.0 - (rms_planarity / self.val_planarity_thresh)

        # ── 4. Rectangularity ────────────────────────────────────────────────────
        angle_errors = []
        n = len(pts)
        for i in range(n):
            prev_v = pts[(i - 1) % n] - pts[i]
            next_v = pts[(i + 1) % n] - pts[i]
            cos_a  = np.dot(prev_v, next_v) / (
                np.linalg.norm(prev_v) * np.linalg.norm(next_v) + 1e-9
            )
            angle_deg = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
            angle_errors.append(abs(angle_deg - 90.0))

        max_angle_err = max(angle_errors)
        if max_angle_err > self.val_angle_thresh_deg:
            return False, 0.0

        # 1.0 when all perfect 90°, 0.0 when worst corner hits threshold
        scores["rectangularity"] = 1.0 - (max_angle_err / self.val_angle_thresh_deg)

        # ── 5. Side lengths ───────────────────────────────────────────────────────
        # use explicit enum keys instead of index ordering
        bl = world_gate_pos[CornerID.BL]
        tl = world_gate_pos[CornerID.TL]
        tr = world_gate_pos[CornerID.TR]
        br = world_gate_pos[CornerID.BR]

        left   = np.linalg.norm(tl - bl)
        top    = np.linalg.norm(tr - tl)
        right  = np.linalg.norm(br - tr)
        bottom = np.linalg.norm(bl - br)

        avg_h = (left  + right)  / 2.0
        avg_w = (top   + bottom) / 2.0

        for side in (avg_h, avg_w):
            if not (self.val_size_min <= side <= self.val_size_max):
                return False, 0.0

        # Score: how close to the nearest nominal gate size
        best_size_score = 0.0
        for nom_w, nom_h in self.gate_nominal_sizes.values():
            w_err = abs(avg_w - nom_w) / nom_w   # relative error
            h_err = abs(avg_h - nom_h) / nom_h
            # 1.0 = perfect match, 0.0 = 50%+ off
            s = max(0.0, 1.0 - 2.0 * max(w_err, h_err))
            best_size_score = max(best_size_score, s)

        # Also penalise opposite-side mismatch (parallelogram distortion)
        parallel_err = max(
            abs(left - right)  / (avg_h + 1e-9),
            abs(top  - bottom) / (avg_w + 1e-9),
        )
        parallelogram_score = max(0.0, 1.0 - parallel_err / 0.15)

        scores["size"] = 0.6 * best_size_score + 0.4 * parallelogram_score

        # ── 6. Verticality ────────────────────────────────────────────────────────
        normal_z = abs(normal[2])
        if normal_z > 0.5:
            return False, 0.0

        # 1.0 = perfectly vertical (normal_z=0), 0.0 at threshold (normal_z=0.5)
        scores["verticality"] = 1.0 - (normal_z / 0.5)

        # ── Weighted combination ──────────────────────────────────────────────────
        # Planarity and rectangularity carry most weight: they are the strongest
        # indicators of a clean triangulation. Size match is informative but noisier.
        weights = {
            "planarity":      0.35,
            "rectangularity": 0.35,
            "size":           0.20,
            "verticality":    0.10,
        }

        confidence = float(sum(weights[k] * scores[k] for k in weights))
        return True, confidence

    def _get_zone(self, x, y) -> Optional[int]:
        """
        Returns the sector zone index (0-8) for a gate at room position (x, y),
        matching the 9 non-home sectors defined in the arena layout, or None if
        the position is outside the arena or falls in the home zone.

        Zones are numbered 0-8 clockwise from just after the home zone boundary,
        matching the gate_num assignment in draw_sector_guides_2d().
        Only even-indexed zones (0, 2, 4, 6, 8) are gate zones.
        """
        # (TODO:5)

        if not self._in_world_bounds(np.array([x, y, 1.0])):
            return None

        center_xy = np.array([self.room_x / 2, self.room_y / 2])
        home_xy   = np.array([self._homepad_x, self._origin_y])

        dx = x - center_xy[0]
        dy = y - center_xy[1]
        angle = np.arctan2(dy, dx)             # [-π, π]

        home_angle      = np.arctan2(home_xy[1] - center_xy[1],
                                    home_xy[0] - center_xy[0])
        home_half       = np.deg2rad(45.0)
        home_start      = home_angle - home_half
        home_end        = home_angle + home_half

        def _wrap(a):
            return (a + np.pi) % (2 * np.pi) - np.pi

        # Angle relative to home_end, wrapped to [0, 2π)
        rel = (angle - home_end) % (2 * np.pi)

        # Home zone occupies the 90° arc before home_end
        home_span = (home_angle - home_half - home_end) % (2 * np.pi)
        if rel >= (2 * np.pi - 2 * home_half):
            return None    # inside home zone

        zone_idx = int(rel / np.deg2rad(30))
        return min(zone_idx, 8)   # 9 sectors: 0..8

    def _in_world_bounds(self, pos) -> bool:
        """
        Checks if a 3-D position [x, y, z] lies within the physical arena,
        with a small wall clearance margin.
        """
        pos = np.asarray(pos, dtype=np.float64)
        if pos.shape != (3,):
            return False

        x, y, z = pos
        margin = self.wall_clearance

        in_x = (margin <= x <= self.room_x - margin)
        in_y = (margin <= y <= self.room_y - margin)
        in_z = (0.0    <= z <= self.room_z - margin)

        return bool(in_x and in_y and in_z)

    def set_path_through_gate(self, selected_gate) -> None:
        """Sets waypoints to pass through the selected gate candidate."""

        # Basic validation
        if selected_gate is None or not getattr(selected_gate, "corners_world", None):
            return

        gate_center, gate_normal = self._calculate_center_and_normal(selected_gate)

        if self.debug_mode:
            assert gate_center is not None and gate_normal is not None, "Failed to calculate gate center and normal"

        # Ensure normal points from drone toward gate (so approach is sensible)
        drone_pos = self.current_drone_xyz if self.current_drone_xyz is not None else np.zeros(3)
        drone_to_gate = gate_center - drone_pos
        if np.dot(gate_normal, drone_to_gate) < 0:
            gate_normal = -gate_normal

        yaw_to_gate = float(np.arctan2(gate_normal[1], gate_normal[0]))

        # Set waypoints (x, y, z, yaw)
        approach_wp = gate_center - gate_normal * self.gate_approach_distance
        through_wp = gate_center.copy()
        exit_wp = gate_center + gate_normal * self.gate_exit_distance

        def make_setpoint(pos, yaw):
            return [float(pos[0]), float(pos[1]), float(pos[2]), float(yaw)]

        self.setpoint_queue.put(make_setpoint(approach_wp, yaw_to_gate))
        self.setpoint_queue.put(make_setpoint(through_wp, yaw_to_gate))
        self.setpoint_queue.put(make_setpoint(exit_wp, yaw_to_gate))

        return

    def _calculate_center_and_normal(self, selected_gate) -> tuple[np.ndarray, np.ndarray]:
        # Require explicit CornerID keys (BL, TL, TR, BR)
        try:
            bl = selected_gate.corners_world[CornerID.BL]
            tl = selected_gate.corners_world[CornerID.TL]
            tr = selected_gate.corners_world[CornerID.TR]
            br = selected_gate.corners_world[CornerID.BR]
        except KeyError:
            return None, None

        pts = np.vstack([bl, tl, tr, br])
        gate_center = pts.mean(axis=0)

        # Compute gate normal using two edges (TL-BL and TR-BL)
        v1 = tl - bl
        v2 = tr - bl
        gate_normal = np.cross(v1, v2)
        gate_normal[2] = 0.0  # enforce vertical normal (ignore Z component) since gates are vertical
        norm = np.linalg.norm(gate_normal)
        if norm < 1e-6:
            return gate_center, None
        gate_normal = gate_normal / norm

        return gate_center, gate_normal

    def closed_loop_control(self, detections: list[list[CornerDetection]]) -> tuple[list, bool]:
        """
        Identifies the current gate candidate to pass through, sets waypoints.

        Input:
            - detections: A list of detected gates each represented as a list of CornerDetection objects.

        Output:
            - (x, y, z, yaw) new setpoint to fly towards
            - boolean indicating whether we have a valid gate candidate to pass through
        """

        x, y, z = self.current_drone_xyz
        _, _, yaw = self.current_drone_rpy

        zone_id = self._get_zone(x, y)  # Nicolas will check this function, maybe talk with him what you prefer as output taking inaccuracy into account
        
        if self.debug_mode:
            print(f"{len(detections) if isinstance(detections, list) else 0} gates detected")

        for det in detections if isinstance(detections, list) else []:
            for cd in det:
                # pixel coordinates: u_max, v_max = (244, 324)
                u, v = cd.uv
                # corner ID: BL, TL, TR, BR
                corner_id = cd.corner_id

                if corner_id == CornerID.BL:
                    if self.debug_mode:
                        print(f"Detected BL corner at pixel ({u:.1f}, {v:.1f}) with confidence {cd.conf:.2f}")

        ### TODO:VINCENT ###

        return [x, y, z, yaw], False


    def _clear_waypoints(self) -> None:
        """Clears current setpoint and queued waypoints."""
        self.current_setpoint = None
        while not self.setpoint_queue.empty():
            self.setpoint_queue.get()

    def _no_waypoints(self) -> bool:
        """Returns True if there are no current or queued waypoints."""
        return self.current_setpoint is None and self.setpoint_queue.empty()

    def set_control_command(self):
        """
        If the current setpoint is reached within tolerance, pops the next waypoint from the queue and sets it as the new setpoint.
        However, the old setpoint will still be the command until the next frame.
        """

        if self.current_setpoint is None:
            if self.setpoint_queue.empty():
                # If no setpoint is set, just hover in place
                return [self.current_drone_xyz[0], self.current_drone_xyz[1], self.current_drone_xyz[2], self.current_drone_rpy[2]]
            else:
                # Otherwise, get next waypoint from queue
                self.current_setpoint = self.setpoint_queue.get()

        # Set control command to current setpoint
        ctrl_cmd = [self.current_setpoint[0], self.current_setpoint[1], self.current_setpoint[2], self.current_setpoint[3]]

        # Check if we have reached the current waypoint, if so, get next waypoint from queue
        dist_xyz_from_setpoint = np.linalg.norm(np.array(self.current_setpoint[:3]) - np.array([self.current_drone_xyz[0], self.current_drone_xyz[1], self.current_drone_xyz[2]]))
        diff_yaw_from_setpoint = abs(_add_angles(self.current_setpoint[3], -self.current_drone_rpy[2]))
        reached_position = dist_xyz_from_setpoint < self.position_tolerance
        reached_yaw = diff_yaw_from_setpoint < self.yaw_tolerance

        if reached_position and reached_yaw:
            self.current_setpoint = self.setpoint_queue.get() if not self.setpoint_queue.empty() else None

        return ctrl_cmd
    
    def debug_visualization(self) -> None:

        if self.camera_data is None:
            return
        
        vis = self.camera_data.copy()

        # Draw candidate gates
        colors = _generate_color_dict(self.gate_candidates)
        for candidate_id, candidate in self.gate_candidates.items():
            color = colors.get(candidate_id, (0, 255, 0))
            for obs_list in candidate.observations.values():
                for obs in obs_list:
                    if obs.timestamp != self.current_timestamp:
                        continue
                    cv2.circle(vis, tuple(obs.uv.astype(int)), 5, color, -1)

        # Visualize
        cv2.imshow("Debug Visualization", vis)
        cv2.waitKey(1)

        pass


    def compute_command(self, sensor_data, camera_data, dt):
        """Process camera data and sensor data to compute control command"""

        if self.debug_mode:
            self.debug_visualization()
            self.camera_data = camera_data.copy()  # for visualization in debug mode


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

            # Exit state
            if sensor_data['z_global'] > 0.9:
                self.state = DroneState.SEARCH
                self.set_path = False # reset for next state

        ### SEARCH ###
        elif self.state == DroneState.SEARCH:

            # 0) Initialize search path
            if not self.set_path:
                self._clear_waypoints()
                
                ### TODO:1 SET WAYPOINTS FOR SEARCH PATTERN ###
                # self.setpoint_queue.put([x1, y1, z1, yaw1])
                # self.setpoint_queue.put([x2, y2, z2, yaw2])
                # ...

                self.set_path = True

            # 1) Update candidate states based on timeouts and triangulation results
            delete_ids = []
            for i, candidate in self.gate_candidates.items():
                if candidate.last_seen is None:
                    continue
                age = self.current_timestamp - candidate.last_seen.timestamp

                if candidate.state == CandidateState.TRACKING and age > self.forget_time:
                    candidate.state = CandidateState.TRIANGULATING

                if candidate.state == CandidateState.TRIANGULATING:
                    self.triangulate(candidate)

                    if candidate.corners_world is None:
                        candidate.state = CandidateState.REJECTED
                    else:
                        gate, validation_conf = self.validate_gate_pos(candidate.corners_world)
                        candidate.val_conf = validation_conf
                        candidate.state = CandidateState.CONFIRMED if gate else CandidateState.REJECTED

                    if self.debug_mode:
                        if candidate.corners_world:
                            lines = [f"Candidate {i} triangulated corners (world):"]
                            for cid in (CornerID.BL, CornerID.TL, CornerID.TR, CornerID.BR):
                                pos = candidate.corners_world.get(cid)
                                if pos is None:
                                    lines.append(f"  {cid.name:>3}: None")
                                else:
                                    lines.append(f"  {cid.name:>3}: [{pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f}]")
                            if candidate.val_conf is not None:
                                lines.append(f"  Validation confidence: {candidate.val_conf:.2f}")
                            print("\n".join(lines))
                        else:
                            print(f"Candidate {i} triangulated corners (world): None")

                if candidate.state == CandidateState.CONFIRMED:
                    self.detected_gates[i] = candidate
                    delete_ids.append(i) # remove from candidates, now in detected_gates

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

            # 3) Closed loop control by Vincent
            detections = self.detect(camera_data)
            next_setpoint, is_gate = self.closed_loop_control(detections)
            self.current_setpoint = next_setpoint

            # 4) (Optional) Additional adaptation of search path or waypoints based on detections or tracking results
            # ((TODO:4))

            # 4) Update prev_frame for tracking
            self.prev_frame = cv2.cvtColor(camera_data, cv2.COLOR_BGR2GRAY)
            self.prev_frame_timestamp = self.current_timestamp

            # 5) Exit condition for search state
            ### TODO:15 define exit condition for triangulation
            if is_gate:
                self._clear_waypoints() # clear search waypoints
                self.current_setpoint = next_setpoint # set to current setpoint from closed-loop control

                print("Gate(s) confirmed, transitioning to PASS_GATE.")
                self.state = DroneState.PASS_GATE
                self.set_path = True # we do not need to set the gate setpoints in the next state, as we already have a valid gate candidate and setpoint from closed-loop control

            pass

        ### PASS GATE ###
        elif self.state == DroneState.PASS_GATE:

            # 1) Set waypoints to pass through the gate
            if not self.set_path:
                self._clear_waypoints()

                ### TODO:8 SELECT WHICH GATE TO PASS THROUGH ###

                if self.detected_gates:
                    selected_gate = next(iter(self.detected_gates.values()))  # Select the first detected gate
                    self.set_path_through_gate(selected_gate)

                    if self.debug_mode:
                        # print all detected gates with their wworld coordinates and confidence
                        for i, gate in self.detected_gates.items():
                            print(f"Gate {i}:")
                            if gate.corners_world:
                                for cid, pos in gate.corners_world.items():
                                    print(f"  {cid.name}: {pos}")
                            else:
                                print("  No world coordinates")
                            print(f"  Validation confidence: {gate.val_conf:.2f}")

                self.set_path = True

            # 2) (Optional) Additional logic to adapt waypoints if gate position is uncertain or if we have new observations while passing through
            # (TODO:9) ADAPT WAYPOINTS BASED ON NEW OBSERVATIONS OR UNCERTAINTY DURING PASSING

            # 3) Exit condition for PASS_GATE state
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
