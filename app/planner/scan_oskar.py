import os, cv2
from enum import Enum
import numpy as np
from pyglm import glm
from queue import Queue
from typing import Optional, Dict
from dataclasses import dataclass, field
from cv2.typing import MatLike
from overrides import overrides
from app import wrap
from app.io import Measurement
from app.io import Setpoint
from app.planner import Planner
from app.telemetry import Telemetry
from app.telemetry.gate import Gate
from app.telemetry.camera import UP, world2clip, clip2screen, CLIP_PLANES, WIDTH, HEIGHT, view, euler_to_quaternion
from ultralytics import YOLO

MODEL_PATH = os.path.join("controller_detection", "detection_model", "models", "yolov8n_v3bw_r1", "weights", "best.pt")

# ── Corner data structures (ported from DetectionController) ──────────────────

class CornerID(Enum):
    BL = 0
    TL = 1
    TR = 2
    BR = 3

CORNER_ID_MAP = {0: CornerID.BL, 1: CornerID.TL, 2: CornerID.TR, 3: CornerID.BR}

@dataclass
class CornerDetection:
    corner_id: CornerID
    uv: np.ndarray
    conf: float

@dataclass
class CornerObservation:
    corner_id: CornerID
    uv: np.ndarray
    conf: float
    timestamp: float
    P: np.ndarray        # 3×4 projection matrix
    drone_xyz: np.ndarray
    drone_rpy: np.ndarray

class CandidateState(Enum):
    TRACKING      = 0
    TRIANGULATING = 1
    CONFIRMED     = 2
    REJECTED      = 3

@dataclass
class GateCandidate:
    TRIANGULATION_CONF = 0.5
    observations: Dict[CornerID, list] = field(default_factory=dict)
    last_tracked_pts: Optional[Dict[CornerID, np.ndarray]] = field(default=None, repr=False)
    last_tracked_timestamp: Optional[float] = field(default=None, repr=False)
    state: CandidateState = CandidateState.TRACKING
    corners_world: Optional[Dict[CornerID, np.ndarray]] = field(default=None)
    val_conf: Optional[float] = field(default=None)
    _last_seen: Optional[CornerObservation] = field(default=None, repr=False, init=False)

    def add(self, obs: CornerObservation):
        self.observations.setdefault(obs.corner_id, []).append(obs)
        if self._last_seen is None or obs.timestamp > self._last_seen.timestamp:
            self._last_seen = obs

    @property
    def last_seen(self) -> Optional[CornerObservation]:
        return self._last_seen

    def triangulate(self) -> Optional[Dict[CornerID, np.ndarray]]:
        result = {}
        for corner_id, obs_list in self.observations.items():
            filtered = [o for o in obs_list if o.conf >= self.TRIANGULATION_CONF]
            if len(filtered) < 2:
                continue
            pts = np.array([o.uv for o in filtered], dtype=np.float64).T
            Ps  = [o.P for o in filtered]
            result[corner_id] = self._dlt(pts, Ps)
        return result if result else None

    def _dlt(self, pts: np.ndarray, Ps: list) -> np.ndarray:
        A = []
        for uv, P in zip(pts.T, Ps):
            u, v = uv
            A.append(u * P[2] - P[0])
            A.append(v * P[2] - P[1])
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        return X[:3] / X[3]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _euler2rotmat(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    Rr = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
    Rp = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    return Ry @ Rp @ Rr


class Scan(Planner):

    INITIAL_SETPOINT = Setpoint(Planner.HOME_POSITION, wrap(np.deg2rad(-45)))
    GATE_PASS_DIST   = 0.10   # m — offset past gate centre for exit waypoint

    # ── Detection / tracking ──────────────────────────────────────────────────
    YOLO_CONF          = 0.5
    YOLO_IOU           = 0.7
    FORGET_TIME        = 1.0    # s — age after which a TRACKING candidate moves to TRIANGULATING
    MATCH_THRESHOLD    = 100.0  # px — max centroid distance to associate detection to candidate
    TRACK_ERR_MAX      = 20.0   # px — max LK reprojection error to accept a tracked point
    REDETECT_INTERVAL  = 0.0    # s — 0 = detect every frame

    # ── Validation ────────────────────────────────────────────────────────────
    VAL_PLANARITY     = 0.20
    VAL_ANGLE_DEG     = 20.0
    VAL_SIDE_RATIO    = 0.40
    VAL_SIZE_MIN      = 0.20
    VAL_SIZE_MAX      = 0.70
    GATE_NOMINAL_SIZES = {"50x40": (0.50, 0.40), "40x40": (0.40, 0.40), "29x40": (0.29, 0.40)}

    # ── Arena ─────────────────────────────────────────────────────────────────
    ROOM_X         = 4.05
    ROOM_Y         = 2.87
    ROOM_Z         = 3.00
    WALL_CLEARANCE = 0.04

    # ── Gate selection ────────────────────────────────────────────────────────
    # Maximum distance from drone to a confirmed gate to be considered the "next" one
    GATE_SELECT_MAX_DIST = 3.0  # m

    # ── Camera intrinsics (must match your physical camera) ───────────────────
    K = np.array([
        [140.23528025,   0.0,          169.70725091],
        [  0.0,        141.12756104,   148.24022948],
        [  0.0,          0.0,            1.0       ],
    ], dtype=np.float64)
    # Rotation from camera frame to body frame
    CAM_R = np.array([[0,-1,0],[0,0,-1],[1,0,0]], dtype=np.float64)
    # Translation from body origin to camera (x-fwd, y-left, z-up)
    CAM_T = np.array([0.03, 0.0, -0.01], dtype=np.float64)

    class State(Enum):
        FOLLOW_PATH  = 0   # flying the pre-planned scan waypoints
        PASS_GATE    = 1   # flying the approach→through→exit waypoints for one gate
        END          = 2

    def __init__(self) -> None:
        super().__init__()
        self.model = YOLO(MODEL_PATH)
        self.model.eval()

        # Pre-planned scan path (populated by _build_scan_path)
        self._build_scan_path()

        # Detection / tracking state
        self.timestamp          : float                      = 0.0
        self.gate_candidates    : Dict[int, GateCandidate]  = {}
        self.detected_gates     : Dict[int, GateCandidate]  = {}
        self._next_cand_id      : int                        = 0
        self._last_detect_t     : float                      = 0.0
        self._prev_frame        : Optional[np.ndarray]       = None
        self._prev_frame_t      : Optional[float]            = None

        # Gate-passing queue
        self._pass_queue        : Queue = Queue()
        self._pass_setpoint     : Optional[Setpoint] = None
        self._gates_passed      : int   = 0

        self.state = Scan.State.FOLLOW_PATH

        self.load_sim()

    # ─────────────────────────────────────────────────────────────────────────
    # Scan path
    # ─────────────────────────────────────────────────────────────────────────

    def _build_scan_path(self) -> None:
        """
        Populate self.waypoints with a fixed sweep trajectory.
        Flies a horizontal boustrophedon (lawnmower) at constant altitude,
        pointing the nose in the direction of travel at each leg.
        Adjust positions / yaws to suit your arena.
        """
        self.waypoints.clear()
        z_top   = 1.4
        z_bottom = 1.1
        pts = [
            (Planner.HOME_POSITION.x, Planner.HOME_POSITION.y, 1.0, np.deg2rad(-45)),
            (-0.4, -0.7,  z_bottom, np.deg2rad(-5)),
            (-0.4, -0.7,  z_top, np.deg2rad(-5)),
            (-0.7, -1.2,  z_top, np.deg2rad(30)),
            (-0.7, -1.2,  z_bottom, np.deg2rad(30)),
            (0.0, -1.0,  z_bottom, np.deg2rad(-10)),
            (0.0, -1.0,  z_top, np.deg2rad(-10)),
            (0.0, -1.0,  z_top, np.deg2rad(75)),
            (0.0, -1.0,  z_bottom, np.deg2rad(75)),
            (2.3, -1.4,  z_bottom, np.deg2rad(80)),
            (2.3, -1.4,  z_top, np.deg2rad(80)),
            (3.0, -1.5,  z_top, np.deg2rad(125)),
            (3.0, -1.5,  z_bottom, np.deg2rad(125)),
            (3.0, 0.5,  z_bottom, np.deg2rad(160)),
            (3.0, 0.5,  z_top, np.deg2rad(160)),
            (3.0, 0.8,  z_top, np.deg2rad(190)),
            (3.0, 0.8,  z_bottom, np.deg2rad(190)),
            (0.0, 1.0,  z_bottom, np.deg2rad(150)),
            (0.0, 1.0,  z_top, np.deg2rad(150)),
            (0.0, 1.0,  z_top, np.deg2rad(260)),
            (Planner.HOME_POSITION.x, Planner.HOME_POSITION.y, 1.0, np.deg2rad(-45)),
        ]
        for x, y, z_, yaw in pts:
            self.waypoints.append(Setpoint(glm.vec3(x, y, z_), wrap(yaw)))

    # ─────────────────────────────────────────────────────────────────────────
    # Planner overrides
    # ─────────────────────────────────────────────────────────────────────────

    def load_sim(self) -> None:
        gates_directory = os.path.join("gates")
        if not os.path.isdir(gates_directory) or not os.listdir(gates_directory):
            self.sim_gates = []
            return
        file_name  = os.listdir(gates_directory)[0]
        file_path  = os.path.join(gates_directory, file_name)
        raw        = np.genfromtxt(file_path, delimiter=',', dtype=float, ndmin=2, skip_header=1, filling_values=np.nan)
        if raw.size == 0 or np.isnan(np.atleast_1d(raw[0])).any():
            self.sim_gates = []
            return
        positions  = [glm.vec3(r[1], r[2], r[3]) for r in raw]
        yaws       = [wrap(r[4]) for r in raw]
        self.sim_gates = [Setpoint(p, y) for p, y in zip(positions, yaws)]

    @overrides
    def reload(self) -> None:
        self.gate_candidates.clear()
        self.detected_gates.clear()
        self._next_cand_id   = 0
        self._last_detect_t  = 0.0
        self._prev_frame     = None
        self._prev_frame_t   = None
        self.timestamp       = 0.0
        self._gates_passed   = 0
        while not self._pass_queue.empty():
            self._pass_queue.get()
        self._pass_setpoint  = None
        self._build_scan_path()
        self.state = Scan.State.FOLLOW_PATH
        self.load_sim()

    @overrides
    def update(self, measurement: Measurement, frame: MatLike, flags: Telemetry.Flags, dt: float) -> Setpoint:

        self.timestamp += dt

        # ── Derive numpy pose for the detection pipeline ──────────────────────
        drone_xyz = np.array([measurement.position.x, measurement.position.y, measurement.position.z])
        drone_rpy = np.array([measurement.rotation.x, measurement.rotation.y, measurement.rotation.z])

        # ── Detection / tracking / triangulation (runs in every state) ────────
        if Telemetry.Flags.NEW_FRAME in flags:
            if self.timestamp - self._last_detect_t >= Scan.REDETECT_INTERVAL:
                detections = self._detect(frame)
                P = self._projection(drone_xyz, drone_rpy)
                self._associate_and_update(detections, P, drone_xyz, drone_rpy)
                self._last_detect_t = self.timestamp
            else:
                self._track(frame)

            # Visualise whatever find_gates returns (keeps telemetry overlay working)
            sim_gates = self.find_gates(frame, measurement, flags)
            self.gates_detected_event(sim_gates)

            self._prev_frame   = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB if len(frame.shape) == 2 else cv2.COLOR_BGR2GRAY)
            self._prev_frame_t = self.timestamp

        # ── Candidate lifecycle: age-out → triangulate → validate ─────────────
        self._update_candidates()

        # ── State machine ─────────────────────────────────────────────────────
        match self.state:

            case Scan.State.FOLLOW_PATH:
                setpoint, wp_reached = Planner.reach(self.waypoints[-1], measurement, 0.5)

                # Check whether we have a confirmed gate that is the right next gate
                next_gate_id, next_gate = self._select_next_gate(drone_xyz, drone_rpy)

                if next_gate is not None:
                    # Enqueue approach → through → exit waypoints and switch state
                    self._enqueue_gate_passage(next_gate, drone_xyz)
                    del self.detected_gates[next_gate_id]   # consume so we don't re-select it
                    self._gates_passed += 1
                    self.state = Scan.State.PASS_GATE
                    return self.update(measurement, frame, flags, dt)

                # All 5 gates passed → go home
                if self._gates_passed >= 5:
                    self.waypoints.clear()
                    self.waypoints.append(Planner.HOME_SETPOINT)
                    self.state = Scan.State.END
                    return self.update(measurement, frame, flags, dt)

                # Wrap scan path: if we finished the sweep without finding a gate, restart
                if wp_reached and len(self.waypoints) > 0:
                    # Cycle waypoints: pop front, push to back so we keep sweeping
                    self.waypoints.append(self.waypoints.pop(0))

                return setpoint

            case Scan.State.PASS_GATE:
                # Advance through the enqueued approach/through/exit setpoints
                if self._pass_setpoint is None:
                    if self._pass_queue.empty():
                        # Finished passing — resume scan
                        self._build_scan_path()
                        self.state = Scan.State.FOLLOW_PATH
                        return self.update(measurement, frame, flags, dt)
                    self._pass_setpoint = self._pass_queue.get()

                setpoint, reached = Planner.reach(self._pass_setpoint, measurement, 0.05)
                if reached:
                    self._pass_setpoint = self._pass_queue.get() if not self._pass_queue.empty() else None
                return setpoint

            case Scan.State.END:
                setpoint, _ = Planner.reach(self.waypoints[-1], measurement, 0.5)
                return setpoint

    # ─────────────────────────────────────────────────────────────────────────
    # Candidate lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def _update_candidates(self) -> None:
        """Age-out stale candidates, triangulate, validate, promote or reject."""
        promote, reject = [], []

        for cid, candidate in self.gate_candidates.items():
            if candidate.last_seen is None:
                continue
            age = self.timestamp - candidate.last_seen.timestamp

            if candidate.state == CandidateState.TRACKING and age > Scan.FORGET_TIME:
                candidate.state = CandidateState.TRIANGULATING

            if candidate.state == CandidateState.TRIANGULATING:
                candidate.corners_world = candidate.triangulate()
                if candidate.corners_world is None:
                    candidate.state = CandidateState.REJECTED
                else:
                    valid, conf = self._validate(candidate.corners_world)
                    candidate.val_conf = conf
                    candidate.state = CandidateState.CONFIRMED if valid else CandidateState.REJECTED

            if candidate.state == CandidateState.CONFIRMED:
                promote.append(cid)
            elif candidate.state == CandidateState.REJECTED:
                reject.append(cid)

        for cid in promote:
            self.detected_gates[cid] = self.gate_candidates.pop(cid)
        for cid in reject:
            self.gate_candidates.pop(cid, None)

    # ─────────────────────────────────────────────────────────────────────────
    # Gate selection
    # ─────────────────────────────────────────────────────────────────────────

    def _select_next_gate(self, drone_xyz: np.ndarray, drone_rpy: np.ndarray) -> tuple[Optional[int], Optional[GateCandidate]]:
        """
        From confirmed gates, return the one that is:
          1. Within GATE_SELECT_MAX_DIST of the drone, AND
          2. In the correct zone for the current drone position, AND
          3. Nearest to the drone among those candidates.
        Returns (id, candidate) or (None, None).
        """
        drone_zone = self._get_zone(drone_xyz[0], drone_xyz[1])

        best_id, best_cand, best_dist = None, None, float('inf')

        for gid, candidate in self.detected_gates.items():
            center = self._gate_center(candidate)
            if center is None:
                continue

            dist = float(np.linalg.norm(center - drone_xyz))
            if dist > Scan.GATE_SELECT_MAX_DIST:
                continue

            gate_zone = self._get_zone(center[0], center[1])
            if gate_zone != drone_zone:
                continue

            if dist < best_dist:
                best_dist, best_id, best_cand = dist, gid, candidate

        return best_id, best_cand

    def _gate_center(self, candidate: GateCandidate) -> Optional[np.ndarray]:
        if not candidate.corners_world:
            return None
        pts = np.array(list(candidate.corners_world.values()), dtype=np.float64)
        return pts.mean(axis=0)

    def _gate_normal(self, candidate: GateCandidate, drone_xyz: np.ndarray) -> Optional[np.ndarray]:
        cw = candidate.corners_world
        if cw is None:
            return None
        try:
            bl, tl, tr = cw[CornerID.BL], cw[CornerID.TL], cw[CornerID.TR]
        except KeyError:
            return None
        n = np.cross(tl - bl, tr - bl).astype(float)
        n[2] = 0.0
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            return None
        n /= norm
        # Ensure normal points from gate toward drone
        if np.dot(n, drone_xyz - self._gate_center(candidate)) < 0:
            n = -n
        return n

    def _enqueue_gate_passage(self, candidate: GateCandidate, drone_xyz: np.ndarray) -> None:
        center = self._gate_center(candidate)
        normal = self._gate_normal(candidate, drone_xyz)
        if center is None or normal is None:
            return

        yaw = wrap(float(np.arctan2(-normal[1], -normal[0])))  # face the gate

        approach = center - normal * 0.15
        through  = center.copy()
        exit_pt  = center + normal * Scan.GATE_PASS_DIST

        for pt in (approach, through, exit_pt):
            self._pass_queue.put(Setpoint(glm.vec3(float(pt[0]), float(pt[1]), float(pt[2])), yaw))

    # ─────────────────────────────────────────────────────────────────────────
    # Detection
    # ─────────────────────────────────────────────────────────────────────────

    def _detect(self, frame: MatLike) -> list[list[CornerDetection]]:
        if len(frame.shape) == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = frame
        results = self.model.predict(source=rgb, conf=Scan.YOLO_CONF, iou=Scan.YOLO_IOU, verbose=False)

        detected = []
        for r in (results if isinstance(results, (list, tuple)) else [results]):
            kps = getattr(r, "keypoints", None)
            if kps is None or not hasattr(kps, "xy") or kps.xy.numel() == 0:
                continue
            xy_all   = kps.xy.cpu().numpy()
            conf_all = kps.conf.cpu().numpy() if kps.conf is not None else None

            for i, det in enumerate(xy_all):
                pts   = det.reshape(-1, 2)
                confs = conf_all[i] if conf_all is not None else np.ones(len(pts))
                corners = []
                for j, p in enumerate(pts):
                    if p[0] == 0.0 and p[1] == 0.0:
                        continue
                    cid = CORNER_ID_MAP.get(j)
                    if cid is None:
                        continue
                    corners.append(CornerDetection(
                        corner_id=cid,
                        uv=np.array([float(p[0]), float(p[1])]),
                        conf=float(confs[j]) if j < len(confs) else 0.0,
                    ))
                if corners:
                    detected.append(corners)
        return detected

    # ─────────────────────────────────────────────────────────────────────────
    # Tracking (Lucas-Kanade)
    # ─────────────────────────────────────────────────────────────────────────

    def _track(self, frame: MatLike) -> None:
        if self._prev_frame is None:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        for candidate in self.gate_candidates.values():
            if candidate.last_tracked_pts is None:
                continue
            if candidate.last_tracked_timestamp != self._prev_frame_t:
                continue

            corner_ids = list(candidate.last_tracked_pts.keys())
            prev_pts   = np.array(
                [candidate.last_tracked_pts[cid] for cid in corner_ids], dtype=np.float32
            ).reshape(-1, 1, 2)

            new_pts, status, error = cv2.calcOpticalFlowPyrLK(
                self._prev_frame, gray, prev_pts, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )

            tracked = {}
            for i, cid in enumerate(corner_ids):
                if status[i, 0] and error[i, 0] < Scan.TRACK_ERR_MAX:
                    tracked[cid] = new_pts[i, 0]

            if tracked:
                candidate.last_tracked_pts       = tracked
                candidate.last_tracked_timestamp = self.timestamp

    # ─────────────────────────────────────────────────────────────────────────
    # Association
    # ─────────────────────────────────────────────────────────────────────────

    def _associate_and_update(
        self,
        detections: list[list[CornerDetection]],
        P: np.ndarray,
        drone_xyz: np.ndarray,
        drone_rpy: np.ndarray,
    ) -> None:
        for detection in detections:
            det_pts  = {cd.corner_id: cd.uv for cd in detection}
            best_id, best_dist = None, float('inf')

            for cid, candidate in self.gate_candidates.items():
                if not candidate.last_tracked_pts:
                    continue
                if candidate.last_tracked_timestamp is None:
                    continue
                age = self.timestamp - candidate.last_tracked_timestamp
                if age > Scan.FORGET_TIME:
                    continue
                if candidate.last_tracked_timestamp == self.timestamp:
                    continue  # already consumed this frame

                common = det_pts.keys() & candidate.last_tracked_pts.keys()
                if not common:
                    continue
                dist = float(np.mean([np.linalg.norm(det_pts[k] - candidate.last_tracked_pts[k]) for k in common]))
                if dist < best_dist:
                    best_dist, best_id = dist, cid

            if best_id is not None and best_dist < Scan.MATCH_THRESHOLD:
                candidate = self.gate_candidates[best_id]
            else:
                candidate = GateCandidate()
                self.gate_candidates[self._next_cand_id] = candidate
                self._next_cand_id += 1

            # Add observations and update snapshot
            new_snap = {}
            for cd in detection:
                obs = CornerObservation(
                    corner_id=cd.corner_id,
                    uv=cd.uv,
                    conf=cd.conf,
                    timestamp=self.timestamp,
                    P=P,
                    drone_xyz=drone_xyz.copy(),
                    drone_rpy=drone_rpy.copy(),
                )
                candidate.add(obs)
                new_snap[cd.corner_id] = cd.uv
            candidate.last_tracked_pts       = new_snap
            candidate.last_tracked_timestamp = self.timestamp

    # ─────────────────────────────────────────────────────────────────────────
    # Projection matrix
    # ─────────────────────────────────────────────────────────────────────────

    def _projection(self, xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
        R_wb = _euler2rotmat(rpy)
        R_cw = Scan.CAM_R @ R_wb.T
        t_wc = R_wb @ Scan.CAM_T + xyz
        t_cw = -R_cw @ t_wc
        return Scan.K @ np.hstack((R_cw, t_cw.reshape(3, 1)))

    # ─────────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────────

    def _validate(self, cw: Dict[CornerID, np.ndarray]) -> tuple[bool, float]:
        required = [CornerID.BL, CornerID.TL, CornerID.TR, CornerID.BR]
        if not all(c in cw for c in required):
            return False, 0.0
        pts = np.array([cw[c] for c in required], dtype=np.float64)

        # Hard: arena bounds
        if not all(self._in_bounds(p) for p in pts):
            return False, 0.0

        scores = {}

        # Planarity
        centroid  = pts.mean(axis=0)
        _, _, Vt  = np.linalg.svd(pts - centroid)
        normal    = Vt[-1]
        rms       = float(np.sqrt(np.mean(((pts - centroid) @ normal) ** 2)))
        if rms > Scan.VAL_PLANARITY:
            return False, 0.0
        scores["planarity"] = 1.0 - rms / Scan.VAL_PLANARITY

        # Rectangularity
        n = len(pts)
        angle_errs = []
        for i in range(n):
            v1 = pts[(i-1) % n] - pts[i]
            v2 = pts[(i+1) % n] - pts[i]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            angle_errs.append(abs(np.degrees(np.arccos(np.clip(cos_a, -1, 1))) - 90.0))
        max_err = max(angle_errs)
        if max_err > Scan.VAL_ANGLE_DEG:
            return False, 0.0
        scores["rectangularity"] = 1.0 - max_err / Scan.VAL_ANGLE_DEG

        # Side lengths
        bl, tl, tr, br = cw[CornerID.BL], cw[CornerID.TL], cw[CornerID.TR], cw[CornerID.BR]
        avg_h = (np.linalg.norm(tl-bl) + np.linalg.norm(br-tr)) / 2.0
        avg_w = (np.linalg.norm(tr-tl) + np.linalg.norm(bl-br)) / 2.0
        if not (Scan.VAL_SIZE_MIN <= avg_h <= Scan.VAL_SIZE_MAX and
                Scan.VAL_SIZE_MIN <= avg_w <= Scan.VAL_SIZE_MAX):
            return False, 0.0
        best_size = 0.0
        for nom_w, nom_h in Scan.GATE_NOMINAL_SIZES.values():
            s = max(0.0, 1.0 - 2.0 * max(abs(avg_w-nom_w)/nom_w, abs(avg_h-nom_h)/nom_h))
            best_size = max(best_size, s)
        scores["size"] = best_size

        # Verticality
        normal_z = abs(normal[2])
        if normal_z > 0.5:
            return False, 0.0
        scores["verticality"] = 1.0 - normal_z / 0.5

        weights = {"planarity": 0.35, "rectangularity": 0.35, "size": 0.20, "verticality": 0.10}
        conf = float(sum(weights[k] * scores[k] for k in weights))
        return True, conf

    def _in_bounds(self, pos) -> bool:
        pos = np.asarray(pos, dtype=np.float64)
        if pos.shape[0] < 3:
            return False
        x, y, z = pos[:3]
        m = Scan.WALL_CLEARANCE
        return (m <= x <= Scan.ROOM_X - m and
                m <= y <= Scan.ROOM_Y - m and
                0.0 <= z <= Scan.ROOM_Z - m)

    # ─────────────────────────────────────────────────────────────────────────
    # Zone helper
    # ─────────────────────────────────────────────────────────────────────────

    def _get_zone(self, x: float, y: float) -> Optional[int]:
        if not self._in_bounds(np.array([x, y, 1.0])):
            return None
        cx, cy     = Scan.ROOM_X / 2.0, Scan.ROOM_Y / 2.0
        hx, hy     = Planner.HOME_POSITION.x, Planner.HOME_POSITION.y
        angle      = np.arctan2(y - cy, x - cx)
        home_angle = np.arctan2(hy - cy, hx - cx)
        home_half  = np.deg2rad(45.0)
        home_end   = home_angle + home_half
        rel        = (angle - home_end) % (2 * np.pi)
        if rel >= 2 * np.pi - 2 * home_half:
            return None   # inside home zone
        return min(int(rel / np.deg2rad(30)), 8)

    # ─────────────────────────────────────────────────────────────────────────
    # Simulation gate visualisation (unchanged from original)
    # ─────────────────────────────────────────────────────────────────────────

    def find_gates(self, frame: MatLike, measurement: Measurement, flags: Telemetry.Flags) -> list[Gate]:
        if Telemetry.Flags.SIMULATION in flags:
            gates_points = []
            v = view(measurement.position, euler_to_quaternion(measurement.rotation))
            for gate in self.sim_gates:
                if np.abs(wrap(measurement.rotation.z - gate.yaw)) > np.pi / 2:
                    continue
                normal = glm.vec3(np.cos(gate.yaw), np.sin(gate.yaw), 0.0)
                right  = np.cross(UP, normal)
                size   = Gate.HEIGHT / 2
                world  = [
                    gate.position - size * UP - size * right,
                    gate.position - size * UP + size * right,
                    gate.position + size * UP + size * right,
                    gate.position + size * UP - size * right,
                ]
                clip = world2clip(v, world)
                if sum(any(plane(c) < 0 for plane in CLIP_PLANES) for c in clip) >= 1:
                    continue
                screen = [clip2screen(x) for x in clip]
                screen = [glm.clamp(s, glm.vec2(0.0), glm.vec2(WIDTH, HEIGHT)) for s in screen]
                gates_points.append(screen)
            gates_points = np.array(gates_points)
        else:
            rgb         = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) if len(frame.shape) == 2 else frame
            predictions = self.model.predict(rgb, conf=Scan.YOLO_CONF, iou=Scan.YOLO_IOU, verbose=False)
            if len(predictions) != 1:
                return []
            prediction = predictions[0]
            if not hasattr(prediction, "keypoints"):
                return []
            gates_points = prediction.keypoints.cpu().xy.numpy()

        if gates_points.size == 0:
            self.gates_detected_event([])
            return []
        gates = [Gate(corners, measurement) for corners in gates_points]
        gates.sort(key=lambda g: g.distance)
        return gates
