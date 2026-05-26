import os, cv2
from enum import Enum
import numpy as np
from pyglm import glm
from cv2.typing import MatLike
from overrides import overrides
from app import wrap, no_network
from app.io import Measurement, Setpoint
from app.planner import Planner
from app.telemetry import Telemetry
from app.telemetry.gate import Gate
from app.telemetry.camera import (
    UP, world2clip, clip2screen, CLIP_PLANES, WIDTH, HEIGHT,
    view, euler_to_quaternion,
)

with no_network():
    from ultralytics import YOLO

MODEL_PATH = os.path.join(
    "controller_detection", "detection_model",
    "models", "yolov8n_v3bw_r1", "weights", "best.pt",
)
ENABLE_TAS_REF_ANGLE = True

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Yaw sweep rate – slow enough that a 40 cm gate stays in frame for several
# frames even at the closest expected distance (~0.8 m).
YAW_SWEEP_RATE = np.deg2rad(6)          # rad/s

# A detected gate is only accepted when its pixel centre sits within the
# central (1 - 2*MARGIN) fraction of the frame width.  Gates near the edge
# have large pose uncertainty; we wait for the sweep to bring them to centre.
EDGE_MARGIN = 0.20                       # fraction of frame width each side

# Gate must pass the edge test this many consecutive frames before we commit.
CONFIRM_FRAMES = 2

# Gate height used for simulation projection
GATE_HALF_HEIGHT = 0.20                  # m  (40 cm total / 2)

# Plausible Z range for a real gate centre:  1.3 m ± 0.4 m
GATE_Z_MIN = 0.60                        # m
GATE_Z_MAX = 2.00                        # m

# ---------------------------------------------------------------------------

class ScanKillianS(Planner):
    """
    One fixed scan position per gate.  At each position the drone performs a
    slow yaw sweep from the zone's outer edge toward the arena centre so that:
      • the target gate zone is fully covered,
      • opposite gates (180° away) are never in view during the sweep.

    Gate acceptance requires:
      • pixel centre inside the central 60 % of the frame (edge rejection),
      • CONFIRM_FRAMES consecutive passing frames (false-positive rejection),
      • estimated 3-D centre within the expected Z band (height sanity check).

    All scan positions share the same flight height (SCAN_Z) so no Z
    adjustment is needed between gates.
    """

    SCAN_Z         = 1.3    # m
    GATE_PASS_DIST = 0.10   # m past gate centre

    # ------------------------------------------------------------------
    # Per-gate configuration
    #   scan_position : where the drone hovers during the sweep
    #   sweep_start   : yaw at which the sweep begins (outer edge of zone)
    #   sweep_end     : yaw at which the sweep ends   (inner / centre side)
    #
    # Geometry:  arena centre ≈ (1.1, 0.0).
    # For each scan point we compute the bearing *toward* the arena centre;
    # the sweep is centred on that bearing with ±20° margin, running from
    # the outer edge (start) to the inner edge (end).
    #
    #   Gate 0  scan (-0.28,-0.25)  bearing to centre ≈  10°  sweep  -10°→ 30°
    #   Gate 1  scan ( 0.95,-1.00)  bearing to centre ≈  82°  sweep   61°→101°
    #   Gate 2  scan ( 2.34,-1.00)  bearing to centre ≈ 141°  sweep  121°→161°
    #   Gate 3  scan ( 2.58, 0.54)  bearing to centre ≈-160°  sweep -180°→-140°
    #   Gate 4  scan ( 1.36, 1.05)  bearing to centre ≈-104°  sweep -124°→ -84°
    # ------------------------------------------------------------------
    SCAN_CONFIGS = [
        # (scan_position,                         sweep_start_deg, sweep_end_deg)
        (glm.vec3(-0.28, -0.25, SCAN_Z), -75, -20),
        (glm.vec3( 0.95, -1.00, SCAN_Z), -30, 30),
        (glm.vec3( 2.34, -1.00, SCAN_Z), 65, 110),
        (glm.vec3( 2.65,  0.60, SCAN_Z), 140, 180),
        (glm.vec3( 1.38,  1.13, SCAN_Z), 170, 230),
    ]

    INITIAL_SETPOINT      = Setpoint(Planner.HOME_POSITION, np.deg2rad(-45))
    STABILIZATION_TIMEOUT = 2.0   # s

    # Arena bounding box – computed once from room corners
    _BB_CORNERS = [
        glm.vec3(-0.519, -1.009, 1.26),
        glm.vec3( 2.874, -1.288, 1.31),
        glm.vec3( 2.900,  1.163, 1.32),
        glm.vec3(-0.780,  1.191, 1.23),
    ]
    BB_MIN = glm.vec3(min(p.x for p in _BB_CORNERS),
                      min(p.y for p in _BB_CORNERS),
                      min(p.z for p in _BB_CORNERS))
    BB_MAX = glm.vec3(max(p.x for p in _BB_CORNERS),
                      max(p.y for p in _BB_CORNERS),
                      max(p.z for p in _BB_CORNERS))

    class State(Enum):
        REACH_SCAN_POS = 0
        STABILIZE      = 1
        YAW_SWEEP      = 2
        REACH_GATE     = 3
        END            = 4

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        super().__init__()
        self.model = YOLO(MODEL_PATH)
        self.model.eval()
        self.sim_gates: list[Setpoint] = []
        self.load_sim()
        self._reset_state()

    # ------------------------------------------------------------------ state
    def _reset_state(self) -> None:
        self.waypoints.clear()
        self.gates: list[Setpoint] = []
        self.waypoints.append(ScanKillianS.INITIAL_SETPOINT)
        self.state               = ScanKillianS.State.REACH_SCAN_POS
        self.stabilization_timer = 0.0
        self.sweep_yaw           = 0.0
        self.sweep_start         = 0.0
        self.sweep_end           = 0.0
        self.confirm_count       = 0
        self.best_gate: Gate | None = None

    def _gate_index(self) -> int:
        return len(self.gates)

    def _begin_scan(self, idx: int) -> None:
        pos, start_deg, end_deg = ScanKillianS.SCAN_CONFIGS[idx]
        self.sweep_start = np.deg2rad(start_deg)
        self.sweep_end   = np.deg2rad(end_deg)
        self.sweep_yaw   = self.sweep_start
        self.confirm_count = 0
        self.best_gate     = None
        self.waypoints.append(Setpoint(pos, self.sweep_start))

    # ----------------------------------------------------------------- CSV IO
    @staticmethod
    def _is_float(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    def load_sim(self) -> None:
        gate_dir = "gates"
        files = sorted(os.listdir(gate_dir))
        if not files:
            self.sim_gates = []
            return
        fpath = os.path.join(gate_dir, files[0])
        with open(fpath, encoding="utf-8") as f:
            first = f.readline()
        tokens = [t.strip() for t in first.split(",") if t.strip()]
        has_header = any(not self._is_float(t) for t in tokens)
        raw = np.genfromtxt(
            fpath, delimiter=",", dtype=float, ndmin=2,
            skip_header=1 if has_header else 0,
            filling_values=np.nan,
        )
        if raw.size == 0 or np.isnan(np.atleast_1d(raw[0])).any():
            self.sim_gates = []
            return
        positions = [glm.vec3(c[1], c[2], c[3]) for c in raw]
        yaws = [wrap(c[4] - np.pi if ENABLE_TAS_REF_ANGLE else c[4]) for c in raw]
        self.sim_gates = [Setpoint(p, y) for p, y in zip(positions, yaws)]

    # -------------------------------------------------------------- Planner API
    @overrides
    def reload(self) -> None:
        self.load_sim()
        self._reset_state()

    @overrides
    def update(self, measurement: Measurement, frame: MatLike,
               flags: Telemetry.Flags, dt: float) -> Setpoint:

        setpoint, reached = Planner.reach(self.waypoints[-1], measurement, 0.25)

        if Telemetry.Flags.NEW_FRAME in flags:
            gates = self.find_gates(frame, measurement, flags)
            self.gates_detected_event(gates)
        else:
            gates = []

        idx = self._gate_index()

        match self.state:

            # ------------------------------------------------- fly to scan pos
            case ScanKillianS.State.REACH_SCAN_POS:
                if reached:
                    if idx < 5:
                        self._begin_scan(idx)
                        self.state = ScanKillianS.State.STABILIZE
                        return self.update(measurement, frame, flags, dt)
                    self.waypoints.append(Planner.HOME_SETPOINT)
                    self.state = ScanKillianS.State.END
                    return self.update(measurement, frame, flags, dt)
                return setpoint

            # ---------------------------------------------- wait for calm hover
            case ScanKillianS.State.STABILIZE:
                setpoint.position.z = ScanKillianS.SCAN_Z
                if reached:
                    self.stabilization_timer += dt
                    if self.stabilization_timer >= ScanKillianS.STABILIZATION_TIMEOUT:
                        self.stabilization_timer = 0.0
                        self.state = ScanKillianS.State.YAW_SWEEP
                return setpoint

            # --------------------------------------------------- slow yaw sweep
            case ScanKillianS.State.YAW_SWEEP:
                # Advance yaw toward sweep_end
                self.sweep_yaw = min(
                    self.sweep_yaw + YAW_SWEEP_RATE * dt,
                    self.sweep_end,
                )
                pos, _, _ = ScanKillianS.SCAN_CONFIGS[idx]
                sweep_sp = Setpoint(glm.vec3(pos), self.sweep_yaw)

                # Check acceptance criteria
                candidate = self._best_gate(gates)
                if candidate is not None:
                    self.confirm_count += 1
                    self.best_gate = candidate
                else:
                    self.confirm_count = 0

                if self.confirm_count >= CONFIRM_FRAMES and self.best_gate is not None:
                    self._commit_gate(self.best_gate)
                    return self.update(measurement, frame, flags, dt)

                # Sweep finished with no valid gate — go home gracefully
                if self.sweep_yaw >= self.sweep_end:
                    self.waypoints.append(Planner.HOME_SETPOINT)
                    self.state = ScanKillianS.State.END

                return sweep_sp

            # ------------------------------------------- fly through the gate
            case ScanKillianS.State.REACH_GATE:
                if reached:
                    self.state = ScanKillianS.State.REACH_SCAN_POS
                    return self.update(measurement, frame, flags, dt)
                return setpoint

            case ScanKillianS.State.END:
                return setpoint

    # --------------------------------------------------------- gate filter
    def _best_gate(self, gates: list[Gate]) -> Gate | None:
        """
        Return the closest gate that passes all acceptance criteria:
          1. Pixel centre within central (1 - 2*EDGE_MARGIN) of frame width.
          2. Estimated 3-D centre Z within [GATE_Z_MIN, GATE_Z_MAX].
        Gates are already sorted by distance (closest first).
        """
        left  = WIDTH  * EDGE_MARGIN
        right = WIDTH  * (1.0 - EDGE_MARGIN)

        for gate in gates:
            # --- edge test (pixel centre) ---
            cx = float(np.mean([c[0] for c in gate.corners]))
            if not (left <= cx <= right):
                continue

            # --- height sanity check ---
            if not (GATE_Z_MIN <= gate.position.z <= GATE_Z_MAX):
                continue

            return gate
        return None

    # ------------------------------------------------------- commit gate
    def _commit_gate(self, gate: Gate) -> None:
        raw_pos = gate.position + gate.normal * (ScanKillianS.GATE_PASS_DIST)
        clamped = glm.vec3(
            float(np.clip(raw_pos.x, ScanKillianS.BB_MIN.x, ScanKillianS.BB_MAX.x)),
            float(np.clip(raw_pos.y, ScanKillianS.BB_MIN.y, ScanKillianS.BB_MAX.y)),
            raw_pos.z,
        )
        sp = Setpoint(clamped, gate.yaw)
        self.gates.append(sp)
        self.waypoints.append(sp)
        self.confirm_count = 0
        self.best_gate     = None
        self.state         = ScanKillianS.State.REACH_GATE

    # --------------------------------------------------------- find_gates
    def find_gates(self, frame: MatLike, measurement: Measurement,
                   flags: Telemetry.Flags) -> list[Gate]:

        if Telemetry.Flags.SIMULATION in flags:
            gates_points = []
            v = view(measurement.position,
                     euler_to_quaternion(measurement.rotation))
            for gate in self.sim_gates:
                # Skip gates whose face points away from the camera
                if np.abs(wrap(measurement.rotation.z - gate.yaw)) > np.pi / 2:
                    continue
                normal = glm.vec3(np.cos(gate.yaw), np.sin(gate.yaw), 0.0)
                right  = np.cross(UP, normal)
                world  = [
                    gate.position - GATE_HALF_HEIGHT * UP - GATE_HALF_HEIGHT * right,
                    gate.position - GATE_HALF_HEIGHT * UP + GATE_HALF_HEIGHT * right,
                    gate.position + GATE_HALF_HEIGHT * UP + GATE_HALF_HEIGHT * right,
                    gate.position + GATE_HALF_HEIGHT * UP - GATE_HALF_HEIGHT * right,
                ]
                clip = world2clip(v, world)
                if sum(any(p(c) < 0 for p in CLIP_PLANES) for c in clip) >= 3:
                    continue
                screen = [clip2screen(x) for x in clip]
                screen = [glm.clamp(s, glm.vec2(0, 0), glm.vec2(WIDTH, HEIGHT))
                          for s in screen]
                gates_points.append(screen)
            gates_points = np.array(gates_points)

        else:
            frame       = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            predictions = self.model.predict(frame, conf=0.5, iou=0.7, verbose=False)
            if len(predictions) != 1:
                return []
            prediction = predictions[0]
            if not hasattr(prediction, "keypoints"):
                return []
            gates_points = prediction.keypoints.cpu().xy.numpy()

        if gates_points.size == 0:
            self.gates_detected_event([])
            return []

        # Sort closest first (same as original)
        gates = [Gate(corners, measurement) for corners in gates_points]
        gates.sort(key=lambda g: g.distance)
        return gates