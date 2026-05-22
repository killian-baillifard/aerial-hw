import os
import cv2
from enum import Enum
import numpy as np
from pyglm import glm
from cv2.typing import MatLike
from overrides import overrides
from app import wrap
from app.io import Measurement, Setpoint
from app.planner import Planner
from app.telemetry import Telemetry
from app.telemetry.gate import Gate
from app.telemetry.camera import UP, world2clip, clip2screen, CLIP_PLANES, WIDTH, HEIGHT, view, euler_to_quaternion
from ultralytics import YOLO

# ── BULLETPROOF MODEL PATH ───────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "..", "controller_detection", "detection_model", "models", "yolov8n_v3bw_r1", "weights", "best.pt")

class ScanVincent(Planner):

    # Waypoints and logic bounds
    SCAN_YAWS = [-45, 0, 75, 130, 180, 180]
    INITIAL_SETPOINT = Setpoint(Planner.HOME_POSITION, np.deg2rad(SCAN_YAWS[0]))
    STABILIZATION_TIMEOUT = 5.0  # s
    
    # ── DISCRETE STEP SETTINGS (Tuned for low FPS) ──
    # We no longer use speeds (m/s). We use absolute step distances.
    YAW_TOLERANCE     = 25.0     # px — Error required to trigger a Yaw fix
    Z_TOLERANCE       = 20.0     # px — Error required to trigger an Alt fix
    ORBIT_TOLERANCE   = 15.0     # px — Asymmetry required to trigger a Strafe
    
    STEP_FORWARD      = 0.4      # m — How far to jump forward per decision
    MAX_STEP_STRAFE   = 0.3      # m — Maximum lateral jump to fix orbit
    MAX_STEP_Z        = 0.2      # m — Maximum vertical jump
    MAX_STEP_YAW      = 0.35     # rad (~20 deg) — Maximum spin per decision
    
    COMMIT_DISTANCE   = 0.9      # m — Once we are within 90cm, blindly pass through
    BLIND_SPIN_STEP   = 0.34     # rad — Spin chunk if searching blindly
    
    class State(Enum):
        REACH_WAYPOINT  = 0
        STABILIZE       = 1
        ALIGN           = 2
        END             = 3

    def __init__(self) -> None:
        super().__init__()
        self.model = YOLO(MODEL_PATH)
        self.model.eval()
        self.waypoints.append(ScanVincent.INITIAL_SETPOINT)
        
        self.state = ScanVincent.State.REACH_WAYPOINT
        self.stabilization_timeout = 0.0
        
        # ── DISCRETE MEMORY ──
        # This locks in our decision between the slow camera frames
        self.target_setpoint = None
        self.gate_lost_timer = 0.0

        self.load_sim()

    def load_sim(self) -> None:
        gates_directory = os.path.join("gates")
        file_name  = os.listdir(gates_directory)[0]
        file_path  = os.path.join(gates_directory, file_name)
        raw_csv_data: np.ndarray = np.genfromtxt(file_path, delimiter=',')[1:]
        if not any(np.isnan(raw_csv_data[0])):
            positions: list[glm.vec3] = [glm.vec3(col[1], col[2], col[3]) for col in raw_csv_data]
            yaws: list[float]         = [wrap(col[4]) for col in raw_csv_data]
            self.sim_gates            = [Setpoint(position, yaw) for position, yaw in zip(positions, yaws)]

    @overrides
    def reload(self) -> None:
        self.waypoints.clear()
        self.gates.clear()
        self.waypoints.append(ScanVincent.INITIAL_SETPOINT)
        self.state = ScanVincent.State.REACH_WAYPOINT
        self.stabilization_timeout = 0.0
        self.target_setpoint = None
        self.load_sim()

    @overrides
    def update(self, measurement: Measurement, frame: MatLike, flags: Telemetry.Flags, dt: float) -> Setpoint:

        # Interpolate toward last waypoint
        setpoint, reached = Planner.reach(self.waypoints[-1], measurement, 0.5)

        # Always tick the blind timer
        self.gate_lost_timer += dt

        # Only process vision if a genuinely new frame has arrived (3 FPS)
        if Telemetry.Flags.NEW_FRAME in flags:
            gates = self.find_gates(frame, measurement, flags)
            self.gates_detected_event(gates)
        else:
            gates = None # We use None to signify "no new data", not "empty"

        match self.state:
            # ── 0. REACH_WAYPOINT ─────────────────────────────────────────────
            case ScanVincent.State.REACH_WAYPOINT:
                if reached:
                    if len(self.gates) < 5:
                        self.state = ScanVincent.State.STABILIZE
                        return self.update(measurement, frame, flags, dt)
                    else:
                        self.waypoints.append(Planner.HOME_SETPOINT)
                        self.state = ScanVincent.State.END
                        return self.update(measurement, frame, flags, dt)
                return setpoint

            # ── 1. STABILIZE ──────────────────────────────────────────────────
            case ScanVincent.State.STABILIZE:
                self.stabilization_timeout += dt
                if self.stabilization_timeout > ScanVincent.STABILIZATION_TIMEOUT:
                    self.stabilization_timeout = 0.0
                    self.target_setpoint = None # Reset alignment memory
                    self.state = ScanVincent.State.ALIGN
                return setpoint

            # ── 2. ALIGN (Discrete Decision Tree) ─────────────────────────────
            case ScanVincent.State.ALIGN:
                return self._align(gates, measurement, flags)

            # ── 3. END ────────────────────────────────────────────────────────
            case ScanVincent.State.END:
                return setpoint

    # ─────────────────────────────────────────────────────────────────────────
    def _align(self, gates: list[Gate] | None, measurement: Measurement, flags: Telemetry.Flags) -> Setpoint:
        """
        Stop-and-Stare Decision Tree.
        Only makes a movement decision when a NEW frame arrives. Otherwise, outputs
        the same static setpoint so the drone stabilizes in place.
        """
        pos = measurement.position
        rot = measurement.rotation
        x, y, z = pos.x, pos.y, pos.z
        yaw = rot.z

        # Initialize the holding setpoint on the first run
        if self.target_setpoint is None:
            self.target_setpoint = Setpoint(glm.vec3(x, y, z), yaw)

        # IF WE DO NOT HAVE A NEW FRAME -> Just keep stabilizing at current target
        if Telemetry.Flags.NEW_FRAME not in flags:
            return self.target_setpoint

        # ─── WE HAVE A NEW FRAME ───
        
        if gates:
            # We see a gate! Reset the blind timer.
            self.gate_lost_timer = 0.0
            gate = gates[0]

            corners = gate.corners  
            tl = np.array([corners[3][0], corners[3][1]])  
            tr = np.array([corners[2][0], corners[2][1]])  
            bl = np.array([corners[0][0], corners[0][1]])  
            br = np.array([corners[1][0], corners[1][1]])  

            h_left  = np.linalg.norm(tl - bl)
            h_right = np.linalg.norm(tr - br)
            
            avg_u = (tl[0] + tr[0] + bl[0] + br[0]) / 4.0
            avg_v = (tl[1] + tr[1] + bl[1] + br[1]) / 4.0

            # Calculate raw errors
            yaw_error = (WIDTH / 2.0) - avg_u
            z_error   = (HEIGHT / 2.0) - avg_v
            orbit_err = h_left - h_right

            # Base our next decision on our CURRENT physical position so errors don't compound
            tx, ty, tz = x, y, z
            tyaw = yaw

            # ── THE DECISION TREE (Prioritize one discrete fix at a time) ──

            # 1. YAW is the most important. If we aren't looking at it, fix that first.
            if abs(yaw_error) > ScanVincent.YAW_TOLERANCE:
                step = np.clip(yaw_error * 0.005, -ScanVincent.MAX_STEP_YAW, ScanVincent.MAX_STEP_YAW)
                tyaw = yaw + step
                print(f"[Decision] Fix Yaw: {np.degrees(step):.1f} deg")

            # 2. ALTITUDE is next. If we are too high/low, fix that.
            elif abs(z_error) > ScanVincent.Z_TOLERANCE:
                step = np.clip(z_error * 0.003, -ScanVincent.MAX_STEP_Z, ScanVincent.MAX_STEP_Z)
                tz = z + step
                print(f"[Decision] Fix Alt: {step:.2f} m")

            # 3. ORBIT (Lateral alignment). If left/right legs are asymmetrical, strafe.
            elif abs(orbit_err) > ScanVincent.ORBIT_TOLERANCE:
                step = np.clip(orbit_err * 0.008, -ScanVincent.MAX_STEP_STRAFE, ScanVincent.MAX_STEP_STRAFE)
                # Positive orbit_err means left leg is taller -> we are too far right -> strafe left
                tx = x - step * np.sin(yaw)
                ty = y + step * np.cos(yaw)
                print(f"[Decision] Fix Orbit: Strafe {step:.2f} m")

            # 4. If all alignments are within tolerance, take a clean STEP FORWARD.
            else:
                tx = x + ScanVincent.STEP_FORWARD * np.cos(yaw)
                ty = y + ScanVincent.STEP_FORWARD * np.sin(yaw)
                print(f"[Decision] STEP FORWARD {ScanVincent.STEP_FORWARD}m")

            # Lock in the decision
            self.target_setpoint = Setpoint(glm.vec3(tx, ty, tz), tyaw)

            # ── COMMIT CONDITION ──
            # Because we step in 40cm chunks, checking for <= 10cm is dangerous (we might step over it).
            # Instead, once we are within 90cm and centered, commit to the pass-through.
            if gate.distance <= ScanVincent.COMMIT_DISTANCE:
                print(">>> COMMITTING TO PASS GATE <<<")
                target_yaw      = np.deg2rad(ScanVincent.SCAN_YAWS[len(self.waypoints)])
                target_position = gate.position + gate.normal * 0.4 # Push 40cm out the back of the gate
                next_setpoint   = Setpoint(target_position, target_yaw)

                self.gates.append(next_setpoint)
                self.waypoints.append(next_setpoint)
                
                self.target_setpoint = None # Clear memory
                self.state = ScanVincent.State.REACH_WAYPOINT
                return next_setpoint

        else:
            # ── BLIND SEARCH ──
            # We haven't seen a gate in a while. Take discrete spins.
            if self.gate_lost_timer > 1.5:
                print("[Decision] Blind Spin")
                self.gate_lost_timer = 0.0 # reset timer so we wait before spinning again
                
                # Spin in place
                tyaw = yaw + ScanVincent.BLIND_SPIN_STEP
                self.target_setpoint = Setpoint(glm.vec3(x, y, z), tyaw)

        return self.target_setpoint

    # ── Detection (unchanged) ─────────────────────────────────────────────────
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
                screen = [glm.clamp(s, glm.vec2(0.0, 0.0), glm.vec2(WIDTH, HEIGHT)) for s in screen]
                gates_points.append(screen)

            gates_points = np.array(gates_points)

        else:
            frame       = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            # Add imgsz=320 here if you want YOLO to run even faster on the live hardware
            predictions = self.model.predict(frame, conf=0.5, iou=0.7, verbose=False)

            if len(predictions) != 1:
                return []

            prediction = predictions[0]
            if not hasattr(prediction, "keypoints"):
                return []

            keypoints    = prediction.keypoints.cpu()
            gates_points = keypoints.xy.numpy()

        if gates_points.size == 0:
            self.gates_detected_event([])
            return []

        gates = [Gate(corners, measurement) for corners in gates_points]
        gates.sort(key=lambda gate: gate.distance)
        return gates