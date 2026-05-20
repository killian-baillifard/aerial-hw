import os, cv2
from enum import Enum
import numpy as np
from pyglm import glm
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

class Scan(Planner):

    SCAN_YAWS = [-45, 0, 75, 130, 180, 180]
    INITIAL_SETPOINT = Setpoint(Planner.HOME_POSITION, np.deg2rad(SCAN_YAWS[0]))
    STABILIZATION_TIMEOUT = 2.0  # s
    GATE_PASS_DIST = 0.10        # m

    # Alignment tuning
    ALTITUDE_KP         = 0.002  # vertical error → z correction
    YAW_KP              = 0.015  # horizontal pixel error → yaw correction
    FORWARD_STEP        = 0.15   # m pushed ahead per tick in FORWARD state
    ORBIT_BASE_SPEED    = 9.0    # orbit strafe numerator (divided by gate height)
    ALTITUDE_THRESH     = 15.0   # px — vertical error band to leave ALTITUDE state
    ALTITUDE_OVERRIDE   = 25.0   # px — vertical error that forces re-entry to ALTITUDE
    ALIGN_TOLERANCE     = 10.0   # px — h_left/h_right diff band to leave ORBIT state
    BLIND_YAW_RATE      = 0.2    # rad/tick — spin speed during blind search
    BLIND_DRIFT_SPEED   = 0.15   # m/tick — drift toward centre during blind search
    BLIND_CENTER_RADIUS = 0.5    # m — dead-zone radius around room centre
    BLIND_Z_TARGET      = 0.70   # m — hover altitude during blind search
    BLIND_Z_KP          = 0.8    # blind altitude correction gain

    class State(Enum):
        REACH_WAYPOINT  = 0
        STABILIZE       = 1
        ALIGN           = 2
        END             = 3

    class AlignState(Enum):
        ALTITUDE = "ALTITUDE"
        FORWARD  = "FORWARD"
        ORBIT    = "ORBIT"

    def __init__(self) -> None:
        super().__init__()
        self.model = YOLO(MODEL_PATH)
        self.model.eval()
        self.waypoints.append(Scan.INITIAL_SETPOINT)
        self.state       = Scan.State.REACH_WAYPOINT
        self.align_state = Scan.AlignState.ALTITUDE
        self.stabilization_timeout = 0.0
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
        self.waypoints.append(Scan.INITIAL_SETPOINT)
        self.state       = Scan.State.REACH_WAYPOINT
        self.align_state = Scan.AlignState.ALTITUDE
        self.stabilization_timeout = 0.0
        self.load_sim()

    @overrides
    def update(self, measurement: Measurement, frame: MatLike, flags: Telemetry.Flags, dt: float) -> Setpoint:

        # Interpolate toward last waypoint
        setpoint, reached = Planner.reach(self.waypoints[-1], measurement, 0.5)

        # Always run detection so the visualiser stays current
        if Telemetry.Flags.NEW_FRAME in flags:
            gates = self.find_gates(frame, measurement, flags)
            self.gates_detected_event(gates)
        else:
            gates = []

        match self.state:

            # ── 0. REACH_WAYPOINT ─────────────────────────────────────────────
            case Scan.State.REACH_WAYPOINT:
                if reached:
                    if len(self.gates) < 5:
                        self.state = Scan.State.STABILIZE
                        return self.update(measurement, frame, flags, dt)
                    else:
                        self.waypoints.append(Planner.HOME_SETPOINT)
                        self.state = Scan.State.END
                        return self.update(measurement, frame, flags, dt)
                return setpoint

            # ── 1. STABILIZE ──────────────────────────────────────────────────
            case Scan.State.STABILIZE:
                self.stabilization_timeout += dt
                if self.stabilization_timeout > Scan.STABILIZATION_TIMEOUT:
                    self.stabilization_timeout = 0.0
                    self.align_state = Scan.AlignState.ALTITUDE   # always reset on entry
                    self.state = Scan.State.ALIGN
                return setpoint

            # ── 2. ALIGN (closed-loop active-vision state machine) ────────────
            case Scan.State.ALIGN:
                return self._align(gates, measurement, setpoint)

            # ── 3. END ────────────────────────────────────────────────────────
            case Scan.State.END:
                return setpoint

    # ─────────────────────────────────────────────────────────────────────────
    def _align(self, gates: list[Gate], measurement: Measurement, hold: Setpoint) -> Setpoint:
        """
        Active-vision alignment state machine ported from closed_loop_control.
        Returns a Setpoint on every tick; commits a confirmed gate to
        self.gates / self.waypoints and transitions to REACH_WAYPOINT once a
        gate has been found and approached.
        """
        pos = measurement.position          # glm.vec3
        rot = measurement.rotation          # glm.vec3  (roll, pitch, yaw)
        x, y, z   = pos.x, pos.y, pos.z
        yaw       = rot.z

        new_x, new_y, new_z, new_yaw = x, y, z, yaw

        if gates:
            # ── Gate visible: run pixel-space alignment ───────────────────────
            gate = gates[0]   # closest detection

            # Reconstruct 2-D corners from the Gate object.
            # Gate stores its screen corners; we derive the bounding quad.
            corners = gate.corners  # list/array of four glm.vec2 screen points
            tl = np.array([corners[3][0], corners[3][1]])  # top-left
            tr = np.array([corners[2][0], corners[2][1]])  # top-right
            bl = np.array([corners[0][0], corners[0][1]])  # bottom-left
            br = np.array([corners[1][0], corners[1][1]])  # bottom-right

            h_left  = np.linalg.norm(tl - bl)
            h_right = np.linalg.norm(tr - br)
            max_h   = max(h_left, h_right)

            avg_u = (tl[0] + tr[0] + bl[0] + br[0]) / 4.0
            avg_v = (tl[1] + tr[1] + bl[1] + br[1]) / 4.0

            # YAW alignment (always active regardless of align_state)
            center_x  = WIDTH  / 2.0
            yaw_error = center_x - avg_u
            new_yaw   = yaw + yaw_error * Scan.YAW_KP

            # Vertical and horizontal errors
            center_y       = HEIGHT / 2.0
            vertical_error = center_y - avg_v
            height_diff    = h_left - h_right

            # Safety override: severe altitude drift → back to ALTITUDE
            if abs(vertical_error) > Scan.ALTITUDE_OVERRIDE:
                self.align_state = Scan.AlignState.ALTITUDE

            if self.align_state == Scan.AlignState.ALTITUDE:
                new_z = z + vertical_error * Scan.ALTITUDE_KP
                if abs(vertical_error) <= Scan.ALTITUDE_THRESH:
                    self.align_state = Scan.AlignState.FORWARD

            elif self.align_state == Scan.AlignState.FORWARD:
                new_x = x + Scan.FORWARD_STEP * np.cos(yaw)
                new_y = y + Scan.FORWARD_STEP * np.sin(yaw)
                if abs(height_diff) > Scan.ALIGN_TOLERANCE:
                    self.align_state = Scan.AlignState.ORBIT

            elif self.align_state == Scan.AlignState.ORBIT:
                if abs(height_diff) > Scan.ALIGN_TOLERANCE:
                    strafe_speed = Scan.ORBIT_BASE_SPEED / (max_h + 1e-5)
                    if height_diff > 0:    # drone is left of centre → strafe right
                        new_x = x + strafe_speed * np.sin(yaw)
                        new_y = y - strafe_speed * np.cos(yaw)
                    else:                  # drone is right of centre → strafe left
                        new_x = x - strafe_speed * np.sin(yaw)
                        new_y = y + strafe_speed * np.cos(yaw)
                else:
                    self.align_state = Scan.AlignState.FORWARD

            # ── Gate confirmation: once we are close enough, commit it ────────
            if gate.distance <= Scan.GATE_PASS_DIST * 2:
                target_yaw      = np.deg2rad(Scan.SCAN_YAWS[len(self.waypoints)])
                target_position = gate.position + gate.normal * Scan.GATE_PASS_DIST
                next_setpoint   = Setpoint(target_position, target_yaw)

                self.gates.append(next_setpoint)
                self.waypoints.append(next_setpoint)

                self.state = Scan.State.REACH_WAYPOINT

        else:
            # ── Blind search: spin + drift toward room centre ─────────────────
            self.align_state = Scan.AlignState.ALTITUDE   # reset for next gate

            new_yaw = yaw + Scan.BLIND_YAW_RATE

            # Drift toward room centre so the camera sweeps fresh angles
            center_x     = getattr(self, 'room_x', 0.0) / 2.0
            center_y     = getattr(self, 'room_y', 0.0) / 2.0
            vector_x     = center_x - x
            vector_y     = center_y - y
            dist_to_center = np.hypot(vector_x, vector_y)

            if dist_to_center > Scan.BLIND_CENTER_RADIUS:
                new_x = x + (vector_x / dist_to_center) * Scan.BLIND_DRIFT_SPEED
                new_y = y + (vector_y / dist_to_center) * Scan.BLIND_DRIFT_SPEED

            if z < Scan.BLIND_Z_TARGET - 0.05 or z > Scan.BLIND_Z_TARGET + 0.05:
                new_z = z + (Scan.BLIND_Z_TARGET - z) * Scan.BLIND_Z_KP

        return Setpoint(glm.vec3(new_x, new_y, new_z), new_yaw)

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
    