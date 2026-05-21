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
ENABLE_TAS_REF_ANGLE = True

class ScanKillian1(Planner):

    SCAN_YAWS = [-45, 0, 75, 135, 180, 180]
    INITIAL_SETPOINT = Setpoint(Planner.HOME_POSITION, np.deg2rad(SCAN_YAWS[0]))
    STABILIZATION_TIMEOUT = 4.0 # s
    GATE_PASS_DIST = 0.10 # m

    class State(Enum):
        REACH_WAYPOINT = 0
        STABILIZE = 1
        FIND_GATE = 2
        END = 3

    def __init__(self) -> None:
        super().__init__()
        self.model = YOLO(MODEL_PATH)
        self.model.eval()
        self.waypoints.append(ScanKillian1.INITIAL_SETPOINT)
        self.state = ScanKillian1.State.REACH_WAYPOINT
        self.stabilization_timeout = 0.0
        self.gates = []
        self.sim_gates = []
        self.load_sim()

    def load_sim(self) -> None:
        gates_directory = os.path.join("gates")
        file_name = os.listdir(gates_directory)[0]
        file_path = os.path.join(gates_directory, file_name)
        with open(file_path, 'r', encoding='utf-8') as _f:
            first_line = _f.readline()
        tokens = [t.strip() for t in first_line.split(',') if t.strip() != '']
        has_header = False
        for t in tokens:
            try:
                float(t)
            except Exception:
                has_header = True
                break
        if has_header:
            raw_csv_data = np.genfromtxt(file_path, delimiter=',', dtype=float, ndmin=2,
                                         skip_header=1, filling_values=np.nan)
        else:
            raw_csv_data = np.genfromtxt(file_path, delimiter=',', dtype=float, ndmin=2,
                                         filling_values=np.nan)

        # If the CSV is empty, genfromtxt returns an empty array — handle that gracefully.
        if raw_csv_data.size == 0:
            self.gates = []
            self.sim_gates = []
            return

        # ensure we have an array (handles scalar rows) and test for any NaNs
        row0 = np.atleast_1d(raw_csv_data[0])
        if not np.isnan(row0).any():
            positions: list[glm.vec3]   = [glm.vec3(col[1], col[2], col[3]) for col in raw_csv_data]
            if ENABLE_TAS_REF_ANGLE:
                yaws: list[float] = [wrap(col[4] - np.pi) for col in raw_csv_data]
            else:
                yaws: list[float] = [wrap(col[4]) for col in raw_csv_data]
            self.sim_gates              = [Setpoint(position, yaw) for position, yaw in zip(positions, yaws)]

    @overrides
    def reload(self) -> None:
        self.waypoints.clear()
        self.gates.clear()
        self.waypoints.append(ScanKillian1.INITIAL_SETPOINT)
        self.state = ScanKillian1.State.REACH_WAYPOINT
        self.stabilization_timeout = 0.0
        self.load_sim()

    @overrides
    def update(self, measurement: Measurement, frame: MatLike, flags: Telemetry.Flags, dt: float) -> Setpoint:

        # Automatic interpolation to last waypoint in list
        setpoint, reached = Planner.reach(self.waypoints[-1], measurement, 0.25)

        # Run inference on each incoming image (for visualization purposes)
        if Telemetry.Flags.NEW_FRAME in flags:
            gates = self.find_gates(frame, measurement, flags)
            self.gates_detected_event(gates)
        else:
            gates = []

        match self.state:

            case ScanKillian1.State.REACH_WAYPOINT:

                # Waypoint reached
                if reached:

                    # Look for next gate until 5 were found
                    if len(self.gates) < 5:
                        self.state = ScanKillian1.State.STABILIZE
                        return self.update(measurement, frame, flags, dt)
                    
                    # All gates have been crossed, return home
                    else:
                        self.waypoints.append(Planner.HOME_SETPOINT)
                        self.state = ScanKillian1.State.END
                        return self.update(measurement, frame, flags, dt)

                # Waypoint not reached yet, keep going
                else:
                    return setpoint

            case ScanKillian1.State.STABILIZE:
                self.stabilization_timeout += dt
                if self.stabilization_timeout > ScanKillian1.STABILIZATION_TIMEOUT:
                    self.stabilization_timeout = 0
                    self.state = ScanKillian1.State.FIND_GATE
                return setpoint
            
            case ScanKillian1.State.FIND_GATE:

                # Gate detected
                if len(gates) > 0:

                    # Keep closest gate
                    gate = gates[0]

                    # Compute next waypoint
                    target_yaw = np.deg2rad(ScanKillian1.SCAN_YAWS[len(self.waypoints)])
                    target_position = gate.position + gate.normal * ScanKillian1.GATE_PASS_DIST
                    next_setpoint = Setpoint(target_position, target_yaw)

                    # Append it to lists
                    self.gates.append(next_setpoint)
                    self.waypoints.append(next_setpoint)

                    # Reach gate center
                    self.state = ScanKillian1.State.REACH_WAYPOINT
                    return self.update(measurement, frame, flags, dt)
                    
                # Gate not found, stay static
                else:
                    return setpoint
                
            case ScanKillian1.State.END:
                return setpoint

    def find_gates(self, frame: MatLike, measurement: Measurement, flags: Telemetry.Flags) -> list[Gate]:

        # If in simulation mode, project gates stored in file onto screen
        if Telemetry.Flags.SIMULATION in flags:
            gates_points = []
            v = view(measurement.position, euler_to_quaternion(measurement.rotation))
            for gate in self.sim_gates:

                # Skip gates not facing camera
                if np.abs(wrap(measurement.rotation.z - gate.yaw)) > np.pi / 2:
                    continue

                # Compute corners in world space assuming fixed gates size
                normal = glm.vec3(np.cos(gate.yaw), np.sin(gate.yaw), 0.0)
                right = np.cross(UP, normal)
                size = Gate.HEIGHT / 2
                world = [
                    gate.position - size * UP - size * right,
                    gate.position - size * UP + size * right,
                    gate.position + size * UP + size * right,
                    gate.position + size * UP - size * right
                ]

                # Transform to clip space and cull gates outside screen
                clip = world2clip(v, world)
                if sum(any(plane(c) < 0 for plane in CLIP_PLANES) for c in clip) >= 3:
                    continue
                
                # Transform gates to screen space and clamp them to border if necessary
                screen = [clip2screen(x) for x in clip]
                screen = [glm.clamp(s, glm.vec2(0.0, 0.0), glm.vec2(WIDTH, HEIGHT)) for s in screen]
                gates_points.append(screen)

            gates_points = np.array(gates_points)

        # Run inference on frame
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            predictions = self.model.predict(frame, conf=0.5, iou=0.7, verbose=False)

            # Expect only one prediction
            if len(predictions) != 1:
                return []

            # Expect result to contain a keypoints attribute
            prediction = predictions[0]
            if not hasattr(prediction, "keypoints"):
                return []
            
            # Expect at least one keypoint
            keypoints = prediction.keypoints.cpu()
            gates_points = keypoints.xy.numpy()

        # Skip if no gates
        if gates_points.size == 0:
            self.gates_detected_event([])
            return []

        # Build gates from keypoints and return them from closest to furthest
        gates = [Gate(corners, measurement) for corners in gates_points]
        gates.sort(key = lambda gate: gate.distance)        
        return gates
