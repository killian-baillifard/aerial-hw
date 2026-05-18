import os, cv2
from enum import Enum
import numpy as np
from cv2.typing import MatLike
from overrides import overrides
from app import wrap
from app.io import Measurement
from app.io import Setpoint
from app.planner import Planner
from app.telemetry import Telemetry
from app.telemetry.gate import Gate
from ultralytics import YOLO

MODEL_PATH = os.path.join("controller_detection", "detection_model", "models", "yolov8n_v2bw_r1", "weights", "best.pt")

class Scan(Planner):

    INITIAL_SETPOINT = Setpoint(Planner.HOME_POSITION, np.deg2rad(-45))

    class State(Enum):
        REACH_WAYPOINT = 0
        FIND_GATE = 1

    def __init__(self):
        super().__init__()
        self.model = YOLO(MODEL_PATH)
        self.model.eval()
        self.waypoints.append(Scan.INITIAL_SETPOINT)
        self.state = Scan.State.REACH_WAYPOINT

    @overrides
    def reload(self) -> None:
        self.waypoints.clear()
        self.gates.clear()
        self.waypoints.append(Scan.INITIAL_SETPOINT)
        self.state = Scan.State.REACH_WAYPOINT

    @overrides
    def update(self, measurement: Measurement, frame: MatLike, flags: Telemetry.Flags, dt: float) -> Setpoint:

        # Automatic interpolation to last waypoint in list
        setpoint, reached = Planner.reach(self.waypoints[-1], measurement, 0.5)

        # Run inference on each incoming image (for visualization purposes)
        if Telemetry.Flags.NEW_FRAME in flags:
            gates = self.find_gates(frame, measurement)
            self.gates_detected_event(gates)
        else:
            gates = []

        match self.state:

            case Scan.State.REACH_WAYPOINT:

                # Waypoint reached
                if reached:

                    # Look for next gate until 5 were found
                    if len(self.gates) < 5:
                        self.state = Scan.State.FIND_GATE
                        return self.update(measurement, frame, flags, dt)
                    
                    # All gates have been crossed, return home
                    else:
                        return Planner.HOME_SETPOINT

                # Waypoint not reached yet, keep going
                else:
                    return setpoint

            case Scan.State.FIND_GATE:

                # Gate detected
                if len(gates) > 0:

                    # Keep closest gate
                    gate = gates[0]

                    # Compute next waypoint
                    next_yaw = wrap(self.waypoints[-1].yaw + np.deg2rad(45))
                    next_setpoint = Setpoint(gate.position, next_yaw)

                    # Append it to lists
                    self.gates.append(next_setpoint)
                    self.waypoints.append(next_setpoint)

                    # Fire GUI events
                    self.gates_detected_event([gate])
                    self.gate_found_event(next_setpoint)

                    # Reach gate center
                    self.state = Scan.State.REACH_WAYPOINT
                    return self.update(measurement, frame, flags, dt)
                    
                # Gate not found, stay static
                else:
                    return setpoint

    def find_gates(self, frame: MatLike, measurement: Measurement) -> list[Gate]:

        # Run inference on frame
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        predictions = self.model.predict(frame, conf=0.5, iou=0.7, verbose=False)

        # Expect only one prediction
        if len(predictions) < 1 or 1 < len(predictions):
            return []

        # Expect result to contain a keypoints attribute
        prediction = predictions[0]
        if not hasattr(prediction, "keypoints"):
            return []
        
        # Expect at least one keypoint
        keypoints = prediction.keypoints.cpu()
        gates_points = keypoints.xy.numpy()
        if gates_points.size == 0:
            self.gates_detected_event([])
            return []

        # Build gates from keypoints and return them from closest to furthest
        gates = [Gate(corners, measurement) for corners in gates_points]
        gates.sort(key = lambda gate: gate.distance)        
        return gates
