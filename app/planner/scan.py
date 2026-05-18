import os, cv2
import numpy as np
from pyglm import glm
from cv2.typing import MatLike
from overrides import overrides
from app import wrap
from app.io import Measurement
from app.io import Setpoint
from app.planner import Planner
from app.telemetry import Telemetry
from ultralytics import YOLO

MODEL_PATH = os.path.join("controller_detection", "detection_model", "models", "yolov8n_v2bw_r1", "weights", "best.pt")

class ScanPlanner(Planner):

    def __init__(self):
        super().__init__()
        self.model = YOLO(MODEL_PATH)
        self.model.eval()

    @overrides
    def reload(self) -> None:

        # Fill waypoints with all scan positions
        self.waypoints.clear()
        self.waypoints.append(ScanPlanner.HOME_SETPOINT)

    def run_inference(self, frame: MatLike) -> None:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        predictions = self.model.predict(frame, conf=0.5, iou=0.7, verbose=False)
        print(predictions)

    @overrides
    def update(self, measurement: Measurement, frame: MatLike, flags: Telemetry.Flags, dt: float) -> Setpoint:

        # Waypoint list empty, go back to home position
        # if(len(self.waypoints) == 0):
        #     return Planner.HOME_SETPOINT
        
        # Until waypoint is reached, return interpolated setpoint
        # setpoint, reached = Planner.reach(self.waypoints[0], measurement)

        # When new camera measurement is avalaible, run inference
        if Telemetry.Flags.NEW_FRAME in flags:
            self.run_inference(frame)

        # TODO find and go though gate
        # Call self.gate_found_event(gate) to draw it on HUD
        
        # When reached, call this function recursively to get next setpoint
        # self.waypoints.pop(0)
        # return self.update(measurement, frame, flags, dt)
        return Planner.HOME_SETPOINT 
