import os
import numpy as np
import cv2
import atexit
from datetime import datetime
from cv2.typing import MatLike
from app.telemetry.measurement import Measurement

class Logger:

    CAPTURES_PATH = os.path.join("captures")
    MEASUREMENTS_FILE = "measures.npy"

    def __init__(self) -> None:

        # Initialize caches
        self.measurements: list[Measurement] = []
        self.i: int = 0

        # Create logging folders
        if not os.path.exists(Logger.CAPTURES_PATH):
            os.mkdir(Logger.CAPTURES_PATH)

        # Create session subfolder
        session_code = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.session_path = os.path.join(Logger.CAPTURES_PATH, session_code)
        os.mkdir(self.session_path)

        # Register save on exit handler
        atexit.register(self.save_measurements)

    def save_measurements(self) -> None:
        if len(self.measurements) > 0:
            array = np.array(self.measurements)
            path = os.path.join(self.session_path, Logger.MEASUREMENTS_FILE)
            np.save(path, array)
        else:
            os.rmdir(self.session_path)

    def log(self, measurement: Measurement, frame: MatLike):
        self.measurements.append(measurement.to_array())
        path = os.path.join(self.session_path, f"{self.i}.png")
        cv2.imwrite(path, frame)
        self.i += 1
