import os
import csv
import cv2
from pyglm import glm
from datetime import datetime
from cv2.typing import MatLike
from app.io import Setpoint, Measurement
from app.telemetry.camera import WIDTH, HEIGHT
import atexit

class Recorder:

    RECORDINGS_PATH = os.path.join("recordings")
    GATES_PATH = os.path.join("gates")
    GATES_FILE_NAME = "gates.csv"
    GATES_FILE_PATH = os.path.join(GATES_PATH, GATES_FILE_NAME)

    def __init__(self) -> None:

        # Create recordings folders
        if not os.path.exists(Recorder.RECORDINGS_PATH):
            os.mkdir(Recorder.RECORDINGS_PATH)

        # Create gates folders and file with header row
        if not os.path.exists(Recorder.GATES_PATH):
            os.mkdir(Recorder.GATES_PATH)
        if not os.path.isfile(Recorder.GATES_FILE_PATH):
            with open(Recorder.GATES_FILE_PATH, "w", newline="") as gates_file:
                csv_writer = csv.writer(gates_file)
                csv_writer.writerow(["x", "y", "z", "yaw"])
                gates_file.flush()

        # Set crash saver
        atexit.register(self.save_on_exit)

        # Set initial state and get video codex
        self.fourcc = cv2.VideoWriter.fourcc(*'FFV1')
        self.recording = False
        self.csv_file = None
        self.csv_writer = None
        self.mp4_writer = None

    def start_recording(self) -> None:

        # Generate session code
        session = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.recording = True

        # Open csv file for stream writes
        csv_path = os.path.join(Recorder.RECORDINGS_PATH, f"{session}.csv")
        self.csv_file = open(csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp", "x", "y", "z", "roll", "pitch", "yaw", "battery"])
        self.csv_file.flush()

        # Open mp4 file for stream writes
        avi_path = os.path.join(Recorder.RECORDINGS_PATH, f"{session}.avi")
        self.avi_writer = cv2.VideoWriter(avi_path, self.fourcc, fps=30, frameSize=(WIDTH, HEIGHT), isColor=True)

    def record(self, measurement: Measurement, frame: MatLike):
        if self.recording:
            self.csv_writer.writerow(measurement.to_array())
            self.csv_file.flush()
            self.avi_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def stop_recording(self) -> None:
        self.recording = False
        self.csv_file.close()
        self.avi_writer.release()
        self.csv_file = None
        self.csv_writer = None
        self.avi_writer = None

    def save_on_exit(self) -> None:
        if self.recording:
            self.stop_recording()

    def add_gate(self, measurement: Measurement) -> None:
        with open(Recorder.GATES_FILE_PATH, "a", newline="") as gates_file:
            csv_writer = csv.writer(gates_file)
            csv_writer.writerow(measurement.as_setpoint().to_array())

    def load_recording(self, session_name: str) -> list[tuple[Measurement, MatLike]]:
        csv_path = os.path.join(Recorder.RECORDINGS_PATH, f"{session_name}.csv")
        avi_path = os.path.join(Recorder.RECORDINGS_PATH, f"{session_name}.avi")

        results = []

        # Read measurements from CSV
        measurements = []
        with open(csv_path, "r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                measurement = Measurement(
                    timestamp=float(row["timestamp"]),
                    position=glm.vec3(float(row["x"]), float(row["y"]), float(row["z"])),
                    rotation=glm.vec3(float(row["roll"]), float(row["pitch"]), float(row["yaw"])),
                    battery=float(row["battery"])
                )
                measurements.append(measurement)

        # Read frames from AVI
        cap = cv2.VideoCapture(avi_path)
        try:
            for measurement in measurements:
                ret, frame = cap.read()
                if not ret:
                    break
                results.append((measurement, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        finally:
            cap.release()

        return results
