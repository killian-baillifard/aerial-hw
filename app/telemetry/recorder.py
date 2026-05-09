import os
import csv
import cv2
from datetime import datetime
from cv2.typing import MatLike
from app.io import Measurement
from app.telemetry.camera import WIDTH, HEIGHT
import atexit

class Recorder:

    RECORDINGS_PATH = os.path.join("recordings")

    def __init__(self) -> None:

        # Create recordings folders
        if not os.path.exists(Recorder.RECORDINGS_PATH):
            os.mkdir(Recorder.RECORDINGS_PATH)

        # Set crash saver
        atexit.register(self.crash_save)

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

    def crash_save(self) -> None:
        if self.recording:
            self.stop_recording()
