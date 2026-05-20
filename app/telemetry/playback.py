import numpy as np
from cv2.typing import MatLike
from app.generics import Mailbox
from app.io import Measurement
from app.telemetry.camera import WIDTH, HEIGHT
from app.telemetry.recorder import Recorder

class Playback:

    def __init__(self, session: str) -> None:

        # Load measurements csv and frames avi files
        self.recording: list[tuple[Measurement, MatLike]] = Recorder.load_recording(session)
        self.i = 0
        self.measurement: Mailbox[Measurement] = Mailbox(self.recording[self.i][0])
        self.frame: Mailbox[MatLike] = Mailbox(self.recording[self.i][1])
        self.t = self.recording[self.i][0].timestamp / 1000

    def reset(self) -> None:
        self.i = 0
        self.measurement.set(self.recording[self.i][0])
        self.frame.set(self.recording[self.i][1])
        self.t = self.recording[self.i][0].timestamp / 1000

    def step(self, dt: float) -> None:

        # Playback done, early return
        if self.i == len(self.recording) - 1:
            return

        # Step at each timestamp
        self.t += dt
        if self.t >= self.recording[self.i + 1][0].timestamp / 1000:
            self.i += 1
            self.measurement.set(self.recording[self.i][0])
            self.frame.set(self.recording[self.i][1])
