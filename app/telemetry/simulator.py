import numpy as np
from pyglm import glm
import cv2
from cv2.typing import MatLike
from app import wrap
from app.telemetry import Telemetry
from app.telemetry.measurement import Measurement
from app.inputs import Input
from app.inputs.setpoint import Setpoint

class Simulator:

    LINSPEED = 1.0
    YAWRATE = np.pi
    BATT_DECAY = 0.01
    M_PER_PX = 0.01

    def __init__(self) -> None:
        self.measurement = Measurement(battery=1.0)

    def get_last_measurement(self) -> Measurement:
        return self.measurement
    
    def get_last_frame(self) -> MatLike:
        frame = np.zeros(shape=(Telemetry.CAMERA_HEIGHT, Telemetry.CAMERA_WIDTH, 3), dtype=np.uint8)

        # World → pixel: +X is UP (decreasing row), +Y is LEFT (decreasing col)
        cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
        px = int(cx - self.measurement.position.y / Simulator.M_PER_PX)  # +Y → left
        py = int(cy - self.measurement.position.x / Simulator.M_PER_PX)  # +X → up

        # Clamp to frame bounds
        px = np.clip(px, 0, frame.shape[1] - 1)
        py = np.clip(py, 0, frame.shape[0] - 1)

        # Draw position circle
        radius = 10
        cv2.circle(frame, (px, py), radius, (0, 255, 0), 2)

        # Draw yaw direction arrow (+X is up, +Y is left in image space)
        yaw = self.measurement.rotation.z
        arrow_len = 20
        ax = int(px - arrow_len * np.sin(yaw))  # yaw rotates X toward Y → left in image
        ay = int(py - arrow_len * np.cos(yaw))  # +X component → up in image
        cv2.arrowedLine(frame, (px, py), (ax, ay), (0, 0, 255), 2, tipLength=0.3)

        return frame
    
    def update(self, control_input: Input, setpoint: Setpoint, dt: float) -> None:

        roll = -control_input.position.y * (np.pi / 4.0)
        pitch = -control_input.position.x * (np.pi / 4.0)

        position_error = setpoint.position - self.measurement.position
        self.measurement.position += position_error * Simulator.LINSPEED * dt
        if self.measurement.position.z < 0:
            self.measurement.position.z = 0
        
        self.measurement.rotation.z += control_input.yaw * Simulator.YAWRATE * dt
        self.measurement.rotation.y = pitch
        self.measurement.rotation.x = roll

        self.measurement.rotation.x = wrap(self.measurement.rotation.x)
        self.measurement.rotation.y = wrap(self.measurement.rotation.y)
        self.measurement.rotation.z = wrap(self.measurement.rotation.z)
        
        self.measurement.timestamp += dt
        self.measurement.battery = np.clip(self.measurement.battery - Simulator.BATT_DECAY * dt, 0.0, 1.0)


    def reset(self) -> None:
        self.measurement = Measurement(battery=1.0)
