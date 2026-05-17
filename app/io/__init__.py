import numpy as np
from pyglm import glm
from typing import Self
from app import wrap

class Command:

    YAW_RATE = np.pi

    def __init__(self, velocity: glm.vec3 = glm.vec3(0.0, 0.0, 0.0), yaw_rate: float = 0.0) -> None:
        self.velocity = velocity
        self.yaw_rate = yaw_rate * Command.YAW_RATE
        self.capture = False
    
    def update(self, dt: float) -> None:
        raise NotImplementedError()

class Setpoint:

    def __init__(self, position: glm.vec3 = glm.vec3(0.0, 0.0, 0.0), yaw: float = 0.0) -> None:
        self.position: glm.vec3 = position
        self.yaw: float = yaw

    def __str__(self):
        return str(self.position) + " " + str(self.yaw)

    def to_command(self, measurement: "Measurement") -> Command:
        absolute_error = self.position - measurement.position
        relative_error = glm.rotateZ(absolute_error, -measurement.rotation.z)
        yaw_error = wrap(self.yaw - measurement.rotation.z)
        vx = relative_error.x if relative_error.x < 1.0 else 1.0
        vy = relative_error.y if relative_error.y < 1.0 else 1.0
        vz = relative_error.z if relative_error.z < 1.0 else 1.0
        yaw_rate = yaw_error if yaw_error < Command.YAW_RATE else Command.YAW_RATE
        return Command(glm.vec3(vx, vy, vz), yaw_rate)
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.position.x,
            self.position.y,
            self.position.z,
            self.yaw
        ])

class Measurement:

    BATT_SIM_DECAY_RATE = 0.005
    SIM_ROLL_AMPLITUDE = np.pi / 16
    SIM_PITCH_AMPLITUDE = np.pi / 16

    def __init__(self, timestamp: float = 0.0, position: glm.vec3 = glm.vec3(0.0), rotation: glm.vec3 = glm.vec3(0.0), battery: float = 1.0) -> None:
        self.timestamp: float = timestamp
        self.position: glm.vec3 = position
        self.rotation: glm.vec3 = rotation
        self.battery: float = battery

    def __str__(self):
        return str(self.timestamp) + " " + str(self.position) + " " + str(self.rotation) + " " + str(self.battery)
    
    def as_setpoint(self) -> Setpoint:
        return Setpoint(self.position, self.rotation.z)

    def to_array(self) -> np.ndarray:
        return np.array([
            self.timestamp,
            self.position.x,
            self.position.y,
            self.position.z,
            self.rotation.x,
            self.rotation.y,
            self.rotation.z,
            self.battery
        ])
    
    def simulate(self, command: Command, dt: float) -> Self:

        # Change to world frame of reference
        v = glm.rotateZ(command.velocity, self.rotation.z)
        p = self.position + v * dt
        if p.z < 0:
            p.z = 0.0

        # Integrate
        return Measurement(
            self.timestamp + dt,
            p,
            glm.vec3(
                -command.velocity.y * Measurement.SIM_ROLL_AMPLITUDE,
                command.velocity.x * Measurement.SIM_PITCH_AMPLITUDE,
                wrap(self.rotation.z + command.yaw_rate * dt)
            ),
            np.clip(self.battery - Measurement.BATT_SIM_DECAY_RATE * dt, 0.0, 1.0)
        )
