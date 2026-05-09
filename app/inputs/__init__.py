import numpy as np
from pyglm import glm
from app import wrap
from app.telemetry.measurement import Measurement

class Command:

    def __init__(self, velocity: glm.vec2 = glm.vec2(0.0, 0.0), yaw_rate: float = 0.0, altitude: float = 0.0) -> None:
        self.velocity = velocity
        self.yaw_rate = yaw_rate
        self.altitude = altitude

    def update_altitude(self, climb_rate: float, dt: float) -> None:
        self.altitude += climb_rate * dt
    
    def update(self, dt: float) -> None:
        raise NotImplementedError()

class Setpoint:

    MAX_EQUIV_CMD_VELOCITY: glm.vec2 = 1.0
    MAX_EQUIV_CMD_YAW_RATE: float = np.pi / 2

    def __init__(self, position: glm.vec3 = glm.vec3(0.0, 0.0, 0.0), yaw: float = 0.0) -> None:
        self.position: glm.vec3 = position
        self.yaw: float = yaw

    def __str__(self):
        return str(self.position) + " " + str(self.yaw)

    def equivalent_command(self, measurement: Measurement) -> Command:
        absolute = self.position - measurement.position
        relative = glm.rotateZ(absolute, -measurement.rotation.z).xy
        velocity = relative if glm.length(relative) < Setpoint.MAX_EQUIV_CMD_VELOCITY else Setpoint.MAX_EQUIV_CMD_VELOCITY * glm.normalize(relative)
        yaw_error = wrap(measurement.rotation.z - self.yaw)
        yaw_rate = yaw_error if yaw_error < Setpoint.MAX_EQUIV_CMD_YAW_RATE else Setpoint.MAX_EQUIV_CMD_YAW_RATE
        return Command(velocity, yaw_rate, self.position.z)
