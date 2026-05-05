from pyglm import glm

class Setpoint:

    def __init__(self, position: glm.vec3, yaw: float) -> None:
        self.position = position
        self.yaw = yaw

    def __str__(self):
        return str(self.position) + " " + str(self.yaw)
