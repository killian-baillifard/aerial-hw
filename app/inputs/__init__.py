from pyglm import glm

class Input():

    def __init__(self, position: glm.vec3 = glm.vec3(0.0, 0.0, 0.0), yaw: float = 0.0) -> None:
        self.position: glm.vec3 = position
        self.yaw: float = yaw

    def update(self, dt: float) -> None:
        raise NotImplementedError
