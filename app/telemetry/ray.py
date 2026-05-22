import numpy as np
from pyglm import glm
from app.telemetry.camera import *

class Ray:

    def __init__(self, point: glm.vec2, position: glm.vec3, rotation: glm.vec3) -> None:
        """
        Initialize a ray object from a raw camera measurement.

        Parameters
        ----------
        point : glm.vec2
            Ray origin in screen space.
        position : glm.vec3
            Camera position in world space.
        rotation : glm.vec3
            Camera orientation in world space.
        """

        # Compute screen point in NDC
        ndc = glm.vec4(
            (2.0 * point.x / WIDTH) - 1.0,
            1.0 - (2.0 * point.y / HEIGHT), # Flip screen y coordinate
            -1.0, # Near plane in NDC
            1.0
        )

        # Back-project to world space
        world = glm.inverse(PROJECTION * view(position, euler_to_quaternion(rotation))) * ndc     
        world /= world.w # Divide perspective
        self.position = position
        self.direction = glm.normalize(glm.vec3(world) - position)
