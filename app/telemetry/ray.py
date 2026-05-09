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

        self.point = point
        self.position = position
        self.rotation = euler_to_quaternion(rotation)

    def triangulate(self, other: "Ray") -> glm.vec3:
        """
        Intersect this ray with another to triangulate a point in world space.

        Parameters
        ----------
        ray : Ray
            Other ray pointing at the same position in world space from a different point of view.

        Returns
        -------
        position : glm.vec3
            Best approximation of both rays intersection.
        
        Raises
        ------
        LinAlgError
            If the SVD computation does not converge.
        """
        
        # Raycast measurments
        p, r = self.cast()
        q, s = other.cast()

        # Solve system of equations
        a = np.array([
            [r.x, -s.x],
            [r.y, -s.y],
            [r.z, -s.z],
        ])
        b = np.array(q - p)
        sol = (np.linalg.pinv(a) @ b).flatten()

        # Compute best approximation of position
        l: float = sol[0]
        m: float = sol[1]
        f = p + l * r
        g = q + m * s
        position = (f + g) / 2
        return glm.vec3(position)

    def cast(self) -> tuple[glm.vec3, glm.vec3]:
        """
        Deferred ray casting into world space.

        Returns
        -------
        origin : glm.vec3
            Ray origin in world space.
        direction : glm.vec3
            Ray unit direction in world space.
        """

        # Compute screen point in NDC
        ndc = glm.vec4(
            (2.0 * self.point.x / WIDTH) - 1.0,
            1.0 - (2.0 * self.point.y / HEIGHT),    # Flip screen y coordinate
            -1.0,                                   # Near plane in NDC
            1.0
        )

        # Back-project to world space
        world = glm.inverse(PROJECTION * view(self.position, self.rotation)) * ndc     
        world /= world.w                                        # Divide perspective
        ray = glm.normalize(glm.vec3(world) - self.position)    # Normalize ray
        return self.position, ray
