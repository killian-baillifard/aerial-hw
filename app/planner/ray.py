import numpy as np
from pyglm import glm
from app.telemetry import Telemetry

CAMERA_FOV_Y: float             = 1.5                   # radians
CAMERA_RESOLUTION: float        = 300                   # px
CAMERA_NEAR_PLANE: float        = 0.01                  # m
CAMERA_ASPECT_RATIO             = Telemetry.CAMERA_WIDTH / Telemetry.CAMERA_HEIGHT
CAMERA_PROJECTION: glm.mat4x4   = glm.infinitePerspective(CAMERA_FOV_Y, CAMERA_ASPECT_RATIO, CAMERA_NEAR_PLANE)

class Ray:

    FORWARD = glm.vec3(1, 0, 0)
    UP = glm.vec3(0, 0, 1)

    def __init__(self, point: glm.vec2, position: glm.vec3, quaternion: glm.quat) -> None:
        """
        Initialize a ray object from a raw camera measurement.

        Parameters
        ----------
        point : glm.vec2
            Ray origin in screen space.
        position : glm.vec3
            Camera position in world space.
        quaternion : glm.quat
            Camera orientation in world space.
        """

        self.point = point
        self.position = position
        self.quaternion = quaternion

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

        # Compute view matrix
        forward = self.quaternion * Ray.FORWARD
        up = self.quaternion * Ray.UP
        view = glm.lookAt(
            self.position,
            self.position + forward,
            up
        )

        # Compute screen point in NDC
        ndc = glm.vec4(
            (2.0 * self.point.x / CAMERA_RESOLUTION) - 1.0,
            1.0 - (2.0 * self.point.y / CAMERA_RESOLUTION),     # Flip screen y coordinate
            -1.0,                                               # Near plane in NDC
            1.0
        )

        # Back-project to world space
        world = glm.inverse(CAMERA_PROJECTION * view) * ndc     
        world /= world.w                                        # Divide perspective
        ray = glm.normalize(glm.vec3(world) - self.position)    # Compute normalized ray
        return self.position, ray
