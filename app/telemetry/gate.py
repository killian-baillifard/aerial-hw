import numpy as np
from pyglm import glm
from app.io import Measurement
from app.telemetry.camera import UP
from app.telemetry.ray import Ray

class Gate:

    HEIGHT = 0.40 # real gate height in meters m

    def __init__(self, corners: np.ndarray, measurement: Measurement) -> None:

        # Parse corners
        assert corners.shape == (4, 2)
        self.corners = [glm.vec2(corner) for corner in corners]

        # Split left / right then top / bottom corners
        sorted_x = sorted(self.corners, key=lambda p: p.x)
        left  = sorted_x[:2]
        right = sorted_x[2:]
        tl, bl = sorted(left,  key=lambda p: p.y)
        tr, br = sorted(right, key=lambda p: p.y)

        # Cast rays through the four corners
        ray_tl = Ray(tl, measurement.position, measurement.rotation)
        ray_bl = Ray(bl, measurement.position, measurement.rotation)
        ray_tr = Ray(tr, measurement.position, measurement.rotation)
        ray_br = Ray(br, measurement.position, measurement.rotation)

        def vertical_angle(ray_direction: glm.vec3) -> float:
            """Elevation angle of a ray (angle above/below horizontal plane)."""
            horizontal = glm.length(glm.vec2(ray_direction.x, ray_direction.y))
            return np.arctan2(ray_direction.z, horizontal)

        elev_tl = vertical_angle(ray_tl.direction)
        elev_bl = vertical_angle(ray_bl.direction)
        elev_tr = vertical_angle(ray_tr.direction)
        elev_br = vertical_angle(ray_br.direction)

        # Distance to left/right edges using elevation difference
        dl = Gate.HEIGHT / (np.tan(elev_tl) - np.tan(elev_bl))
        dr = Gate.HEIGHT / (np.tan(elev_tr) - np.tan(elev_br))

        # Gate center elevation from camera
        z_center_l = measurement.position.z + dl * np.tan((elev_tl + elev_bl) / 2)
        z_center_r = measurement.position.z + dr * np.tan((elev_tr + elev_br) / 2)

        # Horizontal midpoint ray directions
        dir_ml_h = glm.normalize(glm.vec2(ray_tl.direction.x + ray_bl.direction.x,
                                        ray_tl.direction.y + ray_bl.direction.y))
        dir_mr_h = glm.normalize(glm.vec2(ray_tr.direction.x + ray_br.direction.x,
                                        ray_tr.direction.y + ray_br.direction.y))

        pl = glm.vec3(measurement.position.x + dir_ml_h.x * dl,
                    measurement.position.y + dir_ml_h.y * dl,
                    z_center_l)
        pr = glm.vec3(measurement.position.x + dir_mr_h.x * dr,
                    measurement.position.y + dir_mr_h.y * dr,
                    z_center_r)

        # Gate center position
        self.center = (pl + pr) / 2
        self.distance = glm.distance(measurement.position, self.center)

        # Gate yaw from axis in world space
        gate_axis = glm.normalize(pr - pl)
        self.normal = glm.normalize(glm.cross(UP, gate_axis))
        self.yaw  = np.arctan2(self.normal.y, self.normal.x)

    def __str__(self) -> str:
        return "Gate(" + str(self.center) + " " + str(self.yaw) + ")"
