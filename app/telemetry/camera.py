import numpy as np
from pyglm import glm

FORWARD = glm.vec3(1, 0, 0)
LEFT = glm.vec3(0, 1, 0)
UP = glm.vec3(0, 0, 1)
WIDTH                    = 324                   # px
HEIGHT                   = 244                   # px
CAM_FULL_HEIGHT          = 330                   # px
FOV_Y: float             = 1.0                   # radians
NEAR_PLANE: float        = 0.001                 # m
ASPECT_RATIO             = WIDTH / HEIGHT
PROJECTION: glm.mat4x4   = glm.infinitePerspective(FOV_Y, ASPECT_RATIO, NEAR_PLANE)
CLIP_PLANES = [
    lambda v:  v.w + v.z,  # near:   z >= -w
    lambda v:  v.w + v.x,  # left:   x >= -w
    lambda v:  v.w - v.x,  # right:  x <=  w
    lambda v:  v.w + v.y,  # bottom: y >= -w
    lambda v:  v.w - v.y,  # top:    y <=  w
]

def euler_to_quaternion(euler_angles: glm.vec3) -> glm.quat:
    qx = glm.angleAxis(euler_angles.x, FORWARD)
    qy = glm.angleAxis(euler_angles.y, LEFT)
    qz = glm.angleAxis(euler_angles.z, UP)
    return qz * qy * qx

def view(position: glm.vec3, rotation: glm.quat) -> glm.mat4x4:
    forward = rotation * FORWARD
    up = rotation * UP
    return glm.lookAt(
        position,
        position + forward,
        up
    )

def world2clip(view: glm.mat4x4, vertices: list[glm.vec3]) -> list[glm.vec4]:
    view_projection = PROJECTION * view
    return [view_projection * glm.vec4(vertex, 1.0) for vertex in vertices]

def clip_line(a: glm.vec4, b: glm.vec4) -> tuple[glm.vec4, glm.vec4] | None:
    pa, pb = a, b
    for plane in CLIP_PLANES:
        da = plane(pa)
        db = plane(pb)
        if da < 0 and db < 0:
            return None
        if da < 0:
            t = da / (da - db)
            pa = pa + t * (pb - pa)
        elif db < 0:
            t = da / (da - db)
            pb = pa + t * (pb - pa)
    return pa, pb

def clip2screen(v: glm.vec4) -> glm.vec2:
    ndc = glm.vec3(v) / v.w
    sx = (ndc.x + 1.0) * 0.5 * WIDTH
    sy = (1.0 - ndc.y) * 0.5 * HEIGHT
    return glm.vec2(sx, sy)

class Line:

    def __init__(self, begin: glm.vec2, end: glm.vec2) -> None:
        self.begin = begin
        self.end = end

class Mesh:

    def __init__(self, vertices: list[glm.vec3], indices: tuple[int, int]) -> None:
        self.vertices = vertices
        self.indices = indices

    def project(self, view: glm.mat4x4) -> list[Line]:
        clipped_vertices = world2clip(view, self.vertices)
        lines: list[Line] = []
        for i0, i1 in self.indices:
            clipped_line = clip_line(clipped_vertices[i0], clipped_vertices[i1])
            if clipped_line is None:
                continue
            c0, c1 = clipped_line
            lines.append(Line(clip2screen(c0), clip2screen(c1)))
        return lines
