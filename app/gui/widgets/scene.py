import numpy as np
import cv2
from cv2.typing import MatLike
from pyglm import glm
from pygame import Surface, draw
from overrides import override
from app.gui.widgets import Widget
from app.telemetry.camera import Line, Mesh, view, euler_to_quaternion, WIDTH as CAM_W, HEIGHT as CAM_H

class Scene(Widget):

    DEPTH: float    = 4.05
    WIDTH: float    = 2.87
    HEIGHT: float   = 2.0
    COLOR = (6, 206, 0, 255)

    def __init__(self, position: glm.uvec2, size: glm.uvec2, z_index: int = 0) -> None:
        super().__init__(z_index)

        # Compute scale factor that maximizes fill while preserving aspect ratio
        self.scale = min(size.x / CAM_W, size.y / CAM_H)
        self.offset = position + glm.uvec2(int((size.x - self.scale * CAM_W) / 2), 0.0)

        # Generate room mesh
        self.room = Mesh(
            [
                glm.vec3(-Scene.DEPTH / 2, -Scene.WIDTH / 2, 0.0),
                glm.vec3(-Scene.DEPTH / 2, -Scene.WIDTH / 2, Scene.HEIGHT),
                glm.vec3(-Scene.DEPTH / 2, +Scene.WIDTH / 2, 0.0),
                glm.vec3(-Scene.DEPTH / 2, +Scene.WIDTH / 2, Scene.HEIGHT),
                glm.vec3(+Scene.DEPTH / 2, -Scene.WIDTH / 2, 0.0),
                glm.vec3(+Scene.DEPTH / 2, -Scene.WIDTH / 2, Scene.HEIGHT),
                glm.vec3(+Scene.DEPTH / 2, +Scene.WIDTH / 2, 0.0),
                glm.vec3(+Scene.DEPTH / 2, +Scene.WIDTH / 2, Scene.HEIGHT),
            ],
            [
                (0, 1), (1, 3), (3, 2), (2, 0),
                (4, 5), (5, 7), (7, 6), (6, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]
        )

        # Generate sectors mesh
        sectors_vertices = [glm.vec3(0.0, 0.0, 0.0)]
        sectors_indices = []
        for i in range(10):
            angle = np.deg2rad(45) + i * np.deg2rad(30)
            x = -(Scene.WIDTH / 2) * np.cos(angle)
            y = -(Scene.WIDTH / 2) * np.sin(angle)
            sectors_vertices.append(glm.vec3(x, y, 0.0))
            sectors_indices.append((0, i + 1))
        self.sectors = Mesh(sectors_vertices, sectors_indices)

        # Create empty lines to draw
        self.lines_to_draw: list[Line] = []

    def set_view(self, position: glm.vec3, rotation: glm.vec3):
        view_matrix = view(position, euler_to_quaternion(rotation))
        self.lines_to_draw.clear()
        self.lines_to_draw = self.room.project(view_matrix) + self.sectors.project(view_matrix)

    @override
    def draw(self, surface: Surface) -> None:
        for line in self.lines_to_draw:
            scaled_begin = self.scale * line.begin
            scaled_end = self.scale * line.end
            offset_begin = self.offset + glm.uvec2(int(scaled_begin.x), int(scaled_begin.y))
            offset_end = self.offset + glm.uvec2(int(scaled_end.x), int(scaled_end.y))
            draw.line(surface, Scene.COLOR, offset_begin, offset_end, 3)

    def overlay(self, frame: MatLike) -> None:
        for line in self.lines_to_draw:
            begin = glm.uvec2(line.begin.x, line.begin.y)
            end = glm.uvec2(line.end.x, line.end.y)
            cv2.line(frame, begin, end, Scene.COLOR[:2], 1)
