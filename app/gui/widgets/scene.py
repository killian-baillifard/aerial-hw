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
        self.offset = position
        self.box = Mesh(
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
                # Back face (-D)
                (0, 1), (1, 3), (3, 2), (2, 0),

                # Front face (+D)
                (4, 5), (5, 7), (7, 6), (6, 4),

                # Connecting edges
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]
        )
        self.box_lines: list[Line] = []

        # Compute scale factor that maximizes fill while preserving aspect ratio
        self.scale = min(size.x / CAM_W, size.y / CAM_H)
        self.offset += glm.uvec2(int((size.x - self.scale * CAM_W) / 2), 0.0)

    def set_view(self, position: glm.vec3, rotation: glm.vec3):
        view_matrix = view(position, euler_to_quaternion(rotation))
        self.box_lines = self.box.project(view_matrix)

    @override
    def draw(self, surface: Surface) -> None:
        for line in self.box_lines:
            scaled_begin = self.scale * line.begin
            scaled_end = self.scale * line.end
            offset_begin = self.offset + glm.uvec2(int(scaled_begin.x), int(scaled_begin.y))
            offset_end = self.offset + glm.uvec2(int(scaled_end.x), int(scaled_end.y))
            draw.line(surface, Scene.COLOR, offset_begin, offset_end, 3)
