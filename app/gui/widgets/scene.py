import numpy as np
import cv2
from cv2.typing import MatLike
from pyglm import glm
from pygame import Surface, draw
from overrides import override
from app.gui.widgets import Widget
from app.io import Setpoint
from app.telemetry.camera import Line, Mesh, view, euler_to_quaternion, WIDTH as CAM_W, HEIGHT as CAM_H
from app.telemetry.gate import Gate

class Scene(Widget):

    DEPTH: float        = 4.05  # m
    WIDTH: float        = 2.87  # m
    HEIGHT: float       = 2.4   # m
    GATE_WIDTH: float   = 0.4   # m
    GATE_HEIGHT: float  = 0.4   # m
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

        # Create empty gates
        self.gates_mesh: list[Mesh] = []
        self.gates_detection_overlay: list[Gate] = []

        # Create empty lines to draw
        self.lines_to_draw: list[Line] = []

    def add_gate(self, gate: Setpoint):

        # Create vertices
        local_vertices = [
            glm.vec3(0.0, -Scene.GATE_WIDTH / 2, -Scene.GATE_HEIGHT / 2), # 0: Bottom Left
            glm.vec3(0.0, -Scene.GATE_WIDTH / 2, +Scene.GATE_HEIGHT / 2), # 1: Top Left
            glm.vec3(0.0, +Scene.GATE_WIDTH / 2, +Scene.GATE_HEIGHT / 2), # 2: Top Right
            glm.vec3(0.0, +Scene.GATE_WIDTH / 2, -Scene.GATE_HEIGHT / 2)  # 3: Bottom Right
        ]

        # Rotate vertices
        rotation = glm.rotate(glm.mat4(1.0), gate.yaw, glm.vec3(0, 0, 1))
        world_vertices = []
        for v in local_vertices:
            transformed = glm.vec3(rotation * glm.vec4(v, 1.0)) + gate.position
            world_vertices.append(transformed)

        # Add poles base
        ground_left = glm.vec3(world_vertices[0].x, world_vertices[0].y, 0.0)
        ground_right = glm.vec3(world_vertices[3].x, world_vertices[3].y, 0.0)
        world_vertices.append(ground_left)
        world_vertices.append(ground_right)

        # Defines indices (gate + poles)
        indices = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (0, 4),
            (3, 5)
        ]
        self.gates_mesh.append(Mesh(world_vertices, indices))

    def new_gates_points(self, gates: list[Gate]) -> None:
        self.gates_detection_overlay = gates

    def set_view(self, position: glm.vec3, rotation: glm.vec3):
        view_matrix = view(position, euler_to_quaternion(rotation))
        self.lines_to_draw.clear()
        self.lines_to_draw = self.room.project(view_matrix) + self.sectors.project(view_matrix)
        for gate_mesh in self.gates_mesh:
            self.lines_to_draw += gate_mesh.project(view_matrix)

    @override
    def draw(self, surface: Surface) -> None:

        # Compute gates detection overlay
        for gate in self.gates_detection_overlay:
            gui_coords: list[glm.uvec2] = []
            for corner in gate.corners:
                scaled_frame_coords = self.scale * corner
                frame_coords = glm.uvec2(scaled_frame_coords.x, scaled_frame_coords.y)
                gui_coords.append(self.offset + frame_coords)
                draw.circle(surface, Scene.COLOR, gui_coords[-1], 6)
            for i in range(4):
                begin = gui_coords[i]
                end = gui_coords[(i + 1) % 4]
                draw.line(surface, Scene.COLOR, begin, end, 14)

        # Draw meshes
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
            cv2.line(frame, begin, end, (255, 0, 0), 1)
