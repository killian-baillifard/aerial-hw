import numpy as np
from pygame import Surface, draw
from overrides import override
from pyglm import glm
from app.gui.widgets import Widget
from app.telemetry.camera import FOV_Y

class Pitch(Widget):

    D1 = 400
    D2 = 550
    COLOR = (6, 206, 0, 255)

    def __init__(self, center: glm.uvec2, pitch: float, amplitude: int, z_index: int = 0):
        super().__init__(z_index)
        self.center = center
        self.pitch = pitch
        self.amplitude = amplitude

    def __del__(self) -> None:
        super().__del__()

    @override
    def draw(self, surface: Surface) -> None:

        # Compute vertical position from pitch
        pitch = np.clip(self.pitch, -FOV_Y / 2, FOV_Y / 2)
        dy = self.amplitude * np.tan(pitch) / (2 * np.tan(FOV_Y / 2))

        # Left bar
        l_begin = glm.ivec2(self.center) - glm.ivec2(Pitch.D1, -dy)
        l_end = glm.ivec2(self.center) - glm.ivec2(Pitch.D2, -dy)
        draw.line(surface, Pitch.COLOR, l_begin, l_end, 3)

        # Right bar
        r_begin = glm.ivec2(self.center) + glm.ivec2(Pitch.D1, dy)
        r_end = glm.ivec2(self.center) + glm.ivec2(Pitch.D2, dy)
        draw.line(surface, Pitch.COLOR, r_begin, r_end, 3)

    def set_pitch(self, pitch: float):
        self.pitch = pitch
