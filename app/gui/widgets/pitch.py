from pygame import Surface, draw
from overrides import override
from pyglm import glm
from app.gui.widgets import Widget

class Pitch(Widget):

    D1 = 400
    D2 = 550
    COLOR = (6, 206, 0, 255)

    def __init__(self, center: glm.uvec2, pitch: float, z_index: int = 0):
        super().__init__(z_index)
        self.center = center
        self.pitch = pitch

    def __del__(self) -> None:
        super().__del__()

    @override
    def draw(self, surface: Surface) -> None:

        # Left bar
        l_begin = glm.ivec2(self.center) - glm.ivec2(Pitch.D1, -self.pitch)
        l_end = glm.ivec2(self.center) - glm.ivec2(Pitch.D2, -self.pitch)
        draw.line(surface, Pitch.COLOR, l_begin, l_end, 3)

        # Right bar
        r_begin = glm.ivec2(self.center) + glm.ivec2(Pitch.D1, self.pitch)
        r_end = glm.ivec2(self.center) + glm.ivec2(Pitch.D2, self.pitch)
        draw.line(surface, Pitch.COLOR, r_begin, r_end, 3)

    def set_pitch(self, pitch: float):
        self.pitch = pitch
