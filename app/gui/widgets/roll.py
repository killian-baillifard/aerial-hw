import numpy as np
from pygame import Surface, draw
from overrides import override
from pyglm import glm
from app.gui.widgets import Widget

class Roll(Widget):

    R1 = 100
    R2 = 300
    COLOR = (6, 206, 0, 255)

    def __init__(self, center: glm.uvec2, roll: float, z_index: int = 0):
        super().__init__(z_index)
        self.center = center
        self.roll = roll

    def __del__(self) -> None:
        super().__del__()

    @override
    def draw(self, surface: Surface) -> None:

        # Left bar
        l_begin = glm.ivec2(self.center) - glm.ivec2(Roll.R1 * glm.vec2(np.cos(self.roll), np.sin(self.roll)))
        l_end = glm.ivec2(self.center) - glm.ivec2(Roll.R2 * glm.vec2(np.cos(self.roll), np.sin(self.roll)))
        draw.line(surface, Roll.COLOR, l_begin, l_end, 3)

        # Right bar
        r_begin = glm.ivec2(self.center) + glm.ivec2(Roll.R1 * glm.vec2(np.cos(self.roll), np.sin(self.roll)))
        r_end = glm.ivec2(self.center) + glm.ivec2(Roll.R2 * glm.vec2(np.cos(self.roll), np.sin(self.roll)))
        draw.line(surface, Roll.COLOR, r_begin, r_end, 3)

    def set_roll(self, roll: float):
        self.roll = roll
