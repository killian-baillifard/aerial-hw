import numpy as np
from pygame import Surface, draw
from overrides import override
from pyglm import glm
from app.gui.widgets import Widget

class Shutter(Widget):

    COLOR = (6, 206, 0, 255)

    def __init__(self, center: glm.uvec2, offset: glm.uvec2, duration: int = 20, z_index: int = 0):
        super().__init__(z_index)
        self.center = center
        self.offset = offset
        self.duration = duration
        self.cooldown = 0

    def __del__(self) -> None:
        super().__del__()

    @override
    def draw(self, surface: Surface) -> None:

        # Show only for cooldown duration
        self.cooldown -= 1
        if self.cooldown > 0:

            # Top left chevron
            tl_center = glm.ivec2(self.center) + glm.ivec2(-self.offset.x, -self.offset.y)
            tl_vertical = tl_center + glm.ivec2(0, 50)
            tl_horizontal = tl_center + glm.ivec2(50, 0)
            draw.line(surface, Shutter.COLOR, tl_center, tl_vertical, 3)
            draw.line(surface, Shutter.COLOR, tl_center, tl_horizontal, 3)

            # Top right chevron
            tl_center = glm.ivec2(self.center) + glm.ivec2(self.offset.x, -self.offset.y)
            tl_vertical = tl_center + glm.ivec2(0, 50)
            tl_horizontal = tl_center - glm.ivec2(50, 0)
            draw.line(surface, Shutter.COLOR, tl_center, tl_vertical, 3)
            draw.line(surface, Shutter.COLOR, tl_center, tl_horizontal, 3)

            # Bottom left chevron
            tl_center = glm.ivec2(self.center) + glm.ivec2(-self.offset.x, self.offset.y)
            tl_vertical = tl_center - glm.ivec2(0, 50)
            tl_horizontal = tl_center + glm.ivec2(50, 0)
            draw.line(surface, Shutter.COLOR, tl_center, tl_vertical, 3)
            draw.line(surface, Shutter.COLOR, tl_center, tl_horizontal, 3)

            # Bottom right chevron
            br_center = glm.ivec2(self.center) + glm.ivec2(self.offset.x, self.offset.y)
            br_vertical = br_center - glm.ivec2(0, 50)
            br_horizontal = br_center - glm.ivec2(50, 0)
            draw.line(surface, Shutter.COLOR, br_center, br_vertical, 3)
            draw.line(surface, Shutter.COLOR, br_center, br_horizontal, 3)

    def trigger(self):
        self.cooldown = self.duration
