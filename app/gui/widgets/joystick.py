from pygame import Surface, draw
from overrides import override
from pyglm import glm
from app.gui.widgets import Widget

class Joystick(Widget):

    COLOR = (6, 206, 0, 255)

    def __init__(self, origin: glm.uvec2, delta: glm.ivec2, z_index: int = 0):
        super().__init__(z_index)
        self.origin = origin
        self.delta = delta

    def __del__(self) -> None:
        super().__del__()

    @override
    def draw(self, surface: Surface) -> None:
        draw.circle(surface, Joystick.COLOR, self.origin, 5)
        draw.line(surface, Joystick.COLOR, self.origin, glm.ivec2(self.origin) + self.delta, 3)
        draw.circle(surface, Joystick.COLOR, glm.ivec2(self.origin) + self.delta, 9, 3)
        draw.circle(surface, Joystick.COLOR, glm.ivec2(self.origin) + self.delta, 3)

    def set_delta(self, delta: glm.ivec2):
        self.delta = delta
