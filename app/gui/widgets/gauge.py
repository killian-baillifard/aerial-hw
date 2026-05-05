from pygame import Rect, Surface, draw
from pygame.colordict import THECOLORS as COLORS
import numpy as np
from pyglm import glm
from enum import Enum
from overrides import override
from app.gui.widgets import Widget

class Gauge(Widget):

    BORDER_COLOR = (6, 206, 0, 255)
    FILL_COLOR = (6, 206, 0, 255)
    BG_COLOR = COLORS['gray30']

    class Direction(Enum):
        NORTH = 0
        SOUTH = 1
        EAST = 2
        WEST = 3

    def __init__(self, pos: glm.uvec2, direction: Direction, end: float = 0, begin: float = 0, width: int = 20,
                 length: int = 300, corner_radius: int = 5, border_width: int = 2, z_index: int = 0):
        super().__init__(z_index)
        self.direction: Gauge.Direction = direction
        self.corner_radius: int = corner_radius
        self.border_width: int = border_width
        self.set_progress(end, begin)
        match direction:
            case Gauge.Direction.NORTH | Gauge.Direction.SOUTH:
                self.rect = Rect(pos.x, pos.y, width, length)
            case Gauge.Direction.EAST | Gauge.Direction.WEST:
                self.rect = Rect(pos.x, pos.y, length, width)

    def __del__(self) -> None:
        super().__del__()

    @override
    def draw(self, surface: Surface) -> None:

        # Draw background
        draw.rect(surface, Gauge.BG_COLOR, self.rect, 0, self.corner_radius)
    
        # Draw progress
        begin = min(self.begin, self.end)
        end = max(self.begin, self.end)
        if(begin != end):
            match self.direction:
                case Gauge.Direction.NORTH:
                    y_begin = self.rect.y + self.rect.h * (1.0 - begin)
                    y_end = self.rect.y + self.rect.h * (1.0 - end)
                    rect = Rect(self.rect.x, y_end, self.rect.w, y_begin - y_end)
                    draw.rect(surface, Gauge.FILL_COLOR, rect, 0, self.corner_radius)
                case Gauge.Direction.SOUTH:
                    y_begin = self.rect.y + self.rect.h * begin
                    y_end = self.rect.y + self.rect.h * end
                    rect = Rect(self.rect.x, y_begin, self.rect.w, y_end - y_begin)
                    draw.rect(surface, Gauge.FILL_COLOR, rect, 0, self.corner_radius)
                case Gauge.Direction.EAST:
                    x_begin = self.rect.x + self.rect.w * begin
                    x_end = self.rect.x + self.rect.w * end
                    rect = Rect(x_begin, self.rect.y, x_end - x_begin, self.rect.h)
                    draw.rect(surface, Gauge.FILL_COLOR, rect, 0, self.corner_radius)
                case Gauge.Direction.WEST:
                    x_begin = self.rect.x + self.rect.w * (1.0 - begin)
                    x_end = self.rect.x + self.rect.w * (1.0 - end)
                    rect = Rect(x_end, self.rect.y, x_begin - x_end, self.rect.h)
                    draw.rect(surface, Gauge.FILL_COLOR, rect, 0, self.corner_radius)

        # Draw border
        draw.rect(surface, Gauge.BORDER_COLOR, self.rect, self.border_width, self.corner_radius)
    
    def get_progress(self) -> tuple[float, float]:
        return self.end, self.begin

    def set_progress(self, end: float, begin: float = None):
        self.end: float = float(np.clip(end, 0.0, 1.0))
        if begin != None:
            self.begin: float = float(np.clip(begin, 0.0, 1.0))
