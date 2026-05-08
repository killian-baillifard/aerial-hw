from pygame import Rect, Surface, draw, mouse
from pygame.colordict import THECOLORS as COLORS
from app.gui.widgets import Color
from overrides import override
from enum import Enum
from pyglm import glm
from typing import Callable
from app.gui.widgets import Widget
from app.gui.widgets.label import Label

class Button(Widget):

    CORNER_RADIUS = 5
    BORDER_WIDTH = 2

    BG_DISABLED_COLOR = (0, 0, 0, 0)
    BG_IDLE_COLOR = COLORS["gray20"]
    BG_HOVERED_COLOR = COLORS["gray30"]
    BG_PRESSED_COLOR = COLORS["gray40"]

    BORDER_DISABLED_COLOR = (3, 170, 0, 255)
    BORDER_IDLE_COLOR = (6, 206, 0, 255)
    BORDER_HOVERED_COLOR = (9, 220, 2, 255)
    BORDER_PRESSED_COLOR = (15, 250, 4, 255)
    
    class State(Enum):
        DISABLED = 0
        IDLE = 1
        HOVERED = 2
        PRESSED = 3

    def __init__(self, pos: glm.uvec2, text: str = "Button", width: int = 100, height: int = 30, disabled: bool = False, z_index: int = 0) -> None:
        super().__init__(z_index)
        
        # Set properties
        self.rect = Rect(pos.x, pos.y, width, height)
        self.label = Label(pos, text, color=Button.BORDER_IDLE_COLOR, z_index=z_index + 1)
        self.state = Button.State.DISABLED if disabled else Button.State.IDLE
        self.on_press_handler: list[Callable[[], None]] = []
        self.on_release_handler: list[Callable[[], None]] = []
        self.enabled = not disabled

        # Center label on button
        label_rect = self.label.get_rect()
        centered_pos = glm.uvec2(
            pos.x + (width - label_rect.w) / 2,
            pos.y + (height - label_rect.h) / 2
        )
        self.label.set_position(centered_pos)

        # Set default colors
        self.bg_disabled_color = Button.BG_DISABLED_COLOR
        self.bg_idle_color = Button.BG_IDLE_COLOR
        self.bg_hovered_color = Button.BG_HOVERED_COLOR
        self.bg_pressed_color = Button.BG_PRESSED_COLOR
        self.border_disabled_color = Button.BORDER_DISABLED_COLOR
        self.border_idle_color = Button.BORDER_IDLE_COLOR
        self.border_hovered_color = Button.BORDER_HOVERED_COLOR
        self.border_pressed_color = Button.BORDER_PRESSED_COLOR

    def __del__(self) -> None:
        super().__del__()

    def disable(self) -> None:
        self.enabled = False

    def enable(self) -> None:
        self.enabled = True

    def set_bg_color(self, disabled: Color, idle: Color, hover: Color, pressed: Color) -> None:
        self.bg_disabled_color = disabled
        self.bg_idle_color = idle
        self.bg_hovered_color = hover
        self.bg_pressed_color = pressed

    def set_border_color(self, disabled: Color, idle: Color, hover: Color, pressed: Color) -> None:
        self.border_disabled_color = disabled
        self.border_idle_color = idle
        self.border_hovered_color = hover
        self.border_pressed_color = pressed
        self.label.set_color(idle)

    def add_press_handler(self, handler: Callable[[], None]) -> None:
        self.on_press_handler.append(handler)

    def add_release_handler(self, handler: Callable[[], None]) -> None:
        self.on_release_handler.append(handler)

    def set_text(self, text: str) -> None:
        self.label.set_text(text)
        label_rect = self.label.get_rect()
        centered_pos = glm.uvec2(
            self.rect.x + (self.rect.w - label_rect.w) / 2,
            self.rect.y + (self.rect.h - label_rect.h) / 2
        )
        self.label.set_position(centered_pos)

    @override
    def update(self) -> None:

        # Compute collision with mouse
        x, y = mouse.get_pos()
        x_collision = self.rect.x <= x and x <= self.rect.x + self.rect.w
        y_collision = self.rect.y <= y and y <= self.rect.y + self.rect.h
        hovered = x_collision and y_collision

        # Compute next button state
        left, _, _ = mouse.get_pressed()
        if not self.enabled:
            self.state = Button.State.DISABLED
        elif self.state is Button.State.DISABLED:
            self.state = Button.State.IDLE
        elif hovered and left and self.state is Button.State.HOVERED:
            self.state = Button.State.PRESSED
            for handler in self.on_press_handler:
                handler()
        elif hovered and not left:
            if self.state is Button.State.PRESSED:
                self.state = Button.State.HOVERED
                for handler in self.on_release_handler:
                    handler()
            if self.state is Button.State.IDLE:
                self.state = Button.State.HOVERED
        elif not hovered and self.state is not Button.State.IDLE:
            self.state = Button.State.IDLE

    @override
    def draw(self, surface: Surface) -> None:
        match self.state:
            case Button.State.DISABLED:
                draw.rect(surface, self.bg_disabled_color, self.rect, 0, Button.CORNER_RADIUS)
                draw.rect(surface, self.border_disabled_color, self.rect, Button.BORDER_WIDTH, Button.CORNER_RADIUS)
            case Button.State.IDLE:
                draw.rect(surface, self.bg_idle_color, self.rect, 0, Button.CORNER_RADIUS)
                draw.rect(surface, self.border_idle_color, self.rect, Button.BORDER_WIDTH, Button.CORNER_RADIUS)
            case Button.State.HOVERED:
                draw.rect(surface, self.bg_hovered_color, self.rect, 0, Button.CORNER_RADIUS)
                draw.rect(surface, self.border_hovered_color, self.rect, Button.BORDER_WIDTH, Button.CORNER_RADIUS)
            case Button.State.PRESSED:
                draw.rect(surface, self.bg_pressed_color, self.rect, 0, Button.CORNER_RADIUS)
                draw.rect(surface, self.border_pressed_color, self.rect, Button.BORDER_WIDTH, Button.CORNER_RADIUS)
