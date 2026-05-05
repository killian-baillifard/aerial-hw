import numpy as np
from pyglm import glm
from pygame import Rect, Surface, draw, transform
from pygame.colordict import THECOLORS as COLORS
from overrides import override
from app.gui.widgets import Widget

class Image(Widget):

    PALETTE: np.ndarray = np.linspace((0, 0, 0), (255, 255, 255), 256, dtype=np.uint8)

    def __init__(self, pos: glm.uvec2, width: int = 640, height: int = 480, zindex: int = 0) -> None:
        super().__init__(zindex)
        self.rect = Rect(pos.x, pos.y, width, height)
        self.image = None

    def __del__(self) -> None:
        super().__del__()

    def set_color_image(self, surface: Surface):

        # Get image dimensions
        w = surface.get_width()
        h = surface.get_height()

        # Compute scale factor that maximizes fill while preserving aspect ratio
        scale = min(self.rect.width / w, self.rect.height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Save new image
        self.image = transform.scale(surface, (new_w, new_h))

    def set_grayscale_image(self, surface: Surface):

        # Get image dimensions
        w = surface.get_width()
        h = surface.get_height()

        # Compute scale factor that maximizes fill while preserving aspect ratio
        scale = min(self.rect.width / w, self.rect.height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Save new image
        self.image = transform.scale(surface, (new_w, new_h))
        self.image.set_palette(Image.PALETTE)

    @override
    def draw(self, surface: Surface) -> None:
        if self.image is not None:
            x = self.rect.x + (self.rect.width  - self.image.get_width())  // 2
            y = self.rect.y + (self.rect.height - self.image.get_height()) // 2
            surface.blit(self.image, (x, y))
        else:
            draw.rect(surface, COLORS["black"], self.rect)
