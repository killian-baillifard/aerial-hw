from pyglm import glm
from pygame import Rect, Surface
from pygame.font import Font, SysFont
from overrides import override
from app.gui.widgets import Widget

class Label(Widget):

    def __init__(self, pos: glm.uvec2, text: str = 'Label', size: int = 20, color = 'black', z_index: int = 0) -> None:
        super().__init__(z_index)
        self.rect: Rect = Rect(pos.x, pos.y, 0, 0)
        self.text: str = text
        self.font: Font = None
        self.color = color
        self.image: Surface = None
        self.set_font('Consolas', size)

    def __del__(self) -> None:
        super().__del__()

    def get_rect(self) -> Rect:
        return self.rect
    
    def set_position(self, pos: glm.uvec2) -> None:
        self.rect = Rect(pos.x, pos.y, self.rect.w, self.rect.h)
    
    def set_text(self, text: str) -> None:
        self.text = text
        self.image = self.font.render(text, True, self.color)
        _, _, width, height = self.image.get_rect()
        self.rect = Rect(self.rect.x, self.rect.y, width, height)
    
    def set_font(self, font: str, size: int) -> None:
        self.font = SysFont(font, size)
        self.set_text(self.text)

    @override
    def draw(self, surface: Surface) -> None:
        surface.blit(self.image, (self.rect))
