from app.gui.widgets.button import Button

class Toggle(Button):

    def __init__(self, pos, text = "Toggle", width = 100, height = 30, disabled = False, z_index = 0, latched: bool = False) -> None:
        super().__init__(pos, text, width, height, disabled, z_index)
        self.add_release_handler(self.release_handler)
        self.latched = not latched
        self.release_handler()

    def release_handler(self) -> None:
        self.latched = not self.latched
        if self.latched:
            self.set_bg_color(
                Button.BORDER_DISABLED_COLOR,
                Button.BORDER_IDLE_COLOR,
                Button.BORDER_HOVERED_COLOR,
                Button.BORDER_PRESSED_COLOR
            )
            self.set_border_color(
                Button.BG_DISABLED_COLOR,
                Button.BG_IDLE_COLOR,
                Button.BG_HOVERED_COLOR,
                Button.BG_PRESSED_COLOR
            )
        else:
            self.set_bg_color(
                Button.BG_DISABLED_COLOR,
                Button.BG_IDLE_COLOR,
                Button.BG_HOVERED_COLOR,
                Button.BG_PRESSED_COLOR
            )
            self.set_border_color(
                Button.BORDER_DISABLED_COLOR,
                Button.BORDER_IDLE_COLOR,
                Button.BORDER_HOVERED_COLOR,
                Button.BORDER_PRESSED_COLOR
            )
