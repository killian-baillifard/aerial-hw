from pyglm import glm
from app.gui.widgets.image import Image
from app.gui.widgets.label import Label
from app.gui.widgets.button import Button
from app.gui.widgets.toggle import Toggle
from app.gui.widgets.joystick import Joystick
from app.gui.widgets.roll import Roll
from app.gui.widgets.pitch import Pitch
from app.gui.widgets.shutter import Shutter
from app.gui.widgets.gauge import Gauge
from app.gui.widgets.scene import Scene

class Layout:

    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    MARGIN = 10
    SMALL_BTN_WIDTH = 150
    FULL_BTN_WIDTH = 200
    LABEL_WIDTH = 200
    TOP_BAR = 30
    BOTTOM_BAR = 30
    IMAGE_SIZE = glm.uvec2(WINDOW_WIDTH, WINDOW_HEIGHT)
    WINDOW_SIZE = IMAGE_SIZE + glm.uvec2(0.0, TOP_BAR + BOTTOM_BAR + 3 * MARGIN)
    IMAGE_CENTER = (IMAGE_SIZE / 2) + glm.uvec2(0.0, TOP_BAR + 2 * MARGIN)

    def __init__(self) -> None:

        # Top bar
        x = Layout.MARGIN
        self.link_btn = Button(glm.uvec2(x, Layout.MARGIN), "LINK [SIM]", Layout.SMALL_BTN_WIDTH)
        x += Layout.SMALL_BTN_WIDTH + Layout.MARGIN
        self.ctrl_btn = Button(glm.uvec2(x, Layout.MARGIN), "CTRL [MAN]", Layout.SMALL_BTN_WIDTH)
        x += Layout.SMALL_BTN_WIDTH + Layout.MARGIN
        self.source_btn = Button(glm.uvec2(x, Layout.MARGIN), "KEYBOARD", Layout.FULL_BTN_WIDTH)
        x += Layout.FULL_BTN_WIDTH + Layout.MARGIN
        self.con_btn = Toggle(glm.uvec2(x, Layout.MARGIN), "CON [OFF]", Layout.SMALL_BTN_WIDTH)
        x += Layout.SMALL_BTN_WIDTH + Layout.MARGIN
        self.vws_btn = Toggle(glm.uvec2(x, Layout.MARGIN), "VWS [ON]", Layout.SMALL_BTN_WIDTH, latched=True)
        x += Layout.SMALL_BTN_WIDTH + Layout.MARGIN
        self.tkof_land_btn = Toggle(glm.uvec2(x, Layout.MARGIN), "LANDED", Layout.SMALL_BTN_WIDTH, disabled=True)
        x += Layout.SMALL_BTN_WIDTH + Layout.MARGIN
        self.rec_btn = Toggle(glm.uvec2(x, Layout.MARGIN), "REC [OFF]", Layout.SMALL_BTN_WIDTH)

        # Image and overlay
        self.camera_image = Image(glm.uvec2(0, Layout.TOP_BAR + 2 * Layout.MARGIN), Layout.WINDOW_WIDTH, Layout.WINDOW_HEIGHT, z_index=0)
        self.scene = Scene(glm.uvec2(0, Layout.TOP_BAR + 2 * Layout.MARGIN), glm.vec2(Layout.WINDOW_WIDTH, Layout.WINDOW_HEIGHT), z_index=1)
        self.xy_joystick = Joystick(Layout.IMAGE_CENTER, glm.ivec2(0, 0), z_index=2)
        _ = Label(glm.uvec2(Layout.IMAGE_CENTER.x + Layout.MARGIN, Layout.IMAGE_CENTER.y + Layout.MARGIN), "XY", z_index=2)
        self.z_joystick = Joystick(glm.uvec2(Layout.WINDOW_WIDTH - 2 * Layout.MARGIN, Layout.IMAGE_CENTER.y), glm.ivec2(0, 0), z_index=2)
        _ = Label(glm.uvec2(Layout.WINDOW_WIDTH - 4 * Layout.MARGIN - 2, Layout.IMAGE_CENTER.y - Layout.MARGIN + 2), "Z", z_index=2)
        self.yaw_joystick = Joystick(glm.uvec2(Layout.IMAGE_CENTER.x, Layout.TOP_BAR + Layout.WINDOW_HEIGHT), glm.ivec2(0, 0), z_index=2)
        _ = Label(glm.uvec2(Layout.IMAGE_CENTER.x - Layout.MARGIN - 7, Layout.TOP_BAR + Layout.WINDOW_HEIGHT - 3 * Layout.MARGIN), "YAW", z_index=2)
        self.roll_indicator = Roll(Layout.IMAGE_CENTER, 0.0, z_index=2)
        self.pitch_indicator = Pitch(Layout.IMAGE_CENTER, 0.0, z_index=2)
        self.shutter_indicator = Shutter(Layout.IMAGE_CENTER, glm.uvec2(Layout.IMAGE_CENTER.x - Layout.MARGIN, Layout.WINDOW_HEIGHT / 2 - Layout.MARGIN), z_index=2)

        # Bottom bar
        self.x_indicator = Label(glm.uvec2(Layout.MARGIN, Layout.TOP_BAR + Layout.WINDOW_HEIGHT + 3 * Layout.MARGIN), "[X = 0.000 m]", z_index=2)
        self.y_indicator = Label(glm.uvec2(2 * Layout.MARGIN + Layout.LABEL_WIDTH, Layout.TOP_BAR + Layout.WINDOW_HEIGHT + 3 * Layout.MARGIN), "[Y = 0.000 m]", z_index=2)
        self.z_indicator = Label(glm.uvec2(3 * Layout.MARGIN + 2 * Layout.LABEL_WIDTH, Layout.TOP_BAR + Layout.WINDOW_HEIGHT + 3 * Layout.MARGIN), "[Z = 0.000 m]", z_index=2)
        self.yaw_indicator = Label(glm.uvec2(4 * Layout.MARGIN + 3 * Layout.LABEL_WIDTH, Layout.TOP_BAR + Layout.WINDOW_HEIGHT + 3 * Layout.MARGIN), "[YAW = 0.000 °]", z_index=2)
        self.batt_indicator = Label(glm.uvec2(10 * Layout.MARGIN + 4 * Layout.LABEL_WIDTH, Layout.TOP_BAR + Layout.WINDOW_HEIGHT + 3 * Layout.MARGIN), "[BATT = 0 %]", z_index=2)
        self.batt_gauge = Gauge(glm.uvec2(6 * Layout.MARGIN + 5 * Layout.LABEL_WIDTH, Layout.TOP_BAR + Layout.WINDOW_HEIGHT + 3 * Layout.MARGIN), Gauge.Direction.WEST, length=Layout.LABEL_WIDTH)
