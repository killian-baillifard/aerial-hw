import pygame, os, time
from enum import Enum
from copy import deepcopy
from pyglm import glm
from app.gui.widgets import Widget
from app.gui.widgets.image import Image
from app.gui.widgets.label import Label
from app.gui.widgets.button import Button
from app.gui.widgets.joystick import Joystick
from app.gui.widgets.roll import Roll
from app.gui.widgets.pitch import Pitch
from app.gui.widgets.shutter import Shutter
from app.gui.widgets.gauge import Gauge

MARGIN = 10
BTN_WIDTH = 200
TOP_BAR = 30
WIDTH = 1280
HEIGHT = 720
BOTTOM_BAR = 30

class Gui:

    WINDOW_SIZE         = (WIDTH, HEIGHT + TOP_BAR + BOTTOM_BAR + 3 * MARGIN)
    PRIMARY_COLOR       = (6, 206, 0, 255)
    BG_COLOR            = (30, 30, 30)
    REFRESH_FREQUENCY   = 60.0
    REFRESH_PERIOD      = 1.0 / REFRESH_FREQUENCY

    class Event(Enum):
        QUIT_BUTTON         = 0
        TELEMETRY_BUTTON    = 1
        CAMERA_BUTTON       = 2
        CONTROLS_BUTTON     = 3
        SIMULATION_BUTTON   = 4
        PLANNER_BUTTON      = 5

    def __init__(self) -> None:

        # Initialize window
        pygame.init()
        pygame.joystick.init()
        icon = pygame.image.load(os.path.join('assets/icon.png'))
        pygame.display.set_icon(icon)
        self.screen = pygame.display.set_mode(Gui.WINDOW_SIZE)
        pygame.display.set_caption('Crazyfly telemetry tool')

        # Declare GUI controls

        image_center = glm.uvec2(WIDTH / 2, TOP_BAR + HEIGHT / 2 + 2 * MARGIN)

        self.telemetry_button = Button(glm.uvec2(MARGIN, MARGIN), 'TELEMETRY [OFF]', BTN_WIDTH)
        self.camera_button = Button(glm.uvec2(2 * MARGIN + BTN_WIDTH, MARGIN), 'CAMERA [OFF]', BTN_WIDTH)
        self.controls_button = Button(glm.uvec2(3 * MARGIN + 2 * BTN_WIDTH, MARGIN), 'CONTROLLER [OFF]', BTN_WIDTH)
        self.simulation_button = Button(glm.uvec2(4 * MARGIN + 3 * BTN_WIDTH, MARGIN), 'SIMULATION [OFF]', BTN_WIDTH)
        self.planner_button = Button(glm.uvec2(5 * MARGIN + 4 * BTN_WIDTH, MARGIN), 'PLANNER [OFF]', BTN_WIDTH)

        self.camera_image = Image(glm.uvec2(0, TOP_BAR + 2 * MARGIN), WIDTH, HEIGHT)

        self.xy_joystick = Joystick(image_center, glm.ivec2(0, 0))
        _ = Label(glm.uvec2(image_center.x + MARGIN, image_center.y + MARGIN), 'XY', color=Gui.PRIMARY_COLOR, z_index=2)
        
        self.z_joystick = Joystick(glm.uvec2(WIDTH - 2 * MARGIN, image_center.y), glm.ivec2(0, 0))
        _ = Label(glm.uvec2(WIDTH - 4 * MARGIN - 2, image_center.y - MARGIN + 2), 'Z', color=Gui.PRIMARY_COLOR, z_index=2)

        self.yaw_joystick = Joystick(glm.uvec2(image_center.x, TOP_BAR + HEIGHT), glm.ivec2(0, 0))
        _ = Label(glm.uvec2(image_center.x - MARGIN - 7, TOP_BAR + HEIGHT - 3 * MARGIN), 'YAW', color=Gui.PRIMARY_COLOR, z_index=2)

        self.roll_indicator = Roll(image_center, 0.0, 1)
        self.pitch_indicator = Pitch(image_center, 0.0, 1)
        self.shutter_indicator = Shutter(image_center, glm.uvec2(image_center.x - MARGIN, HEIGHT / 2 - MARGIN))

        self.x_indicator = Label(glm.uvec2(MARGIN, TOP_BAR + HEIGHT + 3 * MARGIN), '[X = 0.000 m]', color=Gui.PRIMARY_COLOR, z_index=2)
        self.y_indicator = Label(glm.uvec2(2 * MARGIN + BTN_WIDTH, TOP_BAR + HEIGHT + 3 * MARGIN), '[Y = 0.000 m]', color=Gui.PRIMARY_COLOR, z_index=2)
        self.z_indicator = Label(glm.uvec2(3 * MARGIN + 2 * BTN_WIDTH, TOP_BAR + HEIGHT + 3 * MARGIN), '[Z = 0.000 m]', color=Gui.PRIMARY_COLOR, z_index=2)
        self.yaw_indicator = Label(glm.uvec2(4 * MARGIN + 3 * BTN_WIDTH, TOP_BAR + HEIGHT + 3 * MARGIN), '[YAW = 0.000 °]', color=Gui.PRIMARY_COLOR, z_index=2)
        
        _ = Label(glm.uvec2(5 * BTN_WIDTH - 2 * MARGIN, TOP_BAR + HEIGHT + 3 * MARGIN), '[BATT]', color=Gui.PRIMARY_COLOR, z_index=2)
        self.batt_gauge = Gauge(glm.uvec2(6 * MARGIN + 5 * BTN_WIDTH, TOP_BAR + HEIGHT + 3 * MARGIN), Gauge.Direction.WEST, length=BTN_WIDTH)

        # Set event handlers
        self.telemetry_button.set_release_handler(self.on_telemetry_button_click)
        self.camera_button.set_release_handler(self.on_camera_button_click)
        self.controls_button.set_release_handler(self.on_controls_button_click)
        self.simulation_button.set_release_handler(self.on_simulation_button_click)
        self.planner_button.set_release_handler(self.on_planner_button_click)

        # Initialize gui state
        self.last_time = time.perf_counter()
        self.events: list[Gui.Event] = []

    def on_telemetry_button_click(self) -> None:
        self.events.append(Gui.Event.TELEMETRY_BUTTON)

    def on_camera_button_click(self) -> None:
        self.events.append(Gui.Event.CAMERA_BUTTON)

    def on_controls_button_click(self) -> None:
        self.events.append(Gui.Event.CONTROLS_BUTTON)

    def on_simulation_button_click(self) -> None:
        self.events.append(Gui.Event.SIMULATION_BUTTON)

    def on_planner_button_click(self) -> None:
        self.events.append(Gui.Event.PLANNER_BUTTON)

    def update(self) -> list[Event]:

        # Draw next frame
        self.screen.fill(Gui.BG_COLOR)
        Widget.draw_instances(self.screen)
        pygame.display.flip()

        # Throttle GUI execution
        new_time = time.perf_counter()
        time_delta = new_time - self.last_time
        self.last_time = new_time
        time_to_wait = Gui.REFRESH_PERIOD - time_delta
        if time_to_wait > 0:
            time.sleep(time_to_wait)

        # Poll for window close button click
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.events.append(Gui.Event.QUIT_BUTTON)
        
        # Update widgets logic
        Widget.update_instances()

        # Return GUI events
        events = deepcopy(self.events)
        self.events.clear()
        return events

    def quit(self) -> None:
        pygame.quit()
