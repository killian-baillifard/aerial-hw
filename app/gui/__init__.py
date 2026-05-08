import os, pygame
import numpy as np
from enum import Enum
from cv2.typing import MatLike
from pyglm import glm
from app.gui.widgets import Widget
from app.gui.layout import Layout
from app.gui.audio import Audio
from app.gui.voicewarningsystem import VoiceWarningSystem
from app.telemetry.measurement import Measurement
from app.inputs import Input

class Gui:

    CLEAR_COLOR = (30, 30, 30)
    JOYSTICKS_LEN = 200

    class Link(Enum):
        SIMULATION  = 0
        WIFI        = 1
        RADIO       = 2

    class ControlMode(Enum):
        MANUAL  = 0
        PLANNER = 1

    class InputSource(Enum):
        KEYBOARD    = 0
        CONTROLLER  = 1

    class LapType(Enum):
        SCAN    = 0
        RACE    = 1

    class ConnectionStatus(Enum):
        DISCONNECTED    = 0
        CONNECTING      = 1
        CONNECTED       = 2
        DISCONNECTING   = 3

    def __init__(self) -> None:

        # Initialize window
        pygame.init()
        pygame.joystick.init()
        icon = pygame.image.load(os.path.join("assets", "icon.png"))
        pygame.display.set_icon(icon)
        self.screen = pygame.display.set_mode(Layout.WINDOW_SIZE)
        pygame.display.set_caption("Crazyflie telemetry")

        # Intialize submodules
        self.layout = Layout()
        self.audio = Audio()
        self.voice_warning_system = VoiceWarningSystem(self.audio)

        # Register event listeners
        self.layout.link_btn.add_release_handler(self.link_btn_click_handler)
        self.layout.ctrl_btn.add_release_handler(self.ctrl_btn_click_handler)
        self.layout.src_lap_btn.add_release_handler(self.src_lap_btn_click_handler)
        self.layout.con_btn.add_release_handler(self.con_btn_click_handler)
        self.layout.vws_btn.add_release_handler(self.vws_btn_click_handler)
        self.layout.eng_btn.add_release_handler(self.eng_btn_click_handler)
        self.layout.rec_btn.add_release_handler(self.rec_btn_click_handler)

        # Initialize state
        self.link = Gui.Link.SIMULATION
        self.control_mode = Gui.ControlMode.MANUAL
        self.input_source = Gui.InputSource.KEYBOARD
        self.lap_type = Gui.LapType.SCAN
        self.connection_status = Gui.ConnectionStatus.DISCONNECTED

    def update(self, measurement: Measurement, frame: MatLike, input: Input, dt: float) -> bool:

        # Update sensors measurement indicators
        self.layout.x_indicator.set_text(f"[X = {measurement.position.x:.3f} m]")
        self.layout.y_indicator.set_text(f"[Y = {measurement.position.y:.3f} m]")
        self.layout.z_indicator.set_text(f"[Z = {measurement.position.z:.3f} m]")
        self.layout.yaw_indicator.set_text(f"[YAW = {np.rad2deg(measurement.rotation.z):.3f} °]")
        self.layout.roll_indicator.set_roll(measurement.rotation.x)
        self.layout.pitch_indicator.set_pitch(measurement.rotation.y * Gui.JOYSTICKS_LEN)
        self.layout.batt_indicator.set_text(f"[BATT = {int(100 * measurement.battery):d} %]")
        self.layout.batt_gauge.set_progress(measurement.battery)

        # Update camera image
        h, w = frame.shape[:2]
        surface = pygame.image.frombuffer(frame.tobytes(), (w, h), "RGB")
        self.layout.camera_image.set_color_image(surface)

        # Update input indicators
        self.layout.xy_joystick.set_delta(glm.ivec2(-input.position.y * Gui.JOYSTICKS_LEN, -input.position.x * Gui.JOYSTICKS_LEN))
        self.layout.z_joystick.set_delta(glm.ivec2(0, -input.position.z * Gui.JOYSTICKS_LEN))
        self.layout.yaw_joystick.set_delta(glm.ivec2(-input.yaw * Gui.JOYSTICKS_LEN, 0))

        # Update widgets and submodules logic
        Widget.update_instances()
        self.voice_warning_system.update(measurement, dt)

        # Draw frame
        self.screen.fill(Gui.CLEAR_COLOR)
        Widget.draw_instances(self.screen)
        pygame.display.flip()

        # Poll window events
        quit = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit = True
        return quit

    def link_btn_click_handler(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        match self.link:
            case Gui.Link.SIMULATION:   self.link = Gui.Link.WIFI
            case Gui.Link.WIFI:         self.link = Gui.Link.RADIO
            case Gui.Link.RADIO:        self.link = Gui.Link.SIMULATION
        match self.link:
            case Gui.Link.SIMULATION:
                self.layout.link_btn.set_text("LINK [SIM]")
                self.layout.con_btn.disable()
                self.layout.eng_btn.enable()
            case Gui.Link.WIFI:
                self.layout.link_btn.set_text("LINK [WIFI]")
                self.layout.con_btn.enable()
                self.layout.eng_btn.disable()
            case Gui.Link.RADIO:
                self.layout.link_btn.set_text("LINK [RADIO]")
                self.layout.con_btn.enable()
                self.layout.eng_btn.disable()

    def update_ctrl_src_lap_buttons(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        match self.control_mode:
            case Gui.ControlMode.MANUAL:
                self.layout.ctrl_btn.set_text("CTRL [MAN]")
                match self.input_source:
                    case Gui.InputSource.KEYBOARD:
                        self.layout.src_lap_btn.set_text("SRC [KEYBOARD]")
                    case Gui.InputSource.CONTROLLER:
                        self.layout.src_lap_btn.set_text("SRC [CONTROLLER]")
            case Gui.ControlMode.PLANNER:
                self.layout.ctrl_btn.set_text("CTRL [PLAN]")
                match self.lap_type:
                    case Gui.LapType.SCAN:
                        self.layout.src_lap_btn.set_text("LAP [SCAN]")
                    case Gui.LapType.RACE:
                        self.layout.src_lap_btn.set_text("LAP [RACE]")

    def ctrl_btn_click_handler(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        match self.control_mode:
            case Gui.ControlMode.MANUAL:    self.control_mode = Gui.ControlMode.PLANNER
            case Gui.ControlMode.PLANNER:   self.control_mode = Gui.ControlMode.MANUAL
        self.update_ctrl_src_lap_buttons()

    def src_lap_btn_click_handler(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        match self.control_mode:
            case Gui.ControlMode.MANUAL:
                match self.input_source:
                    case Gui.InputSource.KEYBOARD:      self.input_source = Gui.InputSource.CONTROLLER
                    case Gui.InputSource.CONTROLLER:    self.input_source = Gui.InputSource.KEYBOARD
            case Gui.ControlMode.PLANNER:
                match self.lap_type:
                    case Gui.LapType.SCAN:  self.lap_type = Gui.LapType.RACE
                    case Gui.LapType.RACE:  self.lap_type = Gui.LapType.SCAN
        self.update_ctrl_src_lap_buttons()

    def con_btn_click_handler(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        match self.connection_status:
            case Gui.ConnectionStatus.DISCONNECTED:
                self.connection_status = Gui.ConnectionStatus.CONNECTING
                self.layout.con_btn.set_text("CON [...]")
                self.layout.con_btn.disable()
                print("TODO : Call connect handler")
            case Gui.ConnectionStatus.CONNECTED:
                self.connection_status = Gui.ConnectionStatus.DISCONNECTING
                self.layout.con_btn.set_text("CON [...]")
                self.layout.con_btn.disable()
                print("TODO : Call disconnect handler")

    def vws_btn_click_handler(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        if self.layout.vws_btn.latched:
            self.layout.vws_btn.set_text("VWS [ON]")
            self.voice_warning_system.enable()
        else:
            self.layout.vws_btn.set_text("VWS [OFF]")
            self.voice_warning_system.disable()

    def eng_btn_click_handler(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        if self.layout.eng_btn.latched:
            self.layout.eng_btn.set_text("ENG [ON]")
            self.layout.link_btn.disable()
            print("TODO : Ignition")
        else:
            self.layout.eng_btn.set_text("ENG [OFF]")
            self.layout.link_btn.enable()
            print("TODO : Cutoff")

    def rec_btn_click_handler(self) -> None:
        self.audio.play(Audio.Track.SHUTTER)
        if self.layout.rec_btn.latched:
            self.layout.rec_btn.set_text("REC [ON]")
            print("TODO : Start recording")
        else:
            self.layout.rec_btn.set_text("REC [OFF]")
            print("TODO : Stop recording")

    def quit(self) -> None:
        pygame.quit()
