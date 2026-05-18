import os, pygame
import numpy as np
from threading import Lock
from cv2.typing import MatLike
from pyglm import glm
from app import ControlMode, CommandSource
from app.generics import Event
from app.gui.widgets import Widget
from app.gui.layout import Layout
from app.gui.audio import Audio
from app.gui.voicewarningsystem import VoiceWarningSystem
from app.io import Measurement
from app.io import Command

class Gui:

    CLEAR_COLOR = (30, 30, 30)
    JOYSTICKS_LEN = 200

    #------------------------------ #
    #   Constructor                 #
    #------------------------------ #

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

        # Initialize events
        self.connect_radio_event: Event = Event()
        self.connect_wifi_event: Event = Event()
        self.enable_sim_event: Event = Event()
        self.disconnect_radio_event: Event = Event()
        self.disconnect_wifi_event: Event = Event()
        self.disable_sim_event: Event = Event()
        self.mode_changed_event: Event[ControlMode] = Event[ControlMode]()
        self.planner_changed_event: Event = Event()
        self.tkof_event: Event = Event()
        self.land_event: Event = Event()
        self.start_recording_event: Event = Event()
        self.stop_recording_event: Event = Event()

        # Register event listeners
        self.layout.radio_btn.release_event     += self.radio_btn_click_handler
        self.layout.wifi_btn.release_event      += self.wifi_btn_click_handler
        self.layout.sim_btn.release_event       += self.sim_btn_click_handler
        self.layout.mode_btn.release_event      += self.mode_btn_click_handler
        self.layout.source_btn.release_event    += self.source_btn_click_handler
        self.layout.plan_btn.release_event      += self.plan_btn_click_handler
        self.layout.vws_btn.release_event       += self.vws_btn_click_handler
        self.layout.tkof_land_btn.release_event += self.tkof_land_btn_click_handler
        self.layout.rec_btn.release_event       += self.rec_btn_click_handler
        self.layout.playback_btn.release_event  += self.playback_btn_click_handler

        # Initialize state
        self.lock = Lock()
        self.control_mode = ControlMode.MANUAL
        self.command_source = CommandSource.CONTROLLER

    #------------------------------ #
    #   GUI update functions        #
    #------------------------------ #

    def update_command_indicators(self, command: Command) -> None:
        self.layout.xy_joystick.set_delta(glm.ivec2(-command.velocity.y * Gui.JOYSTICKS_LEN, -command.velocity.x * Gui.JOYSTICKS_LEN))
        self.layout.z_joystick.set_delta(glm.ivec2(0, -command.velocity.z * Gui.JOYSTICKS_LEN))
        self.layout.yaw_joystick.set_delta(glm.ivec2(-(command.yaw_rate / Command.YAW_RATE) * Gui.JOYSTICKS_LEN, 0))

    def update_measurement_indicators(self, measurement: Measurement) -> None:
        self.layout.x_indicator.set_text(f"[X = {measurement.position.x:.3f} m]")
        self.layout.y_indicator.set_text(f"[Y = {measurement.position.y:.3f} m]")
        self.layout.z_indicator.set_text(f"[Z = {measurement.position.z:.3f} m]")
        self.layout.yaw_indicator.set_text(f"[YAW = {np.rad2deg(measurement.rotation.z):.3f} °]")
        self.layout.roll_indicator.set_roll(-measurement.rotation.x)
        self.layout.pitch_indicator.set_pitch(-measurement.rotation.y)
        self.layout.batt_indicator.set_text(f"[BATT = {int(100 * measurement.battery):d} %]")
        self.layout.batt_gauge.set_progress(measurement.battery)
        self.layout.scene.set_view(measurement.position, measurement.rotation)
        self.voice_warning_system.update_measurement(measurement)

    def update_camera_image(self, frame: MatLike) -> None:
        h, w = frame.shape[:2]
        surface = pygame.image.frombuffer(frame.tobytes(), (w, h), "P")
        self.layout.camera_image.set_image(surface)

    def render(self, dt: float) -> None:
        self.voice_warning_system.update_counter(dt)
        Widget.update_instances(dt)
        self.screen.fill(Gui.CLEAR_COLOR)
        self.lock.acquire()
        Widget.draw_instances(self.screen)
        self.lock.release()
        pygame.display.flip()

    def poll_events(self) -> bool:
        quit = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit = True
        return quit
    
    #----------------------------- #
    #   Buttons event handlers     #
    #----------------------------- #
    
    def radio_btn_click_handler(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        self.layout.radio_btn.set_text("TEL [...]")
        self.layout.radio_btn.disable()
        if self.layout.radio_btn.latched:
            self.layout.sim_btn.disable()
            self.layout.playback_btn.disable()
            self.connect_radio_event()
        else:
            self.layout.sim_btn.enable()
            self.layout.playback_btn.enable()
            self.disconnect_radio_event()

    def wifi_btn_click_handler(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        self.layout.wifi_btn.set_text("CAM [...]")
        self.layout.wifi_btn.disable()
        if self.layout.wifi_btn.latched:
            self.layout.sim_btn.disable()
            self.layout.playback_btn.disable()
            self.connect_wifi_event()
        else:
            self.layout.sim_btn.enable()
            self.layout.playback_btn.enable()
            self.disconnect_wifi_event()

    def sim_btn_click_handler(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        if self.layout.sim_btn.latched:
            self.layout.radio_btn.disable()
            self.layout.wifi_btn.disable()
            self.layout.tkof_land_btn.enable()
            self.layout.sim_btn.set_text("SIM [ON]")
            self.enable_sim_event()
        else:
            self.layout.radio_btn.enable()
            self.layout.wifi_btn.enable()
            self.layout.tkof_land_btn.disable()
            self.layout.sim_btn.set_text("SIM [OFF]")
            self.disable_sim_event()

    def update_control_mode_and_source(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        self.mode_changed_event(self.control_mode)
        match self.control_mode:
            case ControlMode.MANUAL:
                self.layout.mode_btn.set_text("MODE [MAN]")
                self.layout.plan_btn.disable()
                match self.command_source:
                    case CommandSource.KEYBOARD:
                        self.layout.source_btn.set_text("KEYB")
                    case CommandSource.CONTROLLER:
                        self.layout.source_btn.set_text("XBOX")
            case ControlMode.PLANNER:
                self.layout.mode_btn.set_text("MODE [PLAN]")
                self.layout.plan_btn.enable()

    def mode_btn_click_handler(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        match self.control_mode:
            case ControlMode.MANUAL:
                self.control_mode = ControlMode.PLANNER
            case ControlMode.PLANNER:
                self.control_mode = ControlMode.MANUAL
        self.update_control_mode_and_source()

    def source_btn_click_handler(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        match self.control_mode:
            case ControlMode.MANUAL:
                match self.command_source:
                    case CommandSource.KEYBOARD:
                        self.command_source = CommandSource.CONTROLLER
                    case CommandSource.CONTROLLER:
                        self.command_source = CommandSource.KEYBOARD
            case ControlMode.PLANNER:
                self.planner_changed_event()
        self.update_control_mode_and_source()

    def plan_btn_click_handler(self) -> None:
        if self.layout.plan_btn.latched:
            self.layout.plan_btn.set_text("PLAN [ON]")
            self.layout.mode_btn.disable()
            self.layout.source_btn.disable()
        else:
            self.layout.plan_btn.set_text("PLAN [OFF]")
            self.layout.mode_btn.enable()
            self.layout.source_btn.enable()

    def vws_btn_click_handler(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        if self.layout.vws_btn.latched:
            self.layout.vws_btn.set_text("VWS [ON]")
            self.voice_warning_system.enable()
        else:
            self.layout.vws_btn.set_text("VWS [OFF]")
            self.voice_warning_system.disable()

    def tkof_land_btn_click_handler(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        self.layout.tkof_land_btn.disable()
        if self.layout.tkof_land_btn.latched:
            self.layout.tkof_land_btn.set_text("TKOF")
            self.tkof_event()
        else:
            self.layout.tkof_land_btn.set_text("LAND")
            self.land_event()

    def rec_btn_click_handler(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        if self.layout.rec_btn.latched:
            self.layout.rec_btn.set_text("REC [ON]")
            self.start_recording_event()
        else:
            self.layout.rec_btn.set_text("REC [OFF]")
            self.stop_recording_event()

    def playback_btn_click_handler(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        if self.layout.playback_btn.latched:
            self.layout.playback_btn.set_text("PLAY [ON]")
            self.layout.radio_btn.disable()
            self.layout.wifi_btn.disable()
            self.layout.sim_btn.disable()
            self.layout.tkof_land_btn.disable()
            self.layout.rec_btn.disable()
        else:
            self.layout.playback_btn.set_text("PLAY [OFF]")
            self.layout.radio_btn.enable()
            self.layout.wifi_btn.enable()
            self.layout.sim_btn.enable()
            self.layout.tkof_land_btn.enable()
            self.layout.rec_btn.enable()

    #------------------------------- #
    #   External event handlers      #
    #------------------------------- #

    def on_radio_connected(self) -> None:
        self.lock.acquire()
        self.layout.radio_btn.set_text("TEL [ON]")
        self.layout.radio_btn.enable()
        self.layout.tkof_land_btn.enable()
        self.lock.release()

    def on_wifi_connected(self) -> None:
        self.lock.acquire()
        self.layout.wifi_btn.set_text("CAM [ON]")
        self.layout.wifi_btn.enable()
        self.lock.release()

    def on_radio_disconnected(self) -> None:
        self.lock.acquire()
        if self.layout.radio_btn.latched:
            self.layout.radio_btn.release_handler()
        self.layout.radio_btn.set_text("TEL [OFF]")
        self.layout.radio_btn.enable()
        self.layout.tkof_land_btn.disable()
        if not self.layout.wifi_btn.latched:
            self.layout.sim_btn.enable()
        self.lock.release()

    def on_wifi_disconnected(self) -> None:
        self.lock.acquire()
        if self.layout.wifi_btn.latched:
            self.layout.wifi_btn.release_handler()
        self.layout.wifi_btn.set_text("CAM [OFF]")
        self.layout.wifi_btn.enable()
        if not self.layout.radio_btn.latched:
            self.layout.sim_btn.enable()
        self.lock.release()

    def on_airborn(self) -> None:
        self.layout.tkof_land_btn.set_text("AIR")
        self.layout.tkof_land_btn.enable()

    def on_landed(self) -> None:
        self.layout.tkof_land_btn.set_text("GND")
        self.layout.tkof_land_btn.enable()

    def on_quit(self) -> None:
        pygame.quit()
