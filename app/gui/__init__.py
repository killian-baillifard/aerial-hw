import os, pygame
import numpy as np
from threading import Lock
from cv2.typing import MatLike
from pyglm import glm
from app import Link, PlanStage, ControlMode, CommandSource
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
        self.connect_event: Event[Link] = Event[Link]()
        self.disconnect_event: Event = Event()
        self.tkof_event: Event = Event()
        self.land_event: Event = Event()
        self.start_recording_event: Event = Event()
        self.stop_recording_event: Event = Event()
        self.manual_cmd_selected_event: Event[CommandSource] = Event[CommandSource]()
        self.planner_selected_event: Event[PlanStage] = Event[PlanStage]()

        # Register event listeners
        self.layout.link_btn.release_event      += self.link_btn_click_handler
        self.layout.ctrl_btn.release_event      += self.ctrl_btn_click_handler
        self.layout.source_btn.release_event    += self.source_btn_click_handler
        self.layout.con_btn.release_event       += self.con_btn_click_handler
        self.layout.vws_btn.release_event       += self.vws_btn_click_handler
        self.layout.tkof_land_btn.release_event += self.tkof_land_btn_click_handler
        self.layout.rec_btn.release_event       += self.rec_btn_click_handler

        # Initialize state
        self.lock = Lock()
        self.link = Link.SIMULATION
        self.control_mode = ControlMode.MANUAL
        self.command_source = CommandSource.CONTROLLER
        self.plan_stage = PlanStage.SCAN

    def update_command_indicators(self, command: Command) -> None:
        self.layout.xy_joystick.set_delta(glm.ivec2(-command.velocity.y * Gui.JOYSTICKS_LEN, -command.velocity.x * Gui.JOYSTICKS_LEN))
        self.layout.z_joystick.set_delta(glm.ivec2(0, -command.velocity.z * Gui.JOYSTICKS_LEN))
        self.layout.yaw_joystick.set_delta(glm.ivec2(-(command.yaw_rate / Command.YAW_RATE) * Gui.JOYSTICKS_LEN, 0))
        if self.layout.rec_btn.latched:
            self.layout.shutter_indicator.trigger()

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
        surface = pygame.image.frombuffer(frame.tobytes(), (w, h), "RGB")
        self.layout.camera_image.set_color_image(surface)

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

    def link_btn_click_handler(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        match self.link:
            case Link.SIMULATION:
                self.link = Link.WIFI
            case Link.WIFI:
                self.link = Link.RADIO
            case Link.RADIO:
                self.link = Link.SIMULATION
        match self.link:
            case Link.SIMULATION:
                self.layout.link_btn.set_text("LINK [SIM]")
            case Link.WIFI:
                self.layout.link_btn.set_text("LINK [WIFI]")
            case Link.RADIO:
                self.layout.link_btn.set_text("LINK [RADIO]")

    def update_control_mode(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        match self.control_mode:
            case ControlMode.MANUAL:
                self.layout.ctrl_btn.set_text("CTRL [MAN]")
                self.manual_cmd_selected_event(self.command_source)
                match self.command_source:
                    case CommandSource.KEYBOARD:
                        self.layout.source_btn.set_text("KEYBOARD")
                    case CommandSource.CONTROLLER:
                        self.layout.source_btn.set_text("CONTROLLER")
            case ControlMode.PLANNER:
                self.layout.ctrl_btn.set_text("CTRL [PLAN]")
                self.planner_selected_event(self.plan_stage)
                match self.plan_stage:
                    case PlanStage.SCAN:
                        self.layout.source_btn.set_text("STAGE [SCAN]")
                    case PlanStage.RACE:
                        self.layout.source_btn.set_text("STAGE [RACE]")

    def ctrl_btn_click_handler(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        match self.control_mode:
            case ControlMode.MANUAL:
                self.control_mode = ControlMode.PLANNER
            case ControlMode.PLANNER:
                self.control_mode = ControlMode.MANUAL
        self.update_control_mode()

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
                match self.plan_stage:
                    case PlanStage.SCAN:
                        self.plan_stage = PlanStage.RACE
                    case PlanStage.RACE:
                        self.plan_stage = PlanStage.SCAN
        self.update_control_mode()

    def con_btn_click_handler(self) -> None:
        self.audio.play(Audio.Track.BUTTON)
        self.layout.con_btn.set_text("CON [...]")
        self.layout.con_btn.disable()
        if self.layout.con_btn.latched:
            self.layout.link_btn.disable()
            self.connect_event(self.link)
        else:
            self.layout.tkof_land_btn.disable()
            self.disconnect_event()

    def connected(self) -> None:
        self.lock.acquire()
        self.layout.con_btn.set_text("CON [ON]")
        self.layout.con_btn.enable()
        self.layout.tkof_land_btn.enable()
        self.lock.release()

    def disconnected(self) -> None:
        self.lock.acquire()
        if self.layout.con_btn.latched:
            self.layout.con_btn.release_handler()
        self.layout.con_btn.set_text("CON [OFF]")
        self.layout.con_btn.enable()
        self.layout.link_btn.enable()
        self.lock.release()

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
            self.layout.con_btn.disable()
            self.tkof_event()
        else:
            self.layout.tkof_land_btn.set_text("LAND")
            self.land_event()

    def airborn(self) -> None:
        self.layout.tkof_land_btn.set_text("AIRBORN")
        self.layout.tkof_land_btn.enable()

    def landed(self) -> None:
        self.layout.tkof_land_btn.set_text("LANDED")
        self.layout.tkof_land_btn.enable()
        self.layout.con_btn.enable()

    def rec_btn_click_handler(self) -> None:
        self.audio.play(Audio.Track.SHUTTER)
        if self.layout.rec_btn.latched:
            self.layout.rec_btn.set_text("REC [ON]")
            self.start_recording_event()
        else:
            self.layout.rec_btn.set_text("REC [OFF]")
            self.stop_recording_event()

    def quit(self) -> None:
        pygame.quit()
