import time
from app import *
from app.gui import Gui
from app.gui.audio import Audio
from app.io import Command
from app.io.controller import Controller
from app.io.keyboard import Keyboard
from app.telemetry import Telemetry
from app.planner import Planner
from app.planner.scan import ScanPlanner
from app.planner.race import RacePlanner
from app.telemetry.recorder import Recorder

class App:

    MAX_FRAMERATE = 60.0 # FPS
    MIN_PERIOD = 1.0 / MAX_FRAMERATE

    def __init__(self) -> None:

        # Initialize application modules
        self.gui            = Gui()
        self.controller     = Controller()
        self.keyboard       = Keyboard()
        self.telemetry      = Telemetry(self.gui.layout.scene.overlay)
        self.recorder       = Recorder()
        self.flight_status  = FlightStatus.LANDED
        self.planners: dict[str, Planner] = {
            "SCAN1": ScanPlanner(),
            "SCAN2": ScanPlanner(),
            "RACE1": RacePlanner(speed=0.25),
            "RACE2": RacePlanner(speed=1.0)
        }
        self.selected_planner = next(iter(self.planners))

        # Initialize event handlers
        self.gui.connect_radio_event            += self.telemetry.on_connect_radio
        self.gui.disconnect_radio_event         += self.telemetry.on_disconnect_radio
        self.gui.connect_wifi_event             += self.telemetry.on_connect_wifi
        self.gui.disconnect_wifi_event          += self.telemetry.on_disconnect_wifi
        self.gui.enable_sim_event               += self.telemetry.on_enable_sim
        self.gui.disable_sim_event              += self.telemetry.on_disable_sim
        self.telemetry.radio_connected_event    += self.gui.on_radio_connected
        self.telemetry.radio_disconnected_event += self.gui.on_radio_disconnected
        self.telemetry.wifi_connected_event     += self.gui.on_wifi_connected
        self.telemetry.wifi_disconnected_event  += self.gui.on_wifi_disconnected
        self.gui.mode_changed_event             += self.mode_changed_event_handler
        self.gui.planner_changed_event          += self.planner_change_event_handler
        self.gui.tkof_event                     += self.tkof_event_handler
        self.gui.land_event                     += self.land_event_handler
        self.gui.start_recording_event          += self.recorder.start_recording
        self.gui.stop_recording_event           += self.recorder.stop_recording
        for _, planner in self.planners.items():
            planner.gate_found_event            += self.gui.layout.scene.add_gate

    def tkof_event_handler(self) -> None:
        self.flight_status = FlightStatus.TKOF

    def land_event_handler(self) -> None:
        self.flight_status = FlightStatus.LAND

    def mode_changed_event_handler(self, mode: ControlMode) -> None:
        match mode:
            case ControlMode.MANUAL:
                self.telemetry.z = self.telemetry.measurement.read().position.z
            case ControlMode.PLANNER:
                self.gui.layout.source_btn.set_text(self.selected_planner)

    def planner_change_event_handler(self) -> None:

        # Find next planner in dict
        planner_iter = iter(self.planners)
        for key in planner_iter:
            if key == self.selected_planner:
                self.selected_planner = next(planner_iter, next(iter(self.planners)))
                break

        # Reload planner and show initialized gates
        self.gui.layout.source_btn.set_text(self.selected_planner)
        self.planners[self.selected_planner].reload()
        self.gui.layout.scene.gates.clear()
        for gate in self.planners[self.selected_planner].gates:
            self.gui.layout.scene.add_gate(gate)

    def run(self) -> None:
        quit: bool      = False
        old_t: float    = time.perf_counter()
        while not quit:

            # Compute dt
            new_t = time.perf_counter()
            dt = new_t - old_t
            if dt < App.MIN_PERIOD:
                time.sleep(App.MIN_PERIOD - dt)
                new_t = time.perf_counter()
                dt = new_t - old_t
            old_t = new_t

            # Read telemetry
            measurement, new_measurement = self.telemetry.measurement.get()
            frame, new_frame = self.telemetry.frame.get()
            flags = Telemetry.Flags.NEITHER
            if new_measurement:
                flags |= Telemetry.Flags.NEW_MEASUREMENT
                self.gui.update_measurement_indicators(measurement)
            if new_frame:
                flags |= Telemetry.Flags.NEW_FRAME
                self.gui.update_camera_image(frame)
                self.recorder.record(measurement, frame)

            # Select command source
            match self.gui.command_source:
                case CommandSource.KEYBOARD:
                    command = self.keyboard
                case CommandSource.CONTROLLER:
                    command = self.controller
                case _:
                    command = Command()
            command.update(dt)

            # Compute command / setpoint
            match self.flight_status:

                case FlightStatus.LANDED:
                    self.telemetry.command = Command()

                case FlightStatus.TKOF:
                    if self.telemetry.tkof(dt):
                        self.flight_status = FlightStatus.AIRBORN
                        self.telemetry.z = self.telemetry.measurement.read().position.z
                        self.gui.on_airborn()
                
                case FlightStatus.AIRBORN:
                    match self.gui.control_mode:
                        case ControlMode.MANUAL:
                            self.telemetry.send_command(command, dt)
                        case ControlMode.PLANNER:
                            if self.gui.layout.plan_btn.latched:
                                setpoint = self.planners[self.selected_planner].update(measurement, frame, flags, dt)
                                self.telemetry.send_setpoint(setpoint, dt)
                
                case FlightStatus.LAND:
                    if self.telemetry.land(dt):
                        self.flight_status = FlightStatus.LANDED
                        self.telemetry.z = self.telemetry.measurement.read().position.z
                        self.gui.on_landed()

            # Capture gates positions and yaw when capture button is pressed
            if command.capture:
                self.recorder.add_gate(measurement)
                self.gui.audio.play(Audio.Track.SHUTTER)
                self.gui.layout.shutter_indicator.trigger()

            # Update GUI
            self.gui.update_command_indicators(self.telemetry.command)
            self.gui.render(dt)
            quit = self.gui.poll_events()
        
        self.gui.on_quit()

if __name__ == "__main__":
    app = App()
    app.run()
