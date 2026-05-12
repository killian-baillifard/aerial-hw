import time
from app import *
from app.gui import Gui
from app.io import Command
from app.io.controller import Controller
from app.io.keyboard import Keyboard
from app.telemetry import Telemetry, TelemetryFlags
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
        self.scan_planner   = ScanPlanner()
        self.race_planner   = RacePlanner()
        self.flight_status  = FlightStatus.LANDED

        # Initialize event handlers
        self.gui.connect_event              += self.telemetry.connect
        self.gui.disconnect_event           += self.telemetry.disconnect
        self.gui.tkof_event                 += self.tkof_event_handler
        self.gui.land_event                 += self.land_event_handler
        self.gui.start_recording_event      += self.recorder.start_recording
        self.gui.stop_recording_event       += self.recorder.stop_recording
        self.gui.manual_cmd_selected_event  += self.manual_cmd_selected_event_handler
        self.gui.planner_selected_event     += self.planner_selected_event_handler
        self.telemetry.connected_event      += self.gui.connected
        self.telemetry.disconnected_event   += self.gui.disconnected
        self.scan_planner.gate_found_event  += self.gui.layout.scene.add_gate

    def tkof_event_handler(self) -> None:
        self.flight_status = FlightStatus.TKOF

    def land_event_handler(self) -> None:
        self.flight_status = FlightStatus.LAND

    def manual_cmd_selected_event_handler(self, source: CommandSource) -> None:
        self.telemetry.z = self.telemetry.measurement.read().position.z

    def planner_selected_event_handler(self, stage: PlanStage) -> None:
        self.gui.layout.scene.gates.clear()
        match stage:
            case PlanStage.SCAN:
                self.scan_planner.reload()
            case PlanStage.RACE:
                self.race_planner.reload()
                for gate in self.race_planner.gates:
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
            flags = TelemetryFlags.NEITHER
            if new_measurement:
                flags |= TelemetryFlags.NEW_MEASUREMENT
                self.gui.update_measurement_indicators(measurement)
            if new_frame:
                flags |= TelemetryFlags.NEW_FRAME
                self.gui.update_camera_image(frame)
                self.recorder.record(measurement, frame)

            # Compute command / setpoint
            match self.flight_status:

                case FlightStatus.LANDED:
                    self.telemetry.command = Command()

                case FlightStatus.TKOF:
                    if self.telemetry.tkof(dt):
                        self.flight_status = FlightStatus.AIRBORN
                        self.gui.airborn()

                case FlightStatus.AIRBORN:
                    match self.gui.control_mode:
                        case ControlMode.MANUAL:
                            match self.gui.command_source:
                                case CommandSource.KEYBOARD:
                                    self.keyboard.update(dt)
                                    self.telemetry.send_command(self.keyboard, dt)
                                case CommandSource.CONTROLLER:
                                    self.controller.update(dt)
                                    self.telemetry.send_command(self.controller, dt)
                        case ControlMode.PLANNER:
                            match self.gui.plan_stage:
                                case PlanStage.SCAN:
                                        self.telemetry.send_setpoint(self.scan_planner.update(measurement, frame, flags, dt), dt)
                                case PlanStage.RACE:
                                    self.telemetry.send_setpoint(self.race_planner.update(measurement, frame, flags, dt), dt)
                
                case FlightStatus.LAND:
                    if self.telemetry.land(dt):
                        self.flight_status = FlightStatus.LANDED
                        self.gui.landed()

            # Update GUI
            self.gui.update_command_indicators(self.telemetry.command)
            self.gui.render(dt)
            quit = self.gui.poll_events()
        
        self.gui.quit()

if __name__ == "__main__":
    app = App()
    app.run()
