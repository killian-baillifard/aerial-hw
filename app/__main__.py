import time
from app import *
from app.gui import Gui
from app.inputs.controller import Controller
from app.inputs.keyboard import Keyboard
from app.telemetry import Telemetry
from app.planner.example import ExamplePlanner
from app.log import Logger

class App:

    MAX_FRAMERATE = 60.0 # FPS
    MIN_PERIOD = 1.0 / MAX_FRAMERATE

    def __init__(self) -> None:

        # Initialize application modules
        self.gui            = Gui()
        self.controller     = Controller()
        self.keyboard       = Keyboard()
        self.telemetry      = Telemetry()
        self.logger         = Logger()
        self.scan_planner   = ExamplePlanner()
        self.race_planner   = ExamplePlanner()
        self.flight_status  = FlightStatus.LANDED

        # Initialize event handlers
        self.gui.connect_event          += self.telemetry.connect
        self.gui.disconnect_event       += self.telemetry.disconnect
        self.gui.tkof_event             += self.tkof_event_handler
        self.gui.land_event             += self.land_event_handler
        self.gui.start_recording_event  += self.start_recording_event_handler
        self.gui.stop_recording_event   += self.stop_recording_event_handler
        self.telemetry.connected_event  += self.gui.connected
        self.telemetry.disconnected_event += self.gui.disconnected

    def tkof_event_handler(self) -> None:
        self.flight_status = FlightStatus.TKOF

    def land_event_handler(self) -> None:
        self.flight_status = FlightStatus.LAND

    def start_recording_event_handler(self) -> None:
        pass

    def stop_recording_event_handler(self) -> None:
        pass

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
            measurement = self.telemetry.get_last_measurement()
            frame = self.telemetry.get_last_frame()

            # Compute command / setpoint
            match self.flight_status:

                case FlightStatus.LANDED:
                    pass

                case FlightStatus.TKOF:
                    if self.telemetry.tkof():
                        self.flight_status = FlightStatus.AIRBORN
                        self.gui.airborn()

                case FlightStatus.AIRBORN:
                    match self.gui.control_mode:
                        case ControlMode.MANUAL:
                            match self.gui.input_source:
                                case InputSource.KEYBOARD:
                                    self.telemetry.send_command(self.keyboard)
                                case InputSource.CONTROLLER:
                                    self.telemetry.send_command(self.controller)
                        case ControlMode.PLANNER:
                            match self.gui.lap_type:
                                case LapType.SCAN:
                                    self.telemetry.send_setpoint(self.scan_planner.update(measurement, frame, dt))
                                case LapType.RACE:
                                    self.telemetry.send_setpoint(self.race_planner.update(measurement, frame, dt))
                
                case FlightStatus.LAND:
                    if self.telemetry.land():
                        self.flight_status = FlightStatus.LANDED
                        self.gui.landed()

            # Update GUI
            # TODO apply overlay on frame = overlay(frame, measurement)
            command = self.telemetry.get_last_command()
            quit = self.gui.update(measurement, frame, command, dt)
            self.telemetry.simulate(dt)
        
        self.gui.quit()

if __name__ == "__main__":
    app = App()
    app.run()
