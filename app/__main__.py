import time
from app import input_to_setpoint, setpoint_to_input
from app.gui import Gui
from app.inputs.setpoint import Setpoint
from app.telemetry import Telemetry
from app.planner import Planner
from app.planner.example import ExamplePlanner
from app.inputs import Input
from app.inputs.controller import Controller
from app.inputs.keyboard import Keyboard
from app.log import Logger
from app.telemetry.simulator import Simulator

MAX_FRAMERATE = 60.0 # FPS
MIN_PERIOD = 1.0 / MAX_FRAMERATE

def main():

    # Initialize application modules
    gui: Gui = Gui()
    controller: Input = Controller()
    keyboard: Input = Keyboard()
    planner: Planner = ExamplePlanner()
    telemetry: Telemetry = Telemetry()
    logger: Logger = Logger()
    simulator: Simulator = Simulator()

    # Initialize application state
    old_t = time.perf_counter()
    quit: bool = False

    # Application loop
    while not quit:

        # Compute dt
        new_t = time.perf_counter()
        dt = new_t - old_t
        if dt < MIN_PERIOD:
            time.sleep(MIN_PERIOD - dt)
            new_t = time.perf_counter()
            dt = new_t - old_t
        old_t = new_t

        # Get simulated / real telemetry and camera feedback
        if gui.link == Gui.Link.SIMULATION:
            measurement = simulator.get_last_measurement()
            frame = simulator.get_last_frame()
        else:
            measurement = telemetry.get_last_measurement()
            frame = telemetry.get_last_frame()

        # Read control input / setpoint from selected control source
        match gui.control_mode:
            case Gui.ControlMode.MANUAL:
                match gui.input_source:
                    case Gui.InputSource.KEYBOARD:
                        control_input = keyboard
                    case Gui.InputSource.CONTROLLER:
                        control_input = controller
                control_input.update(dt)
                setpoint = input_to_setpoint(control_input, measurement)
            case Gui.ControlMode.PLANNER:
                match gui.lap_type:
                    case Gui.LapType.SCAN:
                        setpoint: Setpoint = planner.update(measurement, frame, dt)
                    case Gui.LapType.RACE:
                        setpoint: Setpoint = planner.update(measurement, frame, dt)
                control_input = setpoint_to_input(setpoint, measurement)

        # Render window and gather button events
        quit = gui.update(measurement, frame, control_input, dt)

        # Update simulation with new setpoint
        if gui.link == Gui.Link.SIMULATION and gui.layout.eng_btn.latched:
            simulator.update(control_input, setpoint, dt)

        # On capture press
        #if capture:

            # Display shutter indicator and play sound
            #gui.shutter_indicator.trigger()
            #audio.play(Audio.Track.SHUTTER)

            # Save measurement and frame
            #logger.log(measurement, frame)

        # Update telemetry input command
        #telemetry.set_setpoint(setpoint)
    
    gui.quit()

if __name__ == "__main__":
    main()
