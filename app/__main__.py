import pygame
import numpy as np
from enum import Enum
from pyglm import glm
from app import wrap
from app.gui import Gui
from app.gui.audio import Audio
from app.telemetry import Telemetry
from app.camera import Camera
from app.inputs import Input
from app.inputs.controller import Controller
from app.inputs.keyboard import Keyboard
from app.log import Logger

INDICATOR_LEN = 200

class ControlMode(Enum):
    KEYBOARD    = 0
    CONTROLLER  = 1
    PLANNER     = 2

def main():

    # Initialize application modules
    gui: Gui = Gui()
    audio: Audio = Audio()
    controller: Input = Controller()
    keyboard: Input = Keyboard()
    telemetry: Telemetry = Telemetry()
    logger: Logger = Logger()
    camera: Camera = Camera()
    camera.start()

    # Initialize application state
    control_mode: ControlMode = ControlMode.CONTROLLER
    controller_connected: bool = False
    telemetry_state: Telemetry.State = Telemetry.State.DISCONNECTED
    camera_state: Camera.State = Camera.State.DISCONNECTED
    simulation: bool = False
    planner: bool = False
    quit: bool = False

    # Application loop
    while not quit:

        # Update displayed telemetry state
        new_telemetry_state = telemetry.state.get()
        if new_telemetry_state is not telemetry_state:
            telemetry_state = new_telemetry_state
            match telemetry_state:
                case Telemetry.State.DISCONNECTED:
                    gui.telemetry_button.set_text(f"TELEMETRY [OFF]")
                    gui.telemetry_button.enable()
                case Telemetry.State.CONNECTING:
                    gui.telemetry_button.set_text(f"TELEMETRY [...]")
                    gui.telemetry_button.disable()
                case Telemetry.State.CONNECTED:
                    gui.telemetry_button.set_text(f"TELEMETRY [ON]")
                    gui.telemetry_button.enable()

        # Update displayed camera state
        new_camera_state = camera.state.get()
        if new_camera_state is not camera_state:
            camera_state = new_camera_state
            match camera_state:
                case Camera.State.DISCONNECTED:
                    gui.camera_button.set_text(f"CAMERA [OFF]")
                    gui.camera_button.enable()
                case Camera.State.CONNECTING:
                    gui.camera_button.set_text(f"CAMERA [...]")
                    gui.camera_button.disable()
                case Camera.State.CONNECTED:
                    gui.camera_button.set_text(f"CAMERA [ON]")
                    gui.camera_button.enable()

        # Disable simulation button if either telemetry or camera is not disconnected
        if telemetry_state is not Telemetry.State.DISCONNECTED or camera_state is not Camera.State.DISCONNECTED:
            gui.simulation_button.disable()
        else:
            gui.simulation_button.enable()

        # Render window and gather button events
        events = gui.update()
        audio.update()

        # Play button click sound
        if len(events) > 0:
            audio.play(Audio.Track.BUTTON)
        
        # Handle button events
        new_control_mode = None
        for event in events:
            match event:

                case Gui.Event.TELEMETRY_BUTTON:
                    match telemetry_state:
                        case Telemetry.State.DISCONNECTED:
                            telemetry.connect()
                        case Telemetry.State.CONNECTED:
                            telemetry.disconnect()

                case Gui.Event.CAMERA_BUTTON:
                    match camera_state:
                        case Camera.State.DISCONNECTED:
                            camera.connect()
                        case Camera.State.CONNECTED:
                            camera.disconnect()

                case Gui.Event.CONTROLS_BUTTON:
                    match control_mode:
                        case ControlMode.KEYBOARD:
                            new_control_mode = ControlMode.CONTROLLER
                        case ControlMode.CONTROLLER:
                            new_control_mode = ControlMode.KEYBOARD

                case Gui.Event.SIMULATION_BUTTON:
                    simulation = not simulation
                    gui.simulation_button.set_text(f"SIMULATION [{'ON' if simulation else 'OFF'}]")
                    match simulation:
                        case True:
                            gui.telemetry_button.disable()
                            gui.camera_button.disable()
                        case False:
                            gui.telemetry_button.enable()
                            gui.camera_button.enable()

                case Gui.Event.PLANNER_BUTTON:
                    planner = not planner
                    gui.planner_button.set_text(f"PLANNER [{'ON' if planner else 'OFF'}]")
                    match planner:
                        case True:
                            pass
                        case False:
                            pass

                case Gui.Event.QUIT_BUTTON:
                    quit = True

        # Update displayed control mode
        new_controller_status = controller.is_connected()
        if new_control_mode is not None:
            control_mode = new_control_mode
            match control_mode:
                case ControlMode.KEYBOARD:
                    gui.controls_button.set_text("KEYBOARD [ON]")
                case ControlMode.CONTROLLER:
                    controller_connected = new_controller_status
                    gui.controls_button.set_text(f"CONTROLLER [{'ON' if controller_connected else 'OFF'}]")
        elif control_mode == ControlMode.CONTROLLER and new_controller_status != controller_connected:
            controller_connected = new_controller_status
            gui.controls_button.set_text(f"CONTROLLER [{'ON' if controller_connected else 'OFF'}]")

        # Get last telemetry and camera data
        measurement = telemetry.get_last_measurement()
        frame = camera.get_last_frame()

        # Use selected control mode
        match control_mode:
            case ControlMode.KEYBOARD:
                keyboard.update()
                manual_input = keyboard
            case ControlMode.CONTROLLER:
                controller.update()
                manual_input = controller
            case ControlMode.PLANNER:
                # setpoint = planner.update(measurement, frame)
                manual_input = None
                raise NotImplementedError("Planner not implemented yet")

        # Simulate camera and measurements
        if simulation:
            telemetry.simulate(manual_input)
            camera.simulate(measurement)

        # Compute setpoint
        setpoint = manual_input.to_setpoint(measurement)

        # Compute relative input for indicators
        relative_yaw = wrap(setpoint.yaw - measurement.rotation.z)
        relative_delta = glm.vec2(setpoint.position.x - measurement.position.x, setpoint.position.y - measurement.position.y)
        relative_xy = glm.rotateZ(glm.vec3(relative_delta.x, relative_delta.y, 0.0), -measurement.rotation.z).xy
        relative_z = setpoint.position.z - measurement.position.z

        # Update input indicators
        gui.xy_joystick.set_delta(glm.ivec2(-relative_xy.y * INDICATOR_LEN, -relative_xy.x * INDICATOR_LEN))
        gui.z_joystick.set_delta(glm.ivec2(0, -relative_z * INDICATOR_LEN))
        gui.yaw_joystick.set_delta(glm.ivec2(-relative_yaw * INDICATOR_LEN, 0))

        # Update sensors measurement indicators
        gui.x_indicator.set_text(f"[X = {measurement.position.x:.3f} m]")
        gui.y_indicator.set_text(f"[Y = {measurement.position.y:.3f} m]")
        gui.z_indicator.set_text(f"[Z = {measurement.position.z:.3f} m]")
        gui.yaw_indicator.set_text(f"[YAW = {(measurement.rotation.z * 180.0 / np.pi):.3f} °]")
        gui.roll_indicator.set_roll(measurement.rotation.x)
        gui.pitch_indicator.set_pitch(measurement.rotation.y)
        gui.batt_gauge.set_progress(measurement.battery)

        # Play audio warnings
        if telemetry_state is Telemetry.State.CONNECTED or simulation:
            if measurement.position.z > 2.0:
                audio.play(Audio.Track.ALTITUDE)
            if measurement.battery < 0.3:
                audio.play(Audio.Track.FUEL_LOW)

        # Update image
        h, w = frame.shape[:2]
        if len(frame.shape) == 3 and frame.shape[2] > 1:
            surface = pygame.image.frombuffer(frame.tobytes(), (w, h), "RGB")
            gui.camera_image.set_color_image(surface)
        else:
            surface = pygame.image.frombuffer(frame.tobytes(), (w, h), "P")
            gui.camera_image.set_grayscale_image(surface)

        # On capture press
        if manual_input is not None and manual_input.capture:

            # Display shutter indicator and play sound
            gui.shutter_indicator.trigger()
            audio.play(Audio.Track.SHUTTER)

            # Save measurement and frame
            logger.log(measurement, frame)

        # Update telemetry input command
        telemetry.set_setpoint(setpoint)
    
    camera.stop()
    gui.quit()

if __name__ == "__main__":
    main()
