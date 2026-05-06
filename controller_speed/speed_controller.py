import numpy as np

DEBUG = True

# Module-level singleton so main.py can call detection_controller.get_command()
_controller = None

def get_command(sensor_data, camera_data, dt):
    global _controller
    if _controller is None:
        _controller = SpeedController()
    return _controller.compute_command(sensor_data, camera_data, dt)

### Data structures

### Auxiliary functions

def _add_angles(lhs, rhs):
    """Adds two angles and wraps the result to the range [-pi, pi]"""
    result = lhs + rhs
    while result > np.pi:
        result -= 2 * np.pi
    while result < -np.pi:
        result += 2 * np.pi
    return result

### SpeedController

class SpeedController:
    def __init__(self):

        # State machine
        self.state = "takeoff" # takeoff, search, pass_gate, land

        # Import gate positions

        pass



    def compute_command(self, sensor_data, camera_data, dt):
        """Process camera data and sensor data to compute control command"""


        control_command = [0, 0, 0, 0]  # Placeholder for control command

        return control_command