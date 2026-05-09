import numpy as np
from pyglm import glm
from overrides import override
from pygame import joystick
from app.io import Command

class Controller(Command):

    BUTTONS_MAPPING: dict[str, int] = {"A": 0, "B": 1, "X": 2, "Y": 3}
    CROSS_MAPPING: dict[str, int]   = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
    AXIS_MAPPING: dict[str, int]    = {"LSX": 0, "LSY": 1, "RSX": 2, "RSY": 3, "TRIGX": 4, "TRIGY": 5}
    DEADZONE = 0.2

    def __init__(self) -> None:
        super().__init__()
        self.joystick = None

    @override
    def update(self, dt: float) -> None:
        
        # Detect controller for plug and play behaviour
        if joystick.get_count() > 0:

            # Initialize controller when plugged in
            if self.joystick is None:
                self.joystick = joystick.Joystick(0)
                self.joystick.init()

            # Read axes
            velocity = glm.vec2(
                -self.joystick.get_axis(Controller.AXIS_MAPPING["LSY"]),
                -self.joystick.get_axis(Controller.AXIS_MAPPING["LSX"])
            )
            yaw_rate = -self.joystick.get_axis(Controller.AXIS_MAPPING["RSX"])
            climb_rate = self.joystick.get_axis(Controller.AXIS_MAPPING["TRIGY"])
            climb_rate -= self.joystick.get_axis(Controller.AXIS_MAPPING["TRIGX"])
            climb_rate /= 2.0

            # Apply deadzone to sticks
            velocity = velocity if glm.length(velocity) > Controller.DEADZONE else glm.vec2(0.0, 0.0)
            yaw_rate = yaw_rate if np.abs(yaw_rate) > Controller.DEADZONE else 0

            # Update command
            self.velocity.xy = velocity
            self.velocity.z = climb_rate
            self.yaw_rate = yaw_rate

        elif self.joystick is not None:
            self.joystick = None

    def is_connected(self) -> bool:
        return self.joystick is not None
