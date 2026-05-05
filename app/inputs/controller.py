import numpy as np
from pyglm import glm
from overrides import override
from pygame import joystick
from app.inputs import Input

class Controller(Input):

    BUTTONS_MAPPING: dict[str, int] = {"A": 0, "B": 1, "X": 2, "Y": 3}
    CROSS_MAPPING: dict[str, int]   = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
    AXIS_MAPPING: dict[str, int]    = {"LSX": 0, "LSY": 1, "RSX": 2, "RSY": 3, "TRIGX": 4, "TRIGY": 5}
    DEADZONE = 0.2

    def __init__(self) -> None:
        super().__init__()
        self.joystick = None
        self.prev_a = False

    def reset(self) -> None:
        super().reset()
        self.joystick = None
        self.prev_a = False

    @override
    def update(self) -> None:
        
        # Detect controller for plug and play behaviour
        if joystick.get_count() > 0:

            # Initialize controller when plugged in
            if self.joystick is None:
                self.joystick = joystick.Joystick(0)
                self.joystick.init()

            # Map left stick and triggers to input interface
            self.position.x = -self.joystick.get_axis(Controller.AXIS_MAPPING["LSY"])
            self.position.y = -self.joystick.get_axis(Controller.AXIS_MAPPING["LSX"])
            self.position.z = (self.joystick.get_axis(Controller.AXIS_MAPPING["TRIGY"]) - self.joystick.get_axis(Controller.AXIS_MAPPING["TRIGX"])) / 2.0
            self.yaw = -self.joystick.get_axis(Controller.AXIS_MAPPING["RSX"])

            # Apply deadzone to all axes
            self.position.xy = self.position.xy if glm.length(self.position.xy) > Controller.DEADZONE else glm.vec2(0.0, 0.0)
            self.position.z = self.position.z if np.abs(self.position.z) > Controller.DEADZONE else 0
            self.yaw = self.yaw if np.abs(self.yaw) > Controller.DEADZONE else 0

            # Map A button to capture
            a = self.joystick.get_button(Controller.BUTTONS_MAPPING["A"])
            self.capture = a and not self.prev_a
            self.prev_a = a

        elif self.joystick is not None:
            self.reset()

    def is_connected(self) -> bool:
        return self.joystick is not None
