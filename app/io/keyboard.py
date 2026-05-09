import numpy as np
from overrides import override
from pygame import key
from pygame.key import ScancodeWrapper
from pygame.locals import *
from app.io import Command

class Keyboard(Command):

    ACCELERATION = 3.0 # m / s^2

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def virtual_axis(axis: float, keys: ScancodeWrapper, inc_key: int, dec_key: int, dt: float) -> float:

        # Compute delta
        dv = Keyboard.ACCELERATION * dt
        velocity = 0
        velocity += dv if keys[inc_key] else 0
        velocity -= dv if keys[dec_key] else 0

        # Reset speed on acceleration sign change
        if (axis > 0 and velocity < 0) or (axis < 0 and velocity > 0):
            axis = 0

        # Decelerate toward 0 when no key is pressed
        if velocity == 0:
            if np.abs(axis) <= 2 * dv:
                axis = 0
            elif axis > 0:
                axis -= dv
            elif axis < 0:
                axis += dv

        # Integrate acceleration when either key is pressed
        else:
            if np.abs(axis) + np.abs(velocity) < 1.0:
                axis += velocity
            else:
                axis = 1.0 if axis > 0 else -1.0
        
        # Return updated axis
        return axis

    @override
    def update(self, dt: float) -> None:
        
        # Acquire keys
        keys = key.get_pressed()

        # Update virtual axes
        self.velocity.x = Keyboard.virtual_axis(self.velocity.x, keys, K_w, K_s, dt)
        self.velocity.y = Keyboard.virtual_axis(self.velocity.y, keys, K_a, K_d, dt)
        self.velocity.z = Keyboard.virtual_axis(self.velocity.z, keys, K_LSHIFT, K_LCTRL, dt)
        self.yaw_rate = Keyboard.virtual_axis(self.yaw_rate, keys, K_q, K_e, dt) * Command.YAW_RATE
