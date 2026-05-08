import numpy as np
from overrides import override
from pygame import key
from pygame.key import ScancodeWrapper
from pygame.locals import *
from app.inputs import Input

class Keyboard(Input):

    RATE = 3.0 # 1 / s

    def __init__(self) -> None:
        super().__init__()
        self.prev_space: bool = False
    
    def reset(self) -> None:
        super().reset()
        self.prev_space = False

    @staticmethod
    def virtual_axis(axis: float, keys: ScancodeWrapper, inc_key: int, dec_key: int, dt: float) -> float:

        # Compute delta
        delta = 0
        dx = Keyboard.RATE * dt
        delta += dx if keys[inc_key] else 0
        delta -= dx if keys[dec_key] else 0

        # Reset axis on sign change
        if (axis > 0 and delta < 0) or (axis < 0 and delta > 0):
            axis = 0

        # Decrease accumulated command if no key is pressed
        if delta == 0:

            # Reset to 0 or decrease with respect to sign
            if np.abs(axis) <= 2 * dx:
                axis = 0
            elif axis > 0:
                axis -= dx
            elif axis < 0:
                axis += dx

        # Add delta to axis until it reaches maximum value
        else:
            if np.abs(axis) + np.abs(delta) < 1.0:
                axis += delta
            else:
                axis = 1.0 if axis > 0 else -1.0
        
        # Return updated axis
        return axis

    @override
    def update(self, dt: float) -> None:
        
        # Acquire keys
        keys = key.get_pressed()

        # Update virtual axes
        self.position.x = Keyboard.virtual_axis(self.position.x, keys, K_w, K_s, dt)
        self.position.y = Keyboard.virtual_axis(self.position.y, keys, K_a, K_d, dt)
        self.position.z = Keyboard.virtual_axis(self.position.z, keys, K_LSHIFT, K_LCTRL, dt)
        self.yaw = Keyboard.virtual_axis(self.yaw, keys, K_q, K_e, dt)

        # Raise capture flag on spacebar rising edge
        self.capture = keys[K_SPACE] and not self.prev_space
        self.prev_space = keys[K_SPACE]
