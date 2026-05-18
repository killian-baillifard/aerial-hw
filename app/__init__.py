import numpy as np
from enum import Enum

class ControlMode(Enum):
    MANUAL  = 0
    PLANNER = 1

class CommandSource(Enum):
    KEYBOARD    = 0
    CONTROLLER  = 1

class FlightStatus(Enum):
    LANDED  = 0
    TKOF    = 1
    AIRBORN = 2
    LAND    = 3

def wrap(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi
