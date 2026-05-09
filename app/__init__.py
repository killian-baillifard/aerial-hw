import numpy as np
from enum import Enum

class Link(Enum):
    SIMULATION  = 0
    WIFI        = 1
    RADIO       = 2

class ControlMode(Enum):
    MANUAL  = 0
    PLANNER = 1

class CommandSource(Enum):
    KEYBOARD    = 0
    CONTROLLER  = 1

class PlanStage(Enum):
    SCAN    = 0
    RACE    = 1

class ConnectionStatus(Enum):
    DISCONNECTED    = 0
    CONNECT         = 1
    CONNECTED       = 2
    DISCONNECT      = 3

class FlightStatus(Enum):
    LANDED  = 0
    TKOF    = 1
    AIRBORN = 2
    LAND    = 3

def wrap(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi
