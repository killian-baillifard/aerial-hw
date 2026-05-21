import numpy as np
from enum import Enum
import socket
from contextlib import contextmanager


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

@contextmanager
def no_network():
    _orig_getaddrinfo = socket.getaddrinfo
    _orig_connect = socket.socket.connect
    socket.getaddrinfo = lambda *a, **k: (_ for _ in ()).throw(OSError("blocked"))
    socket.socket.connect = lambda self, addr: (_ for _ in ()).throw(OSError("blocked"))
    try:
        yield
    finally:
        socket.getaddrinfo = _orig_getaddrinfo
        socket.socket.connect = _orig_connect
