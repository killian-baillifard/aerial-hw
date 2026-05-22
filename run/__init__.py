import os
import numpy as np
import contextlib
import socket
from contextlib import contextmanager
from enum import Flag

class Flags(Flag):
    NEITHER         = 0
    NEW_MEASUREMENT = 1
    NEW_FRAME       = 2

@contextlib.contextmanager
def _muted_stderr():
    saved = os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(null)
        os.close(saved)

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


