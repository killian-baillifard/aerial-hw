from threading import Lock
from typing import TypeVar, Generic

T = TypeVar('T')

class Atomic(Generic[T]):

    def __init__(self, value: T) -> None:
        self._lock: Lock = Lock()
        self._value: T = value
        
    def get(self) -> T:
        self._lock.acquire()
        value = self._value
        self._lock.release()
        return value
    
    def set(self, value: T) -> None:
        self._lock.acquire()
        self._value = value
        self._lock.release()
