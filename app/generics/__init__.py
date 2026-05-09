from threading import Lock
from typing import TypeVar, Generic, Callable, TypeVarTuple, Self

Ts = TypeVarTuple('Ts')

T = TypeVar("T")

class Atomic(Generic[T]):

    def __init__(self, value: T) -> None:
        self.lock: Lock = Lock()
        self.value: T = value
        
    def get(self) -> T:
        self.lock.acquire()
        value = self.value
        self.lock.release()
        return value
    
    def set(self, value: T) -> None:
        self.lock.acquire()
        self.value = value
        self.lock.release()

class Mailbox(Generic[T]):

    def __init__(self, value: T) -> None:
        self.lock: Lock = Lock()
        self.value: T = value
        self.updated = True
        
    def get(self) -> tuple[T, bool]:
        self.lock.acquire()
        value = self.value
        updated = self.updated
        self.updated = False
        self.lock.release()
        return value, updated
    
    def read(self) -> T:
        self.lock.acquire()
        value = self.value
        self.lock.release()
        return value
    
    def set(self, value: T) -> None:
        self.lock.acquire()
        self.value = value
        self.updated = True
        self.lock.release()

class Event(Generic[*Ts]):
    
    def __init__(self) -> None:
        self.listeners: list[Callable[[*Ts], None]] = []

    def __iadd__(self, listener: Callable[[*Ts], None]) -> Self:
        self.listeners.append(listener)
        return self

    def __isub__(self, listener: Callable[[*Ts], None]) -> Self:
        self.listeners.remove(listener)
        return self

    def __call__(self, *args: *Ts) -> None:
        for listener in self.listeners:
            listener(*args)
