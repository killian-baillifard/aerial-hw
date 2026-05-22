from typing import Generic, Callable, TypeVarTuple, Self

Ts = TypeVarTuple('Ts')

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
