from pygame import Surface
from typing import Self

type Color = tuple[int, int, int, int]

class Widget:

    instances: list[Self] = []

    def __init__(self, zindex: int = 0) -> None:
        self._zindex = zindex
        Widget.instances.append(self)
        Widget.instances.sort(key=lambda item: item._zindex)
    
    def __del__(self) -> None:
        Widget.instances.remove(self)

    def update(self, dt: float) -> None:
        pass

    def draw(self, surface: Surface) -> None:
        pass

    @staticmethod
    def update_instances(dt: float) -> None:
        for instance in Widget.instances:
            instance.update(dt)

    @staticmethod
    def draw_instances(surface: Surface) -> None:
        for instance in Widget.instances:
            instance.draw(surface)
