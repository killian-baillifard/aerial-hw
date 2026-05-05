import os
from enum import IntEnum
from pygame import mixer
from pygame.mixer import Sound, Channel

class Audio:

    COOLDOWNS: list[float] = [
        0.0,
        0.0,
        6.0,
        6.0
    ]

    class Track(IntEnum):
        BUTTON      = 0
        SHUTTER     = 1
        ALTITUDE    = 2
        FUEL_LOW    = 3

    def __init__(self) -> None:

        # Initialize mixer module
        mixer.init()
        mixer.set_reserved(len(self.Track))

        # Load sound effects
        self.sfxs = [
            Sound(os.path.join('sfx', 'button.wav')),
            Sound(os.path.join('sfx', 'shutter.wav')),
            Sound(os.path.join('sfx', 'altitude.wav')),
            Sound(os.path.join('sfx', 'fuel_low.wav'))
        ]

        # Create channels
        self.channels = [Channel(i) for i in range(len(self.sfxs))]

        # Create cooldowns
        self.cooldowns = len(Audio.COOLDOWNS) * [0.0]

    def update(self) -> None:
        for i, cooldown in enumerate(self.cooldowns):
            if cooldown > 0:
                self.cooldowns[i] = max(0.0, cooldown - (1.0 / 60.0))

    def play(self, sound: Track) -> None:
        if self.cooldowns[sound] == 0:
            self.channels[sound].play(self.sfxs[sound])
            self.cooldowns[sound] = Audio.COOLDOWNS[sound]
