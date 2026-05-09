import os
from enum import StrEnum
from pygame import mixer
from pygame.mixer import Sound, Channel

class Audio:

    class Track(StrEnum):
        BUTTON      = "button"
        SHUTTER     = "shutter"
        ALTITUDE    = "altitude"
        FUEL_LOW    = "fuel_low"

    def __init__(self) -> None:

        # Initialize mixer module
        mixer.init()
        mixer.set_reserved(len(self.Track))

        # Load sound effects and create channels
        self.sounds = {key: Sound(os.path.join("sfx", f"{str(key)}.wav")) for key in Audio.Track}
        self.channels = {key: Channel(i) for i, key in enumerate(Audio.Track)}

    def play(self, track: Track) -> None:
        self.channels[track].play(self.sounds[track])
