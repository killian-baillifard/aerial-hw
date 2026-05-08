from app.gui.audio import Audio
from app.telemetry.measurement import Measurement

class VoiceWarningSystem:

    COOLDOWNS: dict[Audio.Track, float] = {
        Audio.Track.ALTITUDE: 6.0,
        Audio.Track.FUEL_LOW: 6.0
    }

    MIN_BATTERY = 10
    MAX_ALTITUDE = 2.0

    def __init__(self, audio: Audio, enabled: bool = True) -> None:

        # Save parameters
        self.audio = audio
        self.enabled = enabled

        # Create cooldowns caches
        self.cooldowns = {key: 0.0 for key in VoiceWarningSystem.COOLDOWNS.keys()}

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def throttled_play(self, track: Audio.Track) -> None:
        if not self.cooldowns[track]:
            self.audio.play(track)
            self.cooldowns[track] = VoiceWarningSystem.COOLDOWNS[track]

    def update(self, measurement: Measurement, dt: float) -> None:

        # Decrement cooldowns
        for key, value in self.cooldowns.items():
            if value > 0:
                self.cooldowns[key] = max(0.0, value - dt)

        # Test battery warning condition
        if measurement.battery < VoiceWarningSystem.MIN_BATTERY:
            self.throttled_play(Audio.Track.FUEL_LOW)

        # Test altitude warning condition
        if measurement.position.z < VoiceWarningSystem.MAX_ALTITUDE:
            self.throttled_play(Audio.Track.ALTITUDE)
