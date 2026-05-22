"""
Headless Crazyflie controller.

Usage:
    python -m controller                                        # radio only, hover
    python -m controller --video                                # + AI-deck video
    python -m controller --planner controller.planners.scan_killian.ScanKillian

The radio and video threads cache data only.
The main loop ticks at fixed frequency, pulls the latest snapshot, calls
planner.update(), and ALWAYS sends a setpoint to prevent failsafe.
"""

import importlib
import time
import threading
import os

from run import *
from run.telemetry import Setpoint
from run.telemetry.radio import RadioManager
from run.video import VideoReceiver
from run.telemetry.recorder import Recorder
from run.planner import Planner

from run.planner.hover import HoverPlanner

VIDEO       = True  # whether to enable AI-deck video receiver
PLANNER     = 'hover'  # planner: "hover" or dotted module.ClassName
SAVE_DIR    = 'run_recordings'

TICK_HZ     = 10                    # Hz  (period = 100 ms)
TICK_PERIOD = 1.0 / TICK_HZ

# Sent on every tick before the first real measurement arrives.
# z = HOME_POSITION.z so the drone just holds where it is.
_SAFE_SETPOINT = Planner.HOME_SETPOINT

def load_planner(name: str) -> Planner:
    """Resolve a planner by short alias or fully-qualified module.ClassName."""
    builtins = {'hover': HoverPlanner}
    if name in builtins:
        return builtins[name]()
    module_path, _, cls_name = name.rpartition('.')
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)()


def _build_flags(new_meas: bool, new_frame: bool) -> Flags:
    flags = Flags(0)
    if new_meas:
        flags |= Flags.NEW_MEASUREMENT
    if new_frame:
        flags |= Flags.NEW_FRAME
    return flags


def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)

    planner  = load_planner(PLANNER)
    radio    = RadioManager()
    recorder = Recorder(SAVE_DIR)
    video    = VideoReceiver(SAVE_DIR) if VIDEO else None

    if video:
        video.start()
    radio.open()

    stop = threading.Event()

    def loop() -> None:
        # Track previous snapshots to detect new data
        prev_meas_ts:   float | None = None
        prev_frame_ts:  float | None = None

        # Fallback setpoint: always send something even before planner runs
        last_sp: Setpoint = _SAFE_SETPOINT

        next_tick  = time.monotonic()
        last_time  = next_tick

        while not stop.is_set():
            now  = time.monotonic()
            wait = next_tick - now
            if stop.wait(max(0.0, wait)):
                break
            dt        = time.monotonic() - last_time
            last_time = time.monotonic()
            next_tick += TICK_PERIOD

            # ---- pull latest snapshots ----------------------------------------
            meas  = radio.get_measurement()
            frame_snap = video.get_frame() if video else None  # (img, ts) | None

            # detect genuinely new data
            new_meas  = meas  is not None and meas.timestamp  != prev_meas_ts
            new_frame = frame_snap is not None and frame_snap[1] != prev_frame_ts

            if new_meas:
                prev_meas_ts  = meas.timestamp
            if new_frame:
                prev_frame_ts = frame_snap[1]

            frame = frame_snap[0] if frame_snap is not None else None
            flags = _build_flags(new_meas, new_frame)

            # ---- run planner ---------------------------------------------------
            if meas is not None:
                try:
                    sp = planner.update(meas, frame, flags, dt)
                    if sp is not None:
                        last_sp = sp
                except Exception as e:
                    print(f'Planner error: {e}')
                    # last_sp unchanged — keep sending previous setpoint

            # ---- ALWAYS send a setpoint (prevents failsafe) --------------------
            p = last_sp.position
            radio.send_setpoint(p.x, p.y, p.z, last_sp.yaw)

            # ---- record --------------------------------------------------------
            recorder.log(meas, last_sp, frame_snap)

    t = threading.Thread(target=loop, daemon=True)
    t.start()

    try:
        print(f'Controller running at {TICK_HZ} Hz  |  planner: {PLANNER}  |  video: {"ON" if VIDEO else "OFF"}')
        print('Ctrl-C to stop.')
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print('\nStopping...')
    finally:
        stop.set()
        t.join(timeout=2.0)
        if video:
            video.stop()
        recorder.close()
        radio.close()


if __name__ == '__main__':
    main()
