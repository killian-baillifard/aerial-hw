# Headless Crazyflie controller

## Structure

```
controller/
├── __main__.py          # entry point + main loop
├── radio.py             # radio thread  (caches latest Measurement)
├── video.py             # video thread  (caches latest Frame, saves PNGs)
├── recorder.py          # CSV logger    (tick-aligned: meas + setpoint + has_frame)
└── planners/
    ├── hover.py         # built-in hover planner (radio only)
    └── yolo_example.py  # example planner with YOLO gate detection
```

## Running

```bash
# radio only, hover planner
python -m controller

# radio + AI-deck video, hover planner (video saved but ignored by planner)
python -m controller --video

# radio + video + YOLO planner
python -m controller --video --planner controller.planners.yolo_example.YoloPlannerExample

# custom save directory
python -m controller --video --save-dir my_run
```

## Architecture

Three threads run concurrently:

| Thread | Role |
|--------|------|
| Radio  | Receives telemetry at 50 Hz, caches latest `Measurement` |
| Video  | Receives UDP JPEG stream, decodes, saves PNG, caches latest frame |
| Main loop | Ticks at 10 Hz (every 100 ms): pulls snapshot → `planner.update()` → send setpoint → log |

The radio and video threads **only cache**. The main loop owns all logic.

## Writing a planner

```python
from controller.planners.hover import PlannerBase

class MyPlanner(PlannerBase):
    def update(self, measurement, frame):
        # measurement: Measurement dataclass (or None if no radio yet)
        # frame: (np.ndarray [H,W,3 RGB], float timestamp) or None
        if measurement is None:
            return None
        return {'x': 0.0, 'y': 0.0, 'z': 0.5, 'yaw': 0.0}
```

Run with `--planner my_module.MyPlanner`.

## Why detection runs inside the planner

- Not all planners need it — no wasted inference
- Model path, confidence, and IOU thresholds are planner-specific
- The planner can skip detection on frames it has already processed (`ts != last_ts`)
- The video receiver stays a simple, dependency-free cache

## Configuration

Edit the constants at the top of each file:

| File | Constants |
|------|-----------|
| `radio.py` | `URI`, `LOG_PERIOD_MS` |
| `video.py` | `AIDECK_IP`, `AIDECK_PORT`, `LOCAL_PORT` |
| `__main__.py` | `TICK_HZ`, `SAVE_DIR` |
