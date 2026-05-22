"""Recorder: logs measurements and setpoints to a CSV file per session."""

import csv
import os
import threading
from datetime import datetime
from typing import Optional


class Recorder:
    FIELDS = ['wall_time', 'meas_ts',
              'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'battery',
              'sp_x', 'sp_y', 'sp_z', 'sp_yaw', 'has_frame']

    def __init__(self, save_dir: str = 'recordings') -> None:
        os.makedirs(save_dir, exist_ok=True)
        session   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        path      = os.path.join(save_dir, f'{session}.csv')
        self._f   = open(path, 'w', newline='')
        self._w   = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader()
        self._f.flush()
        self._lock = threading.Lock()

    def log(self, meas, sp: Optional[dict], frame) -> None:
        row = {f: '' for f in self.FIELDS}
        import time
        row['wall_time'] = f'{time.time():.6f}'
        row['has_frame'] = int(frame is not None)

        if meas is not None:
            row.update({
                'meas_ts': meas.timestamp,
                'x': meas.x, 'y': meas.y, 'z': meas.z,
                'roll': meas.roll, 'pitch': meas.pitch, 'yaw': meas.yaw,
                'battery': f'{meas.battery:.3f}',
            })
        if sp is not None:
            row.update({
                'sp_x': sp.get('x', ''), 'sp_y': sp.get('y', ''),
                'sp_z': sp.get('z', ''), 'sp_yaw': sp.get('yaw', ''),
            })
        try:
            with self._lock:
                self._w.writerow(row)
                self._f.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
