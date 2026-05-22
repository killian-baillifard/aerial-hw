"""Radio thread: connects to Crazyflie, caches latest telemetry."""

import threading
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from dataclasses import dataclass
from typing import Optional, Callable

URI = 'radio://0/20/2M/E7E7E7E702'

LOG_PERIOD_MS = 20   # telemetry rate


@dataclass(frozen=True)
class Measurement:
    timestamp: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    battery: float   # 0.0 – 1.0


class RadioManager:
    def __init__(self, uri: str = URI,
                 on_measurement: Optional[Callable[[Measurement], None]] = None):
        self.uri = uri
        self._on_measurement = on_measurement
        self._cf   = None
        self._lock = threading.Lock()
        self._latest: Optional[Measurement] = None

    # ------------------------------------------------------------------ open/close
    def open(self) -> None:
        cflib.crtp.init_drivers()
        self._cf = Crazyflie(rw_cache='cache')
        self._cf.connected.add_callback(self._on_connected)
        self._cf.connection_failed.add_callback(
            lambda uri, msg: print(f'Radio connection failed: {msg}'))
        self._cf.disconnected.add_callback(
            lambda uri: print('Radio disconnected'))
        threading.Thread(target=self._cf.open_link, args=(self.uri,),
                         daemon=True).start()
        print(f'Opening radio link to {self.uri}...')

    def close(self) -> None:
        if self._cf:
            try:
                self._cf.close_link()
            except Exception:
                pass

    # ------------------------------------------------------------------ cache read
    def get_measurement(self) -> Optional[Measurement]:
        with self._lock:
            return self._latest

    # ------------------------------------------------------------------ setpoints
    def send_setpoint(self, x: float, y: float, z: float,
                      yaw_deg: float = 0.0) -> None:
        if not self._cf:
            return
        try:
            self._cf.commander.send_position_setpoint(x, y, z, yaw_deg)
        except Exception:
            pass

    def send_hover_setpoint(self, vx: float, vy: float,
                             yaw_rate: float, z: float) -> None:
        if not self._cf:
            return
        try:
            self._cf.commander.send_hover_setpoint(vx, vy, yaw_rate, z)
        except Exception:
            pass

    # ------------------------------------------------------------------ callbacks
    def _on_connected(self, uri: str) -> None:
        print(f'Radio connected: {uri}')
        try:
            self._cf.supervisor.send_arming_request(True)
        except Exception:
            pass
        self._start_logging()

    def _start_logging(self) -> None:
        lc = LogConfig('state', period_in_ms=LOG_PERIOD_MS)
        for var in ('stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z',
                    'stabilizer.roll', 'stabilizer.pitch', 'stabilizer.yaw',
                    'pm.batteryLevel'):
            lc.add_variable(var)
        lc.data_received_cb.add_callback(self._on_data)
        lc.error_cb.add_callback(lambda lc, msg: print('Log error:', msg))
        try:
            self._cf.log.add_config(lc)
            lc.start()
        except KeyError as e:
            print('Log config error:', e)

    def _on_data(self, timestamp, data, _logconf) -> None:
        try:
            m = Measurement(
                timestamp = float(timestamp),
                x         = float(data['stateEstimate.x']),
                y         = float(data['stateEstimate.y']),
                z         = float(data['stateEstimate.z']),
                roll      = float(data.get('stabilizer.roll',  0.0)),
                pitch     = float(data.get('stabilizer.pitch', 0.0)),
                yaw       = float(data.get('stabilizer.yaw',   0.0)),
                battery   = float(data.get('pm.batteryLevel',  0.0)) / 100.0,
            )
        except Exception:
            return
        with self._lock:
            self._latest = m
        if self._on_measurement:
            try:
                self._on_measurement(m)
            except Exception:
                pass
