"""Video thread: receives JPEG frames over UDP from AI-deck, caches latest."""

import os
import socket
import struct
import threading
import time
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np

AIDECK_IP   = '192.168.4.1'
AIDECK_PORT = 5000
LOCAL_PORT  = 5001
START_MAGIC = b'FER'

CPX_HEADER_SIZE  = 4
IMG_HEADER_MAGIC = 0xBC
IMG_HEADER_SIZE  = 11
IMG_WIDTH        = 324
IMG_HEIGHT       = 244
MIN_JPEG_BYTES   = 5000

# Suppress noisy OpenCV JPEG warnings
import contextlib

@contextlib.contextmanager
def _muted_stderr():
    import os as _os
    saved = _os.dup(2)
    null  = _os.open(_os.devnull, _os.O_WRONLY)
    try:
        _os.dup2(null, 2)
        yield
    finally:
        _os.dup2(saved, 2)
        _os.close(null)
        _os.close(saved)


# A frame snapshot: numpy image + wall-clock timestamp
Frame = Tuple[np.ndarray, float]


class VideoReceiver:
    def __init__(self, save_dir: str = 'recordings',
                 ip: str = AIDECK_IP, port: int = AIDECK_PORT,
                 local_port: int = LOCAL_PORT):
        self.save_dir   = save_dir
        self.ip         = ip
        self.port       = port
        self.local_port = local_port
        os.makedirs(save_dir, exist_ok=True)

        self._lock    = threading.Lock()
        self._latest: Optional[Frame] = None
        self._thread  = None
        self._running = False

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------ cache read
    def get_frame(self) -> Optional[Frame]:
        """Return (image_rgb, timestamp) or None."""
        with self._lock:
            return self._latest

    # ------------------------------------------------------------------ receiver loop
    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind(('0.0.0.0', self.local_port))
        sock.settimeout(1.0)
        try:
            sock.sendto(START_MAGIC, (self.ip, self.port))
        except Exception:
            pass

        buffer        = bytearray()
        expected_size = 0
        receiving     = False

        while self._running:
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) < CPX_HEADER_SIZE:
                continue
            payload = data[CPX_HEADER_SIZE:]

            # new frame header
            if len(payload) >= IMG_HEADER_SIZE and payload[0] == IMG_HEADER_MAGIC:
                try:
                    _, w, h, _, _, size = struct.unpack('<BHHBBI', payload[:IMG_HEADER_SIZE])
                except struct.error:
                    continue
                if w == IMG_WIDTH and h == IMG_HEIGHT and 0 < size < 65536:
                    expected_size = size
                    buffer        = bytearray()
                    receiving     = True
                continue

            if not receiving:
                continue

            buffer.extend(payload)
            if len(buffer) < expected_size:
                continue

            # full blob received
            soi = buffer.find(b'\xff\xd8')
            eoi = buffer.rfind(b'\xff\xd9')
            receiving = False
            if soi < 0 or eoi <= soi or (eoi + 2 - soi) < MIN_JPEG_BYTES:
                continue

            jpeg = bytes(buffer[soi:eoi + 2])
            arr  = np.frombuffer(jpeg, np.uint8)
            with _muted_stderr():
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None or img.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
                continue
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ts = time.time()
            with self._lock:
                self._latest = (img, ts)

            # save to disk (non-blocking — write in same thread; fast enough for 10 Hz log)
            self._save(img, ts)

        sock.close()

    def _save(self, img: np.ndarray, ts: float) -> None:
        fname = datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S-%f') + '.png'
        path  = os.path.join(self.save_dir, fname)
        try:
            cv2.imwrite(path, img)
        except Exception:
            pass
