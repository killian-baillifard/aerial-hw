from threading import Thread
from overrides import override
from enum import Enum
import socket, time, struct
import numpy as np
import cv2
import glm
from cv2.typing import MatLike
from app.sync import Atomic
from app.telemetry.measurement import Measurement

class Camera(Thread):

    IP      = "192.168.4.1"
    PORT    = 5000
    TIMEOUT = 10.0

    class State(Enum):
        DISCONNECTED    = 0
        CONNECTING      = 1
        CONNECTED       = 2

    def __init__(self) -> None:
        super().__init__(name="Camera", daemon=True)
        self.alive: Atomic[bool] = Atomic(True)
        self.state: Atomic[Camera.State] = Atomic(Camera.State.DISCONNECTED)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(Camera.TIMEOUT)
        self.image: Atomic[MatLike] = Atomic(np.zeros(shape=(320, 320), dtype=np.uint8))
    
    def receive(self, size) -> bytearray:
        data = bytearray()
        while len(data) < size:
            chunk = self.socket.recv(size - len(data))
            if not chunk:
                raise ConnectionError("Connection closed by remote end")
            data.extend(chunk)
        return data

    @override
    def run(self) -> None:
        while self.alive.get():
            match self.state.get():

                case Camera.State.DISCONNECTED:
                    time.sleep(0.001)

                case Camera.State.CONNECTING:
                    try:
                        self.socket.connect((Camera.IP, Camera.PORT))
                        self.state.set(Camera.State.CONNECTED)
                    except OSError as e:
                        print(e)
                        print("Make sure you are connected to SSID 'crazyflie_02' with password 'epfl_lis_02'")
                        self.reset_socket()

                case Camera.State.CONNECTED:

                    try:
                        # Read packet info
                        raw_packet_info = self.receive(4)
                        [length, routing, function] = struct.unpack("<HBB", raw_packet_info)
                        #print("info")
                        
                        # Read image header
                        image_header = self.receive(length - 2)
                        [magic, width, height, depth, format, size] = struct.unpack("<BHHBBI", image_header)
                        if magic != 0xBC:
                            raise ConnectionError(f"Unexpected magic number: got {magic:#x}")
                        #print("header")

                        # Read image chunk by chunk
                        image_stream = bytearray()
                        while len(image_stream) < size:
                            raw_packet_info = self.receive(4)
                            [length, dst, src] = struct.unpack("<HBB", raw_packet_info)
                            chunk = self.receive(length - 2)
                            image_stream.extend(chunk)
                            #print("chunk")

                        # Decode format
                        if format == 0:
                            bayer = np.frombuffer(image_stream, dtype=np.uint8)
                            bayer = bayer.reshape((244, 324)) # (height, width) ?
                            image = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB) # cv2.COLOR_BayerBG2BGRA ?
                            #print("color")
                        else:
                            array = np.frombuffer(image_stream, np.uint8)
                            image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
                            #print("grayscale")

                        # Set new frame
                        #print("image")
                        self.image.set(image)
                    
                    except OSError as e:
                        print(e)
                        self.reset_socket()

    def connect(self) -> None:
        if self.state.get() == Camera.State.DISCONNECTED:
            self.state.set(Camera.State.CONNECTING)

    def simulate(self, measurement: Measurement) -> MatLike:
        color = int(np.clip(255.0 * glm.length(measurement.position) / 20.0, 0.0, 255.0))
        self.image.set(color * np.ones(shape=(320, 320), dtype=np.uint8))

    def get_last_frame(self) -> MatLike:
        return self.image.get()
        
    def reset_socket(self) -> None:
        try:
            self.socket.close()
        except OSError:
            pass
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(Camera.TIMEOUT)
        self.state.set(Camera.State.DISCONNECTED)
    
    def disconnect(self) -> None:
        if self.state.get() == Camera.State.CONNECTED:
            self.reset_socket()

    def stop(self) -> None:
        self.alive.set(False)
        self.reset_socket()
        self.join()
