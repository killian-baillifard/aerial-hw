import os, logging, struct, time
import numpy as np
import cv2
from typing import Callable
from cv2.typing import MatLike
from threading import Thread
from overrides import override
from pyglm import glm
import cflib.crtp
from cflib.utils import uri_helper
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from app.generics import Atomic, Mailbox, Event
from app.io import Command, Setpoint
from app.io import Measurement
from app.telemetry.camera import WIDTH, HEIGHT
from enum import Enum, Flag
import socket

class Telemetry(Thread):

    RADIO_URI       = uri_helper.uri_from_env(default="radio://0/20/2M/E7E7E7E702")
    UPDATE_PERIOD   = 20    # ms
    
    TKOF_HEIGHT     = 1.0   # m
    LAND_HEIGHT     = 0.1   # m
    TOLERANCE       = 0.15  # m
    MIN_HEIGHT      = 0.05  # m

    AIDECK_IP   = '192.168.4.1'
    AIDECK_PORT = 5000
    LOCAL_PORT  = 5001
    START_MAGIC = b'FER'
    CPX_HEADER_SIZE  = 4
    IMG_HEADER_MAGIC = 0xBC
    IMG_HEADER_SIZE  = 11
    MIN_JPEG_BYTES   = 5000

    class Flags(Flag):
        NEITHER         = 0
        NEW_MEASUREMENT = 1
        NEW_FRAME       = 2
        BOTH            = NEW_MEASUREMENT | NEW_FRAME

    class WifiState(Enum):
        DISCONNECTED    = 0
        CONNECT         = 1
        CONNECTED       = 2
        DISCONNECT      = 3

    #------------------------------ #
    #   Constructor                 #
    #------------------------------ #

    def __init__(self, sim_draw_function: Callable[[MatLike], None]) -> None:
        super().__init__(name='Telemetry', daemon=True)
        logging.basicConfig(level=logging.ERROR)
        cflib.crtp.init_drivers()

        # Create log configuration
        self.log_config = LogConfig("Telemetry", Telemetry.UPDATE_PERIOD)
        self.log_config.add_variable("stateEstimate.x")
        self.log_config.add_variable("stateEstimate.y")
        self.log_config.add_variable("stateEstimate.z")
        self.log_config.add_variable("stabilizer.roll")
        self.log_config.add_variable("stabilizer.pitch")
        self.log_config.add_variable("stabilizer.yaw")
        self.log_config.add_variable("pm.batteryLevel")
        self.log_config.data_received_cb.add_callback(self.on_data_received)
        self.log_config.error_cb.add_callback(self.on_data_error)

        # Create events
        self.radio_connected_event: Event = Event()
        self.radio_disconnected_event: Event = Event()
        self.wifi_connected_event: Event = Event()
        self.wifi_disconnected_event: Event = Event()
        
        # Register event handlers
        self.crazyflie = Crazyflie(ro_cache=None, rw_cache=os.path.join("cache"))
        self.crazyflie.connected.add_callback(self.on_radio_connected)
        self.crazyflie.connection_failed.add_callback(self.on_radio_connection_failed)
        self.crazyflie.disconnected.add_callback(self.on_radio_disconnected)

        # Telemetry state
        self.radio_connected: Atomic[bool] = Atomic(False)
        self.sim_enabled: Atomic[bool] = Atomic(False)
        self.wifi_state: Atomic[Telemetry.WifiState] = Atomic(Telemetry.WifiState.DISCONNECTED)
        self.sim_draw_function = sim_draw_function
        self.measurement: Mailbox[Measurement] = Mailbox(Measurement())
        self.frame: Mailbox[MatLike] = Mailbox(np.zeros(shape=(HEIGHT, WIDTH), dtype=np.uint8))
        self.command: Command = Command()
        self.z: float = 0

        # Start camera thread
        self.start()

    #------------------------------ #
    #   Radio event handlers        #
    #------------------------------ #

    def on_data_received(self, timestamp, data, logconf):

        # Parse fields
        position = glm.vec3(
            data["stateEstimate.x"],
            data["stateEstimate.y"],
            data["stateEstimate.z"]
        )
        rotation = glm.vec3(
            np.deg2rad(data["stabilizer.roll"]),
            np.deg2rad(data["stabilizer.pitch"]),
            np.deg2rad(data["stabilizer.yaw"])
        )
        battery = float(data["pm.batteryLevel"]) / 100.0

        # Set latest measurement
        self.measurement.set(Measurement(timestamp, position, rotation, battery))

    def on_data_error(self, logconf, msg):
        print(f"Error when logging '{logconf.name}': '{msg}'")

    def on_radio_connected(self, link_uri):
        print(f"Connected to {link_uri}")
        try:
            self.crazyflie.supervisor.send_arming_request(True)
            self.crazyflie.log.add_config(self.log_config)
            self.log_config.start()
            self.radio_connected.set(True)
            self.radio_connected_event()
        except KeyError as e:
            print(f"Could not start log configuration, '{e}' not found in TOC")
        except AttributeError:
            print("Could not add Stabilizer log config, bad configuration")

    def on_radio_connection_failed(self, link_uri, msg):
        print(f"Connection to '{link_uri}' failed: '{msg}'")
        self.link.set(None)
        self.radio_disconnected_event()

    def on_radio_disconnected(self, link_uri):
        print(f"Disconnected from '{link_uri}'")
        self.link.set(None)
        self.radio_connected.set(False)
        self.radio_disconnected_event()

    #------------------------------ #
    #   External event handlers     #
    #------------------------------ #

    def on_connect_radio(self) -> None:
        Thread(target=self.crazyflie.open_link, args=(Telemetry.RADIO_URI, ), daemon=True).start()

    def on_disconnect_radio(self) -> None:
        self.crazyflie.supervisor.send_arming_request(False)
        self.crazyflie.close_link()

    def on_connect_wifi(self) -> None:
        self.wifi_state.set(Telemetry.WifiState.CONNECT)

    def on_disconnect_wifi(self) -> None:
        self.wifi_state.set(Telemetry.WifiState.DISCONNECT)

    def on_enable_sim(self) -> None:
        self.sim_enabled.set(True)

    def on_disable_sim(self) -> None:
        self.sim_enabled.set(False)

    #------------------------------ #
    #   Control inputs              #
    #------------------------------ #

    def step_simulation(self, dt: float) -> None:

        # Step simulation
        measurement = self.measurement.read()
        measurement = measurement.simulate(self.command, dt)

        # Overlay simulation on new frame
        frame = np.zeros(shape=(HEIGHT, WIDTH), dtype=np.uint8)
        self.sim_draw_function(frame)

        # Set new frame and measurement
        self.measurement.set(measurement)
        self.frame.set(frame)

    def tkof(self, dt: float) -> bool:
        dz = Telemetry.TKOF_HEIGHT - self.measurement.read().position.z
        airborn = np.abs(dz) < Telemetry.TOLERANCE
        self.command = Command(glm.vec3(0.0, 0.0, np.clip(dz, -1.0, 1.0)))
        if self.radio_connected.get():
            if not airborn:
                self.crazyflie.commander.send_zdistance_setpoint(0.0, 0.0, 0.0, Telemetry.TKOF_HEIGHT)
        elif self.sim_enabled.get():
            self.step_simulation(dt)
        return airborn

    def land(self, dt: float) -> bool:
        dz = Telemetry.LAND_HEIGHT - self.measurement.read().position.z
        landed = np.abs(dz) < Telemetry.TOLERANCE
        self.command = Command(glm.vec3(0.0, 0.0, np.clip(dz, -1.0, 1.0)))
        if self.radio_connected.get():
            if not landed:
                self.crazyflie.commander.send_zdistance_setpoint(0.0, 0.0, 0.0, Telemetry.LAND_HEIGHT)
            else:
                self.crazyflie.commander.send_stop_setpoint()
        elif self.sim_enabled.get():
            self.step_simulation(dt)
        return landed
    
    def send_command(self, command: Command, dt: float) -> None:

        # Bound z below
        self.command = command
        self.z += self.command.velocity.z * dt
        if(self.z < Telemetry.MIN_HEIGHT):
            self.z = Telemetry.MIN_HEIGHT

        # Send command or run simulation
        if self.radio_connected.get():
            self.crazyflie.commander.send_hover_setpoint(
                command.velocity.x,
                command.velocity.y,
                np.rad2deg(command.yaw_rate),
                self.z
            )
        elif self.sim_enabled.get():
            self.step_simulation(dt)
        
    def send_setpoint(self, setpoint: Setpoint, dt: float) -> None:

        # Bound z below
        if(setpoint.position.z < Telemetry.MIN_HEIGHT):
            setpoint.position.z = Telemetry.MIN_HEIGHT
        self.command = setpoint.to_command(self.measurement.read())

        # Send command or run simulation
        if self.radio_connected.get():
            self.crazyflie.commander.send_position_setpoint(
                setpoint.position.x,
                setpoint.position.y,
                setpoint.position.z,
                np.rad2deg(setpoint.yaw)
            )
        elif self.sim_enabled.get():
            self.step_simulation(dt)

    #------------------------------ #
    #   Wifi thread                 #
    #------------------------------ #

    @override
    def run(self) -> None:

        # Locals
        sock: socket.socket | None = None
        buffer: bytearray = bytearray()
        expected_size: int = 0
        receiving: bool = False
        
        while True:
            match self.wifi_state.get():

                case Telemetry.WifiState.DISCONNECTED:
                    time.sleep(0.001)

                case Telemetry.WifiState.CONNECT:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
                    sock.bind(('0.0.0.0', Telemetry.LOCAL_PORT))
                    sock.sendto(Telemetry.START_MAGIC, (Telemetry.AIDECK_IP, self.AIDECK_PORT))
                    self.wifi_state.set(Telemetry.WifiState.CONNECTED)
                    self.wifi_connected_event()

                case Telemetry.WifiState.CONNECTED:
                    
                    # Read new datagram
                    data, _ = sock.recvfrom(2048)

                    # Return early if CPX header was not received
                    if len(data) < Telemetry.CPX_HEADER_SIZE:
                        continue

                    # Discard CPX header
                    payload = data[Telemetry.CPX_HEADER_SIZE:]

                    # Check if image header was received
                    if len(payload) >= Telemetry.IMG_HEADER_SIZE and payload[0] == Telemetry.IMG_HEADER_MAGIC:
                        _, width, height, _, _, size = struct.unpack('<BHHBBI', payload[:Telemetry.IMG_HEADER_SIZE])

                        # Start reception of new image
                        if width == WIDTH and height == HEIGHT and 0 < size < (1 << 17):
                            expected_size = size
                            buffer = bytearray()
                            receiving = True
                            continue

                    # Trash payload, await start of reception
                    if not receiving:
                        continue

                    # Accumulate image data
                    buffer.extend(payload)
                    if len(buffer) < expected_size:
                        continue

                    # Look for start of image and end of image
                    soi = buffer.find(b'\xff\xd8')
                    eoi = buffer.rfind(b'\xff\xd9')
                    if soi < 0 or eoi <= soi:
                        receiving = False
                        continue
                    
                    # Corrupted frame, start over
                    jpeg_len = eoi + 2 - soi
                    if jpeg_len < Telemetry.MIN_JPEG_BYTES:
                        receiving = False
                        continue

                    # Convert JPEG image
                    jpeg = np.frombuffer(buffer, np.uint8, count=jpeg_len, offset=soi)
                    image = cv2.imdecode(jpeg, cv2.IMREAD_UNCHANGED)
                    
                    # Expect decoded image to be of the right dimensions
                    if image is None or image.shape[:2] != (HEIGHT, WIDTH):
                        receiving = False
                        continue
                    
                    # If image has 3 channels, 
                    if image.ndim == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Save image and start new reception
                    self.frame.set(image)
                    receiving = False

                case Telemetry.WifiState.DISCONNECT:
                    sock = None
                    self.wifi_state.set(Telemetry.WifiState.DISCONNECTED)
                    self.wifi_disconnected_event()
