import os, logging, struct, time
import numpy as np
import cv2
from copy import deepcopy
from cv2.typing import MatLike
from threading import Thread
from overrides import override
from pyglm import glm
import cflib.crtp
from cflib.cpx import CPXFunction
from cflib.utils import uri_helper
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from app import Link, wrap
from app.generics import Atomic, Event
from app.inputs import Command, Setpoint
from app.telemetry.measurement import Measurement

class Telemetry(Thread):

    RADIO_URI       = uri_helper.uri_from_env(default="radio://0/20/2M/E7E7E7E702")
    WIFI_URI        = uri_helper.uri_from_env(default="tcp://192.168.4.1:5000")

    CAMERA_WIDTH    = 324   # px
    CAMERA_HEIGHT   = 244   # px

    UPDATE_PERIOD   = 20    # ms

    TKOF_HEIGHT     = 1.0   # m
    LAND_HEIGHT     = 0.1   # m
    TOLERANCE       = 0.05  # m

    def __init__(self) -> None:
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
        self.connected_event: Event = Event()
        self.disconnected_event: Event = Event()
        
        # Register session event handlers
        self.crazyflie = Crazyflie(ro_cache=None, rw_cache=os.path.join("cache"))
        self.crazyflie.connected.add_callback(self.on_connected)
        self.crazyflie.connection_failed.add_callback(self.on_connection_failed)
        self.crazyflie.disconnected.add_callback(self.on_disconnected)

        # Telemetry state
        self.link: Atomic[Link | None] = Atomic(None)
        self.connected: Atomic[bool] = Atomic(False)
        self.last_measurement: Atomic[Measurement] = Atomic(Measurement(battery=1.0))
        self.last_frame: Atomic[MatLike] = Atomic(np.zeros(shape=(Telemetry.CAMERA_HEIGHT, Telemetry.CAMERA_WIDTH, 3), dtype=np.uint8))
        self.last_command: Command = Command()

        # Start camera thread
        self.start()

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
        self.last_measurement.set(Measurement(timestamp, position, rotation, battery))

    def on_data_error(self, logconf, msg):
        print(f"Error when logging '{logconf.name}': '{msg}'")

    def on_connected(self, link_uri):
        print(f"Connected to {link_uri}")
        try:
            self.crazyflie.supervisor.send_arming_request(True)
            self.crazyflie.log.add_config(self.log_config)
            self.log_config.start()
            self.connected.set(True)
            self.connected_event()
        except KeyError as e:
            print(f"Could not start log configuration, '{e}' not found in TOC")
        except AttributeError:
            print("Could not add Stabilizer log config, bad configuration")

    def on_connection_failed(self, link_uri, msg):
        print(f"Connection to '{link_uri}' failed: '{msg}'")
        self.link.set(None)
        self.disconnected_event()

    def on_disconnected(self, link_uri):
        print(f"Disconnected from '{link_uri}'")
        self.link.set(None)
        self.connected.set(False)
        self.disconnected_event()

    def connect(self, link: Link) -> None:
        self.link.set(link)
        match link:
            case Link.SIMULATION:
                self.connected.set(True)
                self.connected_event()
            case Link.RADIO:
                self.crazyflie.open_link(Telemetry.RADIO_URI)
            case Link.WIFI:
                self.crazyflie.open_link(Telemetry.WIFI_URI)

    def disconnect(self) -> None:
        match self.link.get():
            case Link.SIMULATION:
                self.connected.set(False)
                self.disconnected_event()
            case Link.RADIO | Link.WIFI:
                self.crazyflie.supervisor.send_arming_request(False)
                self.crazyflie.close_link()

    def tkof(self) -> bool:
        
        airborn = np.abs(self.last_measurement.get().position.z - Telemetry.TKOF_HEIGHT) < Telemetry.TOLERANCE
        self.last_command = Command(altitude=Telemetry.TKOF_HEIGHT)
        match self.link.get():
            case Link.WIFI | Link.RADIO:
                if not airborn:
                    self.crazyflie.commander.send_zdistance_setpoint(0.0, 0.0, 0.0, Telemetry.TKOF_HEIGHT)
        
        return airborn

    def land(self) -> bool:

        landed = np.abs(self.last_measurement.get().position.z - Telemetry.LAND_HEIGHT) < Telemetry.TOLERANCE
        self.last_command = Command(altitude=Telemetry.LAND_HEIGHT)
        match self.link.get():
            case Link.WIFI | Link.RADIO:
                if not landed:
                    self.crazyflie.commander.send_zdistance_setpoint(0.0, 0.0, 0.0, Telemetry.LAND_HEIGHT)
                else:
                    self.crazyflie.commander.send_stop_setpoint()

        return landed

    @override
    def run(self) -> None:
        while True:

            # Return early if not connected via WiFi
            if self.link.get() is not Link.WIFI or self.connected.get() is not True:
                time.sleep(0.001)
                continue

            # Receive frame
            p = self.crazyflie.link.cpx.receivePacket(CPXFunction.APP)
            [magic, width, height, depth, fmt, size] = struct.unpack('<BHHBBI', p.data[0:11])
            if magic == 0xBC:
                buf = bytearray()
                while len(buf) < size:
                    buf.extend(self.crazyflie.link.cpx.receivePacket(CPXFunction.APP).data)
                img = np.frombuffer(buf, dtype=np.uint8)
                bayer = img.reshape(shape=(Telemetry.CAMERA_HEIGHT, Telemetry.CAMERA_WIDTH))
                color = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB)
                self.last_frame.set(color)
            else:
                print("Invalid frame")

    def get_last_measurement(self) -> Measurement:
        return self.last_measurement.get()
        
    def get_last_frame(self) -> MatLike:
        return self.last_frame.get()
    
    def get_last_command(self) -> Command:
        return self.last_command
    
    def send_command(self, command: Command) -> None:
        self.last_command = deepcopy(command)
        match self.link.get():
            case Link.WIFI | Link.RADIO:
                self.crazyflie.commander.send_hover_setpoint(
                    command.velocity.x,
                    command.velocity.y,
                    np.rad2deg(command.yaw_rate),
                    command.altitude
                )
        
    def send_setpoint(self, setpoint: Setpoint) -> None:
        self.last_command = setpoint.equivalent_command(self.last_measurement.get())
        match self.link.get():
            case Link.WIFI | Link.RADIO:
                self.crazyflie.commander.send_position_setpoint(
                    setpoint.position.x,
                    setpoint.position.y,
                    setpoint.position.z,
                    np.rad2deg(setpoint.yaw)
                )

    def simulate(self, dt: float) -> None:

        if self.connected.get() and self.link.get() is Link.SIMULATION:

            m = self.last_measurement.get()
            v = self.last_command.velocity

            dz = self.last_command.altitude - m.position.z
            dxyz = glm.rotateZ(glm.vec3(v.x, v.y, dz), m.rotation.z)

            self.last_measurement.set(Measurement(
                m.timestamp + dt,
                m.position + dxyz * dt,
                glm.vec3(
                    -v.x * (np.pi / 4.0),
                    -v.y * (np.pi / 4.0),
                    wrap(m.rotation.z + self.last_command.yaw_rate * dt)
                ),
                np.clip(m.battery - 0.01 * glm.length(dxyz) * dt, 0.0, 1.0)
            ))
