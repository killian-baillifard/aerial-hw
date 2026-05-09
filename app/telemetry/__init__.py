import os, logging, struct, time
import numpy as np
import cv2
from typing import Callable
from cv2.typing import MatLike
from threading import Thread
from overrides import override
from pyglm import glm
import cflib.crtp
from cflib.cpx import CPXFunction
from cflib.utils import uri_helper
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from app import Link
from app.generics import Atomic, Mailbox, Event
from app.io import Command, Setpoint
from app.io import Measurement
from app.telemetry.camera import WIDTH, HEIGHT
from enum import Flag

class TelemetryFlags(Flag):
    NEITHER         = 0
    NEW_MEASUREMENT = 1
    NEW_FRAME       = 2
    BOTH            = NEW_MEASUREMENT | NEW_FRAME

class Telemetry(Thread):

    RADIO_URI       = uri_helper.uri_from_env(default="radio://0/20/2M/E7E7E7E702")
    WIFI_URI        = uri_helper.uri_from_env(default="tcp://192.168.4.1:5000")
    UPDATE_PERIOD   = 20    # ms
    TKOF_HEIGHT     = 1.0   # m
    LAND_HEIGHT     = 0.1   # m
    TOLERANCE       = 0.05  # m

    def __init__(self, sim_overlay_func: Callable[[MatLike], None]) -> None:
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
        self.sim_overlay_func = sim_overlay_func
        self.link: Atomic[Link | None] = Atomic(None)
        self.connected: Atomic[bool] = Atomic(False)
        self.measurement: Mailbox[Measurement] = Mailbox(Measurement())
        self.frame: Mailbox[MatLike] = Mailbox(np.zeros(shape=(HEIGHT, WIDTH, 3), dtype=np.uint8))
        self.command: Command = Command()
        self.z: float = 0.0

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
        self.measurement.set(Measurement(timestamp, position, rotation, battery))

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
                Thread(target=self.crazyflie.open_link, args=(Telemetry.RADIO_URI, ), daemon=True).start()
            case Link.WIFI:
                Thread(target=self.crazyflie.open_link, args=(Telemetry.WIFI_URI, ), daemon=True).start()

    def disconnect(self) -> None:
        match self.link.get():
            case Link.SIMULATION:
                self.connected.set(False)
                self.disconnected_event()
            case Link.RADIO | Link.WIFI:
                self.crazyflie.supervisor.send_arming_request(False)
                self.crazyflie.close_link()

    def simulate_crazyflie(self, dt: float) -> None:

        # Step simulation
        measurement = self.measurement.read()
        measurement = measurement.simulate(self.command, dt)

        # Overlay simulation on new frame
        frame = np.zeros(shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.sim_overlay_func(frame)

        # Set new frame and measurement
        self.measurement.set(measurement)
        self.frame.set(frame)

    def tkof(self, dt: float) -> bool:
        
        dz = Telemetry.TKOF_HEIGHT - self.measurement.read().position.z
        airborn = np.abs(dz) < Telemetry.TOLERANCE
        self.command = Command(glm.vec3(0.0, 0.0, np.clip(dz, -1.0, 1.0)))
        if not airborn:
            match self.link.get():
                case Link.SIMULATION:
                        self.simulate_crazyflie(dt)
                case Link.WIFI | Link.RADIO:
                        self.crazyflie.commander.send_zdistance_setpoint(0.0, 0.0, 0.0, Telemetry.TKOF_HEIGHT)
        
        return airborn

    def land(self, dt: float) -> bool:

        dz = Telemetry.LAND_HEIGHT - self.measurement.read().position.z
        landed = np.abs(dz) < Telemetry.TOLERANCE
        self.command = Command(glm.vec3(0.0, 0.0, np.clip(dz, -1.0, 1.0)))
        match self.link.get():
            case Link.SIMULATION:
                if not landed:
                    self.simulate_crazyflie(dt)
                else:
                    self.measurement.set(Measurement())
            case Link.WIFI | Link.RADIO:
                if not landed:
                    self.crazyflie.commander.send_zdistance_setpoint(0.0, 0.0, 0.0, Telemetry.LAND_HEIGHT)
                else:
                    self.crazyflie.commander.send_stop_setpoint()

        return landed
    
    def send_command(self, command: Command, dt: float) -> None:
        self.command = command
        if self.connected.get():
            match self.link.get():
                case Link.SIMULATION:
                    self.simulate_crazyflie(dt)
                case Link.WIFI | Link.RADIO:
                    self.crazyflie.commander.send_hover_setpoint(
                        command.velocity.x,
                        command.velocity.y,
                        np.rad2deg(command.yaw_rate),
                        self.z + self.command.velocity.z * dt
                    )
        
    def send_setpoint(self, setpoint: Setpoint, dt: float) -> None:
        self.command = setpoint.to_command(self.measurement.read())
        if self.connected.get():
            match self.link.get():
                case Link.SIMULATION:
                    self.simulate_crazyflie(dt)
                case Link.WIFI | Link.RADIO:
                    self.crazyflie.commander.send_position_setpoint(
                        setpoint.position.x,
                        setpoint.position.y,
                        setpoint.position.z,
                        np.rad2deg(setpoint.yaw)
                    )

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
                bayer = img.reshape(shape=(HEIGHT, WIDTH))
                color = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB)
                self.frame.set(color)
            else:
                print("Invalid frame")
