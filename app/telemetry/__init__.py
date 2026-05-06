import os
import logging
import struct
from threading import Thread
import time
import numpy as np
from enum import Enum
from overrides import override
from pyglm import glm
from cv2.typing import MatLike
from app.inputs import Input
from cflib.cpx import CPXFunction
import cflib.crtp
from cflib.utils import uri_helper
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from app.telemetry.measurement import Measurement
from app.inputs.setpoint import Setpoint
from app.sync import Atomic
import cv2

class Telemetry(Thread):

    RADIO_URI = uri_helper.uri_from_env(default="radio://0/20/2M/E7E7E7E702")
    WIFI_URI = uri_helper.uri_from_env(default="tcp://192.168.4.1:5000")
    CAM_WIDTH = 324
    CAM_HEIGHT = 244
    PERIOD_MS = 20

    Z_RATE = (1.0 / 60.0)

    class State(Enum):
        DISCONNECTED    = 0
        CONNECTING      = 1
        CONNECTED       = 2

    class LinkType(Enum):
        RADIO = 0
        WIFI  = 1

    def __init__(self) -> None:
        super().__init__(name='Telemetry', daemon=True)
        logging.basicConfig(level=logging.ERROR)
        cflib.crtp.init_drivers()

        self.log_config = LogConfig("Telemetry", Telemetry.PERIOD_MS)
        self.log_config.add_variable("stateEstimate.x")
        self.log_config.add_variable("stateEstimate.y")
        self.log_config.add_variable("stateEstimate.z")
        self.log_config.add_variable("stabilizer.roll")
        self.log_config.add_variable("stabilizer.pitch")
        self.log_config.add_variable("stabilizer.yaw")
        self.log_config.add_variable("pm.batteryLevel")
        self.log_config.data_received_cb.add_callback(self.on_data_received)
        self.log_config.error_cb.add_callback(self.on_data_error)
        
        self.crazyflie = Crazyflie(ro_cache=None, rw_cache=os.path.join("cache"))
        self.crazyflie.connected.add_callback(self.on_connected)
        self.crazyflie.disconnected.add_callback(self.on_disconnected)
        self.crazyflie.connection_failed.add_callback(self.on_connection_failed)
        self.crazyflie.connection_lost.add_callback(self.on_connection_lost)

        self.state: Atomic[Telemetry.State] = Atomic(Telemetry.State.DISCONNECTED)
        self.measurement: Atomic[Measurement] = Atomic(Measurement())
        self.frame: Atomic[MatLike] = Atomic(np.zeros(shape=(244, 324, 3), dtype=np.uint8))
        self.link_type: Atomic[Telemetry.LinkType | None] = Atomic(None)

        self.z_acc = 1.0

        self.start()

    def connect(self, link_type: LinkType) -> None:
        try:
            self.state.set(Telemetry.State.CONNECTING)
            self.link_type.set(link_type)
            match link_type:
                case Telemetry.LinkType.RADIO:
                    self.crazyflie.open_link(Telemetry.RADIO_URI)
                case Telemetry.LinkType.WIFI:
                    self.z_acc = 1.0
                    self.crazyflie.open_link(Telemetry.WIFI_URI)
                    self.crazyflie.supervisor.send_arming_request(True)

        except:
            self.state.set(Telemetry.State.DISCONNECTED)
            self.link_type.set(None)

    @override
    def run(self) -> None:
        while True:
            if self.state.get() is not Telemetry.State.CONNECTED:
                time.sleep(0.001)
            elif self.link_type.get() is not Telemetry.LinkType.WIFI:
                time.sleep(0.001)
            else:
                try:
                    frame = self.receive_image()
                    self.frame.set(frame)
                except ValueError as e:
                    pass

    def receive_image(self) -> MatLike:
        p = self.crazyflie.link.cpx.receivePacket(CPXFunction.APP)
        [magic, width, height, depth, fmt, size] = struct.unpack('<BHHBBI', p.data[0:11])
        if magic == 0xBC:
            buf = bytearray()
            while len(buf) < size:
                buf.extend(self.crazyflie.link.cpx.receivePacket(CPXFunction.APP).data)
            img = np.frombuffer(buf, dtype=np.uint8)
            bayer = img.reshape((Telemetry.CAM_HEIGHT, Telemetry.CAM_WIDTH))
            color = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB)
            return color
        else:
            raise ValueError("Invalid frame")

    def get_last_measurement(self) -> Measurement:
        return self.measurement.get()
        
    def get_last_frame(self) -> MatLike:
        return self.frame.get()
    
    def set_input(self, input: Input) -> None:
        if self.state.get() == Telemetry.State.CONNECTED:
            self.z_acc += input.position.z * Telemetry.Z_RATE
            self.z_acc = np.clip(self.z_acc, 0.0, 3.0)
            self.crazyflie.commander.send_hover_setpoint(
                input.position.x,
                input.position.y,
                np.rad2deg(input.yaw),
                self.z_acc
            )
        
    def set_setpoint(self, setpoint: Setpoint) -> None:
        if self.state.get() == Telemetry.State.CONNECTED:
            self.crazyflie.commander.send_position_setpoint(
                setpoint.position.x,
                setpoint.position.y,
                setpoint.position.z,
                np.rad2deg(setpoint.yaw)
            )

    def disconnect(self) -> None:
        self.crazyflie.commander.send_stop_setpoint()
        self.crazyflie.close_link()

    def on_connected(self, link_uri):
        print(f"Connected to {link_uri}")
        self.state.set(Telemetry.State.CONNECTED)
        try:
            self.crazyflie.log.add_config(self.log_config)
            self.log_config.start()
        except KeyError as e:
            print(f"Could not start log configuration, '{e}' not found in TOC")
        except AttributeError:
            print("Could not add Stabilizer log config, bad configuration")

    def on_data_received(self, timestamp, data, logconf):

        # Store new measurement
        try:
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

            # Drain queue and push new measurement
            self.measurement.set(Measurement(timestamp, position, rotation, battery))

        # Print missing field and available fields
        except KeyError as e:
            print(f"'{e}' key not found in received data, available fields in [{timestamp}][{logconf.name}] are : ", end="")
            for name, value in data.items():
                print(f"{name}: {float(value):3.3f}", end="")
            print()

    def on_data_error(self, logconf, msg):
        print(f"Error when logging '{logconf.name}': '{msg}'")

    def on_connection_failed(self, link_uri, msg):
        print(f"Connection to '{link_uri}' failed: '{msg}'")
        self.state.set(Telemetry.State.DISCONNECTED)
        self.link_type.set(None)

    def on_connection_lost(self, link_uri, msg):
        print(f"Connection to '{link_uri}' lost: '{msg}'")

    def on_disconnected(self, link_uri):
        print(f"Disconnected from '{link_uri}'")
        self.state.set(Telemetry.State.DISCONNECTED)
        self.link_type.set(None)
