import os
import logging
import numpy as np
from enum import Enum
from pyglm import glm
import cflib.crtp
from cflib.utils import uri_helper
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from app import wrap
from app.telemetry.measurement import Measurement
from app.inputs import Input
from app.telemetry.setpoint import Setpoint
from app.sync import Atomic

class Telemetry:

    URI = uri_helper.uri_from_env(default="radio://0/20/2M/E7E7E7E702")
    PERIOD_MS = 20

    class State(Enum):
        DISCONNECTED    = 0
        CONNECTING      = 1
        CONNECTED       = 2

    def __init__(self) -> None:
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
        
        self.crazyflie = Crazyflie(rw_cache=os.path.join("cache"))
        self.crazyflie.connected.add_callback(self.on_connected)
        self.crazyflie.disconnected.add_callback(self.on_disconnected)
        self.crazyflie.connection_failed.add_callback(self.on_connection_failed)
        self.crazyflie.connection_lost.add_callback(self.on_connection_lost)

        self.state: Atomic[Telemetry.State] = Atomic(Telemetry.State.DISCONNECTED)
        self.measurement: Atomic[Measurement] = Atomic(Measurement())

    def connect(self) -> None:
        try:
            self.state.set(Telemetry.State.CONNECTING)
            self.crazyflie.open_link(Telemetry.URI)
        except:
            self.state.set(Telemetry.State.DISCONNECTED)

    def simulate(self, manual_input: Input):
        m = self.measurement.get()
        m.rotation.x = -manual_input.position.y * np.pi / 2.0                               # Infer from forward displacement command
        m.rotation.y = -manual_input.position.x * 250                                       # Infer from lateral displacement command
        m.rotation.z += (1 / 60) * manual_input.yaw                                         # Integrate yaw command
        m.rotation.z = wrap(m.rotation.z)                                                   # Wrap yaw around -pi to pi
        xy = manual_input.position.xy                                                       # Isolate xy components
        m.position.xy += (1 / 60) * glm.rotateZ(glm.vec3(xy.x, xy.y, 0.0), m.rotation.z).xy # Integrate xy displacement with respect to yaw
        m.position.z += (1 / 60) * manual_input.position.z                                  # Integrate z displacement
        m.position.z = np.max([0.0, m.position.z])                                          # Bound z below
        m.battery = np.clip((5.0 - glm.length(m.position.xy)) / 5.0, 0.0, 1.0)              # Make battery decrease with lateral distance (max 20 m)
        self.measurement.set(m)

    def get_last_measurement(self) -> Measurement:
        return self.measurement.get()
        
    def set_setpoint(self, setpoint: Setpoint) -> None:
        if self.state.get() == Telemetry.State.CONNECTED:
            self.crazyflie.commander.send_position_setpoint(
                setpoint.position.x,
                setpoint.position.y,
                setpoint.position.z,
                np.rad2deg(setpoint.yaw)
            )

    def disconnect(self) -> None:
        self.crazyflie.close_link()
        self.state.set(Telemetry.State.DISCONNECTED)

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

    def on_connection_lost(self, link_uri, msg):
        print(f"Connection to '{link_uri}' lost: '{msg}'")

    def on_disconnected(self, link_uri):
        print(f"Disconnected from '{link_uri}'")
        self.state.set(Telemetry.State.DISCONNECTED)
