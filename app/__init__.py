import numpy as np
from pyglm import glm
from app.inputs import Input
from app.inputs.setpoint import Setpoint
from app.telemetry.measurement import Measurement

def wrap(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def input_to_setpoint(input: Input, measurement: Measurement) -> Setpoint:
    return Setpoint(
        measurement.position + glm.rotateZ(input.position, measurement.rotation.z),
        input.yaw + measurement.rotation.z
    )

def setpoint_to_input(setpoint: Setpoint, measurement: Measurement) -> Input:
    return Input(
        glm.rotateZ(setpoint.position - measurement.position, -measurement.rotation.z),
        setpoint.yaw - measurement.rotation.z
    )
