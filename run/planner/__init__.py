"""
Abstract Planner base class.

All concrete planners must subclass Planner and implement:
    reload()  - reset to initial state
    update()  - run one tick, return a Setpoint

Class-level constants can be overridden per planner:
    LAND_HEIGHT, TOLERANCE, MIN_HEIGHT, HOME_POSITION, HOME_YAW, …
"""

import numpy as np
from pyglm import glm
from abc import ABC, abstractmethod
from cv2.typing import MatLike

from run import wrap, Flags
from run.generics import Event
from run.telemetry import Measurement, Setpoint


class Planner(ABC):

    # ------------------------------------------------------------------ tuning
    HOME_POSITION   = glm.vec3(0.0, 0.0, 1.0)  # m
    HOME_YAW        = 0.0                        # rad
    HOME_SETPOINT   = Setpoint(HOME_POSITION, HOME_YAW)

    APPROACH_DIST   = 0.20  # m  – stop this far from gate centre
    POS_TOL         = 0.10  # m  – position reached tolerance
    YAW_TOL         = np.pi / 10  # rad

    LAND_HEIGHT     = 0.10  # m  – target z when landing
    TOLERANCE       = 0.15  # m  – general-purpose proximity tolerance
    MIN_HEIGHT      = 0.05  # m  – floor: never command below this

    # ------------------------------------------------------------------ init
    def __init__(self) -> None:
        self.waypoints:             list[Setpoint]      = []
        self.gates:                 list[Setpoint]      = []
        self.gate_found_event:      Event[Setpoint]     = Event[Setpoint]()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def reach(
        setpoint:    Setpoint,
        measurement: Measurement,
        speed:       float = 1.0,
    ) -> tuple[Setpoint, bool]:
        """
        Interpolate a new setpoint between current measurement and target.

        Parameters
        ----------
        setpoint    : target position and yaw
        measurement : last measured position and yaw
        speed       : max advance step per call (metres)

        Returns
        -------
        interpolated : intermediate setpoint for the PID
        reached      : True when position AND yaw are within tolerance
        """
        error:   glm.vec3 = setpoint.position - measurement.position
        dist_xy: float    = glm.length(error.xy)

        target_heading: float = np.atan2(error.y, error.x)
        heading_error:  float = abs(wrap(target_heading - measurement.rotation.z))

        # 1. Align heading before advancing when far away
        if dist_xy > 1.0 and heading_error > Planner.YAW_TOL:
            return Setpoint(measurement.position, target_heading), False

        # 2. Advance toward target
        direction   = speed * glm.normalize(error.xy) if dist_xy > speed else error.xy
        position    = glm.vec3(measurement.position.xy + direction, setpoint.position.z)
        loc_reached = dist_xy < Planner.POS_TOL
        if not loc_reached:
            return Setpoint(position, target_heading), False

        # 3. Correct yaw once on target
        pos_reached = glm.length(error) < Planner.POS_TOL
        yaw_reached = abs(wrap(measurement.rotation.z - setpoint.yaw)) < Planner.YAW_TOL
        return setpoint, pos_reached and yaw_reached

    def safe_z(self, z: float) -> float:
        """Clamp a z command so it never goes below MIN_HEIGHT."""
        return max(z, self.MIN_HEIGHT)

    # ------------------------------------------------------------------ interface

    @abstractmethod
    def update(
        self,
        measurement: Measurement,
        frame:       MatLike,
        flags:       Flags,
        dt:          float,
    ) -> Setpoint:
        """
        Run one tick of the planner state machine.

        Parameters
        ----------
        measurement : Measurement
            timestamp, position (x,y,z), rotation (roll,pitch,yaw), battery
        frame : MatLike
            BGR image (244 × 324 × 3), or None when video is disabled
        flags : Telemetry.Flags
            NEW_MEASUREMENT and/or NEW_FRAME bits
        dt : float
            seconds since last call

        Returns
        -------
        Setpoint  - never return None; use the last valid setpoint as fallback
        """
