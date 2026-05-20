import numpy as np

# Z-shaping parameters for RacerPlanner pre/post gate waypoints.
# Both app/planner/racer.py and controller_speed/visulazitaion_v2.0.py import from here
# so tuning one constant updates both the drone trajectory and the visualization.

Z_BIAS_MAX    = 0.10    # m  — hard cap on z shift from gate center
Z_BIAS_BASE   = 0.01    # m  — minimum shift when slope exceeds deadband
Z_BIAS_EXTRA  = 0.05    # m  — additional shift scaled by slope boost
Z_SLOPE_START = 0.10    # dz/dxy — slope at which boost begins
Z_SLOPE_FULL  = 0.40    # dz/dxy — slope at which boost is maximal
Z_DEADBAND    = 0.05    # m  — height difference below which no shift is applied


def compute_z_shift(prev_xyz, next_xyz) -> float:
    """
    Slope-proportional z bias for the pre/post waypoints around a gate.

    Parameters
    ----------
    prev_xyz : array-like [x, y, z]  — previous gate center (or home)
    next_xyz : array-like [x, y, z]  — next gate center (or home)

    Returns
    -------
    z_shift : float
        Positive means next gate is higher than prev (drone climbs through).
        Apply as:  pre_z  = gate_z - 0.5 * z_shift
                   post_z = gate_z + 0.5 * z_shift
    """
    prev  = np.asarray(prev_xyz, dtype=float)
    next_ = np.asarray(next_xyz, dtype=float)

    delta  = next_ - prev
    xy_len = float(np.linalg.norm(delta[:2]))
    dz     = float(delta[2])

    if abs(dz) < Z_DEADBAND:
        return 0.0

    slope = abs(dz) / max(xy_len, 1e-6)
    boost = float(np.clip(
        (slope - Z_SLOPE_START) / (Z_SLOPE_FULL - Z_SLOPE_START),
        0.0, 1.0
    ))
    magnitude = Z_BIAS_BASE + Z_BIAS_EXTRA * boost
    return float(np.sign(dz)) * min(magnitude, Z_BIAS_MAX)
