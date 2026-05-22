import numpy as np

# Z-shaping parameters for RacerPlanner pre/post gate waypoints.
# Both app/planner/racer.py and controller_speed/visulazitaion_v2.0.py import from here
# so tuning one constant updates both the drone trajectory and the visualization.

Z_BIAS_MAX    = 0.30    # m  — hard cap on z shift from gate center
Z_BIAS_BASE   = 0.01    # m  — minimum shift when slope exceeds deadband
Z_BIAS_EXTRA  = 0.10    # m  — additional shift scaled by slope boost
#Less aggressive
#Z_BIAS_EXTRA  = 0.05
#More aggressive
#Z_BIAS_EXTRA  = 0.20
Z_SLOPE_START = 0.10    # dz/dxy — slope at which boost begins
Z_SLOPE_FULL  = 0.20    # dz/dxy — slope at which boost is maximal
Z_DEADBAND    = 0.03    # m  — height difference below which no shift is applied


def compute_z_shift(from_xyz, to_xyz) -> float:
    """
    Slope-proportional z shift for one gate transition segment.

    Call twice per gate — asymmetrically:
        pre_shift  = compute_z_shift(prev_gate, current_gate)
        post_shift = compute_z_shift(current_gate, next_gate)

        pre_z  = gate_z - 0.5 * pre_shift
        post_z = gate_z + 0.5 * post_shift

    This way pre reflects the actual incoming slope and post reflects
    the actual outgoing slope independently, allowing asymmetric shaping.
    """
    from_ = np.asarray(from_xyz, dtype=float)
    to_   = np.asarray(to_xyz,   dtype=float)

    delta  = to_ - from_
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
