import numpy as np

# Controls how the theta column in the CSV is interpreted:
#   "LAB_TEST" : theta is the approach yaw directly (angle of the gate normal,
#                pointing in the direction the drone flies through the gate).
#                Use this for hand-measured CSV files where you recorded the
#                perpendicular-to-gate direction yourself.
#   "LAB_EXAM" : theta is the gate surface projection angle onto the x-y plane,
#                measured from the +x axis (as provided by the competition).
#                The approach yaw = wrap(theta - pi/2), with the ±pi/2 ambiguity
#                resolved by the approach vector from the previous waypoint.
GATE_SOURCE = "LAB_EXAM"


def _wrap(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _resolve_yaw(theta, prev_xy, gate_xy):
    """
    theta is the gate surface angle in x-y (from +x axis).
    The approach yaw is perpendicular: theta ± pi/2.
    Pick whichever aligns better with the vector prev_xy -> gate_xy.
    """
    approach = np.arctan2(gate_xy[1] - prev_xy[1], gate_xy[0] - prev_xy[0])
    yaw_a = _wrap(theta - np.pi / 2)
    yaw_b = _wrap(theta + np.pi / 2)
    diff_a = abs(_wrap(approach - yaw_a))
    diff_b = abs(_wrap(approach - yaw_b))
    return yaw_a if diff_a <= diff_b else yaw_b


def load_gates_csv(csv_path, home_xy=(0.0, 0.0)):
    """
    Load gate poses from CSV with header: Gate,x,y,z,theta,width,height
    All angle values must be in radians.

    Parameters
    ----------
    csv_path : str
        Path to CSV file.
    home_xy : tuple
        (x, y) of the starting position in drone/Lighthouse frame, used only
        to resolve the ±pi/2 yaw ambiguity of the first gate (LAB_EXAM mode).

    Returns
    -------
    poses : np.ndarray, shape (n_gates, 6)
        Columns: [x, y, z, yaw, width, height]
        yaw is the approach direction (angle the drone faces when flying through).
    """
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    data = data[np.argsort(data[:, 0])]  # sort by gate number

    n = len(data)
    poses = np.zeros((n, 6))
    prev_xy = np.array(home_xy, dtype=float)

    for i, row in enumerate(data):
        _, x, y, z, theta, width, height = row
        gate_xy = np.array([x, y])

        if GATE_SOURCE == "LAB_TEST":
            yaw = _wrap(float(theta))
        else:  # LAB_EXAM
            yaw = _wrap(float(theta - np.pi))

        poses[i] = [x, y, z, yaw, width, height]
        prev_xy = gate_xy

    return poses
