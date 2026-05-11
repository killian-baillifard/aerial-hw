"""
Compute gate center poses from corner measurement recordings.

measures.npy layout (Measurement.to_array() format):
  shape (N, 8)  —  N = n_gates * 4
  columns: timestamp, x, y, z, roll, pitch, yaw, battery

Corners are stored in groups of 4 per gate, clockwise from top-left:
  row 0: TL, row 1: TR, row 2: BR, row 3: BL
"""

import numpy as np

_X, _Y, _Z, _YAW = 1, 2, 3, 6  # column indices in Measurement.to_array()


def load_gate_poses(npy_path: str) -> np.ndarray:
    """
    Parameters
    ----------
    npy_path : str
        Path to measures.npy recorded by the corner-measurement flight.

    Returns
    -------
    gate_poses : np.ndarray, shape (n_gates, 5)
        Each row is [x, y, z, yaw, measured_width] for one gate center.
        yaw is the gate-normal direction in the x-y plane (approach angle),
        in radians, range [-pi, pi].
        The sign convention matches the drone facing direction during measurement.
        measured_width is the gate opening span in metres (used for size classification).
    """
    data = np.load(npy_path)          # (n_gates*4, 8)
    n_gates = len(data) // 4
    gate_poses = np.zeros((n_gates, 5))

    for i in range(n_gates):
        rows = data[i * 4 : (i + 1) * 4]   # TL, TR, BR, BL

        corners_xyz = rows[:, [_X, _Y, _Z]]
        drone_yaws  = rows[:, _YAW]

        # --- center ---
        center = corners_xyz.mean(axis=0)   # (3,)

        # --- gate yaw + width via PCA in x-y ---
        # The four corners project onto the gate's horizontal axis in x-y.
        # First principal component = that horizontal axis; normal is perpendicular.
        xy = corners_xyz[:, :2]
        xy_c = xy - xy.mean(axis=0)
        _, evecs = np.linalg.eigh(xy_c.T @ xy_c)
        main_axis = evecs[:, -1]                            # gate horizontal direction
        normal = np.array([-main_axis[1], main_axis[0]])   # 90° CCW rotation

        # Align sign: drone faced the gate during measurement, so normal ≈ drone heading
        avg_yaw = np.mean(drone_yaws)
        drone_dir = np.array([np.cos(avg_yaw), np.sin(avg_yaw)])
        if np.dot(normal, drone_dir) < 0:
            normal = -normal

        # Width = full span of corners projected onto the horizontal axis
        projections = xy_c @ main_axis
        measured_width = float(projections.max() - projections.min())

        gate_poses[i] = [center[0], center[1], center[2],
                         np.arctan2(normal[1], normal[0]),
                         measured_width]

    return gate_poses


if __name__ == "__main__":
    import sys
    import os

    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..",
        "saved_captures/2026-05-06-17-32-33_corner_measurements/measures.npy"
    )

    poses = load_gate_poses(path)

    # Rank by measured width to show size classification
    order = np.argsort(poses[:, 4])[::-1]
    size_labels = ["50x40 cm"] + ["40x40 cm"] * 2 + ["29x40 cm"] * 2
    label_by_idx = {int(order[r]): size_labels[r] for r in range(len(order))}

    print("Gate poses  [x, y, z, yaw_rad, measured_width_m]")
    print("-" * 68)
    for i, (x, y, z, yaw, w) in enumerate(poses):
        label = label_by_idx[i]
        print(f"  Gate {i+1} ({label}):  "
              f"x={x:+.3f}  y={y:+.3f}  z={z:.3f}  "
              f"yaw={yaw:+.3f} ({np.degrees(yaw):+.1f}°)  "
              f"w_meas={w:.3f} m")

    print()
    print("NumPy array (cols: x, y, z, yaw, measured_width):")
    print(poses)
