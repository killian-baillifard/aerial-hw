import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from gate_poses import load_gate_poses

# =========================
# Real arena dimensions [m]
# =========================
ROOM_X = 2.87          # sheet vertical direction, world X
ROOM_Y = 2.95          # sheet horizontal direction, world Y
ROOM_Z = 2.40          # useful plotting height

CENTER   = np.array([ROOM_X / 2, ROOM_Y / 2], dtype=float)
HOME_Z   = 0.10

# =========================
# Sector / gate settings
# =========================
# 12 sectors of 30 deg. Home = 3 sectors = 90 deg.
HOME_HALF_ANGLE_DEG  = 45.0

FRAME_WIDTH = 0.08      # gate frame width [m]
GATE_HEIGHT = 0.40      # inner opening height [m] (same for all gates)
WALL_CLEARANCE = 0.04

APPROACH_DIST = 0.20    # [m] waypoint offset from gate center on each side

# Position of drone (0,0,0) in room world coordinates [m].
# From sketch:  X = ROOM_X − 1.64       (164 cm from top wall → 1.23 m from bottom)
#               Y = 2.01 + 0.165         (201 cm from left wall + 16.5 cm Lighthouse offset)
DRONE_ORIGIN_XY = np.array([ROOM_X - 1.64, 2.01 + 0.165])

# Home pad: drone HOME_SETPOINT = (−1.0, 0.0, 1.0) in drone frame → 100 cm behind origin
HOME_XY = np.array([DRONE_ORIGIN_XY[0] - 1.0, DRONE_ORIGIN_XY[1]], dtype=float)

# =========================
# Gate sizes (competition standard)
# Size assignment is determined by ranking measured widths: widest → 50 cm, etc.
# =========================
GATE_NOMINAL_SIZES = {
    "50x40 cm": (0.50, 0.40),
    "40x40 cm": (0.40, 0.40),
    "29x40 cm": (0.29, 0.40),
}

MEASUREMENTS_NPY = os.path.join(
    os.path.dirname(__file__), "..",
    "saved_captures/2026-05-06-17-32-33_corner_measurements/measures.npy"
)


@dataclass
class Gate:
    idx: int
    size_label: str
    width: float          # inner width [m]  (nominal)
    height: float         # inner height [m] (nominal)
    frame: float          # frame width [m]
    center: np.ndarray    # [x, y, z]
    yaw: float            # approach yaw through gate [rad]
    bar_yaw: float        # gate bar direction, perpendicular to yaw [rad]

    @property
    def outer_width(self):
        return self.width + 2 * self.frame

    @property
    def outer_height(self):
        return self.height + 2 * self.frame


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def uvec(a):
    return np.array([np.cos(a), np.sin(a)], dtype=float)


def deg(a):
    return float(np.rad2deg(wrap(a)))


def sheet_xy(p):
    """World [X,Y] -> sheet plot [Y,X]."""
    p = np.asarray(p, dtype=float)
    return np.array([p[1], p[0]])


def sheet_xyz(p):
    """World [X,Y,Z] -> sheet plot [Y,X,Z]."""
    p = np.asarray(p, dtype=float)
    return np.array([p[1], p[0], p[2]])


def sheet_vec_xy(v):
    v = np.asarray(v, dtype=float)
    return np.array([v[1], v[0]])


def ray_to_wall(origin, direction):
    ox, oy = origin
    dx, dy = direction
    hits = []

    if abs(dx) > 1e-12:
        for xb in [0.0, ROOM_X]:
            t = (xb - ox) / dx
            y = oy + t * dy
            if t > 0 and 0 <= y <= ROOM_Y:
                hits.append((t, np.array([xb, y])))

    if abs(dy) > 1e-12:
        for yb in [0.0, ROOM_Y]:
            t = (yb - oy) / dy
            x = ox + t * dx
            if t > 0 and 0 <= x <= ROOM_X:
                hits.append((t, np.array([x, yb])))

    return min(hits, key=lambda h: h[0])[1] if hits else origin.copy()


def poses_to_gates(poses: np.ndarray) -> list:
    """
    Convert gate_poses array (n_gates, 5) to a list of Gate objects.

    Gates are ranked by their measured width (column 4) to assign size labels:
      rank 1 (widest) → "50x40 cm"
      ranks 2–3       → "40x40 cm"
      ranks 4–5       → "29x40 cm"

    The nominal width/height from GATE_NOMINAL_SIZES is used for rendering.
    """
    n = len(poses)
    order = np.argsort(poses[:, 4])[::-1]           # widest first
    size_seq = ["50x40 cm"] + ["40x40 cm"] * 2 + ["29x40 cm"] * 2
    label_by_idx = {int(order[r]): size_seq[r] for r in range(n)}

    gates = []
    for i, (x, y, z, yaw, _) in enumerate(poses):
        label = label_by_idx[i]
        nom_w, nom_h = GATE_NOMINAL_SIZES[label]
        # Convert drone coords → room coords by adding the drone origin offset
        x_room = x + DRONE_ORIGIN_XY[0]
        y_room = y + DRONE_ORIGIN_XY[1]
        gates.append(Gate(
            idx=i + 1,
            size_label=label,
            width=nom_w,
            height=nom_h,
            frame=FRAME_WIDTH,
            center=np.array([x_room, y_room, z]),
            yaw=float(yaw),
            bar_yaw=float(wrap(yaw + np.pi / 2)),
        ))
    return gates


def build_trajectory_room_coords(poses: np.ndarray) -> np.ndarray:
    """
    Compute the planned trajectory waypoints in room coordinates.

    For each gate: approach point (APPROACH_DIST before gate) then exit point
    (APPROACH_DIST after gate), matching the logic in SpeedController._build_trajectory().

    Returns (n_gates * 2, 3) array of [x_room, y_room, z].
    """
    wps = [[HOME_XY[0], HOME_XY[1], 0.5]]   # start at home
    for _ in range(2):
        for x, y, z, yaw, _ in poses:
            dx, dy = np.cos(yaw), np.sin(yaw)
            wps.append([x - APPROACH_DIST * dx + DRONE_ORIGIN_XY[0],
                        y - APPROACH_DIST * dy + DRONE_ORIGIN_XY[1], z])
            wps.append([x + APPROACH_DIST * dx + DRONE_ORIGIN_XY[0],
                        y + APPROACH_DIST * dy + DRONE_ORIGIN_XY[1], z])
    # Final waypoint: return home at flying height
    wps.append([HOME_XY[0], HOME_XY[1], 0.5])
    return np.array(wps)


def draw_trajectory_2d(ax, wps: np.ndarray):
    s = np.array([sheet_xy(wp[:2]) for wp in wps])
    ax.plot(s[:, 0], s[:, 1], '--', color='#FF69B4', lw=1.8, zorder=3,
            label='Planned trajectory')
    ax.scatter(s[:, 0], s[:, 1], color='#FF69B4', s=30, zorder=4)


def draw_trajectory_3d(ax, wps: np.ndarray):
    s = np.array([sheet_xyz(wp) for wp in wps])
    ax.plot(s[:, 0], s[:, 1], s[:, 2], '--', color='#FF69B4', lw=1.8,
            label='Planned trajectory')
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], color='#FF69B4', s=20)


def gate_bar_polys(g):
    c = g.center
    axis = np.array([np.cos(g.bar_yaw), np.sin(g.bar_yaw), 0.0])
    zaxis = np.array([0.0, 0.0, 1.0])

    iw, ih = g.width, g.height
    ow, oh = g.outer_width, g.outer_height

    def p(u, z):
        return c + u * axis + z * zaxis

    specs = [
        (-ow / 2, ow / 2,  ih / 2, oh / 2),      # top bar
        (-ow / 2, ow / 2, -oh / 2, -ih / 2),     # bottom bar
        (-ow / 2, -iw / 2, -ih / 2, ih / 2),     # left bar
        ( iw / 2,  ow / 2, -ih / 2, ih / 2),     # right bar
    ]

    return [
        np.array([p(u0, z0), p(u1, z0), p(u1, z1), p(u0, z1)])
        for u0, u1, z0, z1 in specs
    ]


def draw_room_2d(ax):
    room = np.array([
        [0, 0], [ROOM_X, 0], [ROOM_X, ROOM_Y], [0, ROOM_Y], [0, 0],
    ], dtype=float)
    s = np.array([sheet_xy(p) for p in room])
    ax.plot(s[:, 0], s[:, 1], color="black", lw=2.2)


def draw_room_3d(ax):
    pts = np.array([
        [0, 0, 0],      [ROOM_X, 0, 0],      [ROOM_X, ROOM_Y, 0],      [0, ROOM_Y, 0],
        [0, 0, ROOM_Z], [ROOM_X, 0, ROOM_Z], [ROOM_X, ROOM_Y, ROOM_Z], [0, ROOM_Y, ROOM_Z],
    ], dtype=float)
    s = np.array([sheet_xyz(p) for p in pts])
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    for a, b in edges:
        ax.plot([s[a,0], s[b,0]], [s[a,1], s[b,1]], [s[a,2], s[b,2]],
                color="black", lw=1, alpha=0.35)


def draw_sector_guides_2d(ax):
    c = sheet_xy(CENTER)
    home_angle = np.arctan2(HOME_XY[1] - CENTER[1], HOME_XY[0] - CENTER[0])
    home_start = home_angle - np.deg2rad(HOME_HALF_ANGLE_DEG)
    home_end   = home_angle + np.deg2rad(HOME_HALF_ANGLE_DEG)

    # Home zone (blue, 90°)
    angles_home = np.linspace(home_start, home_end, 30)
    wedge = [CENTER] + [ray_to_wall(CENTER, uvec(a)) for a in angles_home] + [CENTER]
    ws = np.array([sheet_xy(p) for p in wedge])
    ax.fill(ws[:, 0], ws[:, 1], color="#8CBDF2", alpha=0.35, zorder=1, label="Home zone (90°)")

    # 9 alternating zones after home: gate (even k=0,2,4,6,8) and no-gate (odd k=1,3,5,7)
    gate_num = 1
    for k in range(9):
        zone_start = home_end + k * np.deg2rad(30)
        zone_end   = zone_start + np.deg2rad(30)
        zone_mid   = (zone_start + zone_end) / 2
        is_gate    = (k % 2 == 0)

        angles_zone = np.linspace(zone_start, zone_end, 10)
        wedge = [CENTER] + [ray_to_wall(CENTER, uvec(a)) for a in angles_zone] + [CENTER]
        ws = np.array([sheet_xy(p) for p in wedge])

        if is_gate:
            # Gate zones: white (no fill) — label at outer wall edge
            wall_pt = ray_to_wall(CENTER, uvec(zone_mid))
            text_pos = CENTER + 0.88 * (wall_pt - CENTER)
            tps = sheet_xy(text_pos)
            ax.text(tps[0], tps[1], f"gate {gate_num}",
                    color="0.35", fontsize=8, ha="center", va="center", zorder=3)
            gate_num += 1
        else:
            # No-gate zones: light grey fill
            ax.fill(ws[:, 0], ws[:, 1], color="#C8C8C8", alpha=0.45, zorder=1)

    # Spoke lines aligned to home boundary, every 30°
    for k in range(12):
        angle = home_end + k * np.deg2rad(30)
        end = ray_to_wall(CENTER, uvec(angle))
        e = sheet_xy(end)
        ax.plot([c[0], e[0]], [c[1], e[1]], color=(0.78, 0.78, 0.95), lw=1.2, zorder=2)


def draw_sector_guides_3d(ax):
    c = sheet_xyz([CENTER[0], CENTER[1], 0])
    home_angle = np.arctan2(HOME_XY[1] - CENTER[1], HOME_XY[0] - CENTER[0])
    home_start = home_angle - np.deg2rad(HOME_HALF_ANGLE_DEG)
    home_end   = home_angle + np.deg2rad(HOME_HALF_ANGLE_DEG)

    # Home zone (blue)
    angles_home = np.linspace(home_start, home_end, 30)
    wedge = [[CENTER[0], CENTER[1], 0]]
    wedge += [[*ray_to_wall(CENTER, uvec(a)), 0] for a in angles_home]
    wedge += [[CENTER[0], CENTER[1], 0]]
    ws = np.array([sheet_xyz(p) for p in wedge])
    ax.add_collection3d(Poly3DCollection([ws], facecolor="#8CBDF2", edgecolor="none", alpha=0.25))

    # No-gate zones: light grey
    for k in range(9):
        if k % 2 != 0:
            zone_start = home_end + k * np.deg2rad(30)
            zone_end   = zone_start + np.deg2rad(30)
            angles_zone = np.linspace(zone_start, zone_end, 10)
            wdg = [[CENTER[0], CENTER[1], 0]]
            wdg += [[*ray_to_wall(CENTER, uvec(a)), 0] for a in angles_zone]
            wdg += [[CENTER[0], CENTER[1], 0]]
            ws_z = np.array([sheet_xyz(p) for p in wdg])
            ax.add_collection3d(Poly3DCollection([ws_z], facecolor="#C8C8C8", edgecolor="none", alpha=0.25))

    # Spokes aligned to home boundary, every 30°
    for k in range(12):
        angle = home_end + k * np.deg2rad(30)
        end = ray_to_wall(CENTER, uvec(angle))
        e = sheet_xyz([end[0], end[1], 0])
        ax.plot([c[0], e[0]], [c[1], e[1]], [0, 0], color=(0.78, 0.78, 0.95), lw=1.2)


def draw_gate_3d(ax, g):
    polys = [np.array([sheet_xyz(p) for p in poly]) for poly in gate_bar_polys(g)]
    ax.add_collection3d(Poly3DCollection(polys, facecolor="#D62828", edgecolor="#8B0000", alpha=0.95))

    c = sheet_xyz(g.center)
    ax.scatter(c[0], c[1], c[2], color="blue", s=35)

    d = sheet_xyz([np.cos(g.yaw), np.sin(g.yaw), 0])
    ax.quiver(c[0], c[1], c[2], 0.18 * d[0], 0.18 * d[1], 0, color="green", lw=1.4)

    ax.text(c[0], c[1], c[2] + 0.07, f"G{g.idx}", fontsize=9)

    # floor legs from lower outer-frame corners
    ow, oh = g.outer_width, g.outer_height
    axis = np.array([np.cos(g.bar_yaw), np.sin(g.bar_yaw), 0.0])
    for side in [-1, 1]:
        top = g.center + side * ow / 2 * axis + np.array([0, 0, -oh / 2])
        bottom = top.copy(); bottom[2] = 0
        ts, bs = sheet_xyz(top), sheet_xyz(bottom)
        ax.plot([bs[0], ts[0]], [bs[1], ts[1]], [bs[2], ts[2]],
                color="#D62828", lw=1.4, alpha=0.75)


def draw_gate_2d(ax, g):
    axis = uvec(g.bar_yaw)
    p0 = sheet_xy(g.center[:2] - g.outer_width / 2 * axis)
    p1 = sheet_xy(g.center[:2] + g.outer_width / 2 * axis)
    c  = sheet_xy(g.center[:2])

    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#D62828", lw=5, solid_capstyle="round")

    x, y, z = g.center
    label = f"G{g.idx}: x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={deg(g.yaw):.1f}°, {g.size_label}"
    ax.scatter(c[0], c[1], color="blue", s=45, zorder=5, label=label)

    d = sheet_vec_xy([np.cos(g.yaw), np.sin(g.yaw)])
    ax.arrow(c[0], c[1], 0.18 * d[0], 0.18 * d[1],
             width=0.004, head_width=0.045, head_length=0.055,
             color="green", length_includes_head=True)

    ax.text(c[0] + 0.02, c[1] + 0.02, f"G{g.idx}", fontsize=10, weight="bold")


def print_gate_table(gates):
    print("\nMeasured real-arena gates:")
    print("gate | size       | x [m] | y [m] | z [m] | yaw through gate [deg]")
    print("-----+------------+-------+-------+-------+-----------------------")
    for g in gates:
        x, y, z = g.center
        print(f"G{g.idx:<3d} | {g.size_label:<10s} | {x:5.2f} | {y:5.2f} | {z:5.2f} | {deg(g.yaw):>21.1f}")


def plot_arena(gates, trajectory_wps: np.ndarray = None):
    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax2d = fig.add_subplot(gs[0, 1])

    # --- 3D view ---
    draw_room_3d(ax3d)
    draw_sector_guides_3d(ax3d)

    ax3d.scatter(*sheet_xyz([CENTER[0], CENTER[1], 0]), color="gray", s=25)
    _half = 0.15
    _home3d = np.array([
        [HOME_XY[0]-_half, HOME_XY[1]-_half, 0],
        [HOME_XY[0]+_half, HOME_XY[1]-_half, 0],
        [HOME_XY[0]+_half, HOME_XY[1]+_half, 0],
        [HOME_XY[0]-_half, HOME_XY[1]+_half, 0],
    ])
    _h3s = np.array([sheet_xyz(p) for p in _home3d])
    ax3d.add_collection3d(Poly3DCollection([_h3s], facecolor="orange", edgecolor="darkorange", alpha=0.75))
    ax3d.scatter(*sheet_xyz([HOME_XY[0], HOME_XY[1], 0.01]),
                 color="orange", s=1, label="Home pad (30×30 cm)")

    for g in gates:
        draw_gate_3d(ax3d, g)

    if trajectory_wps is not None:
        draw_trajectory_3d(ax3d, trajectory_wps)

    ax3d.set_xlim(0, ROOM_Y)
    ax3d.set_ylim(0, ROOM_X)
    ax3d.set_zlim(0, ROOM_Z)
    ax3d.invert_xaxis()
    ax3d.set_box_aspect((ROOM_Y, ROOM_X, ROOM_Z))
    ax3d.set_xlabel("Y [m]")
    ax3d.set_ylabel("X [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_title("3D real arena gate map")
    ax3d.view_init(elev=24, azim=-60)
    ax3d.legend(loc="upper left")

    # --- 2D top view ---
    draw_room_2d(ax2d)
    draw_sector_guides_2d(ax2d)

    ax2d.scatter(*sheet_xy(CENTER), color="gray", s=25, zorder=4)
    _half = 0.15
    _home2d = np.array([
        [HOME_XY[0]-_half, HOME_XY[1]-_half],
        [HOME_XY[0]+_half, HOME_XY[1]-_half],
        [HOME_XY[0]+_half, HOME_XY[1]+_half],
        [HOME_XY[0]-_half, HOME_XY[1]+_half],
    ])
    _h2s = np.array([sheet_xy(p) for p in _home2d])
    ax2d.fill(_h2s[:, 0], _h2s[:, 1], color="orange", alpha=0.75, zorder=5, label="Home pad (30×30 cm)")
    ax2d.plot(np.append(_h2s[:, 0], _h2s[0, 0]),
              np.append(_h2s[:, 1], _h2s[0, 1]),
              color="darkorange", lw=1.5, zorder=5)

    for g in gates:
        draw_gate_2d(ax2d, g)

    if trajectory_wps is not None:
        draw_trajectory_2d(ax2d, trajectory_wps)

    ax2d.set_xlim(0, ROOM_Y)
    ax2d.set_ylim(0, ROOM_X)
    ax2d.invert_xaxis()
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.set_xlabel("Y [m]")
    ax2d.set_ylabel("X [m]")
    ax2d.set_title("Top view: gates + planned trajectory")

    ax2d.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.95)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    poses = load_gate_poses(MEASUREMENTS_NPY)
    gates = poses_to_gates(poses)
    trajectory_wps = build_trajectory_room_coords(poses)
    print_gate_table(gates)
    plot_arena(gates, trajectory_wps)
