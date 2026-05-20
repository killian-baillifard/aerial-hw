import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from csv_gates import load_gates_csv, GATE_SOURCE

# Shared z-shaping parameters and formula (single source of truth with app/planner/racer.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.planner.racer_z_params import compute_z_shift

# =========================
# Real arena dimensions [m]
# =========================
ROOM_X = 4.05
ROOM_Y = 2.87
ROOM_Z = 3.00

CENTER   = np.array([ROOM_X / 2, ROOM_Y / 2], dtype=float)
HOME_Z   = 0.10

# =========================
# Sector / gate settings
# =========================
HOME_HALF_ANGLE_DEG  = 45.0

FRAME_WIDTH    = 0.08
GATE_HEIGHT    = 0.40
WALL_CLEARANCE = 0.04

APPROACH_DIST  = 0.20
LAND_HOVER_Z   = 0.30
HOME_HOVER_Z   = 0.50   # race hover altitude (z of home waypoint in trajectory)
PLOT_MARGIN    = 0.50

_LIGHTHOUSE_X = ROOM_X - 2.01 - 0.165
_HOMEPAD_X    = ROOM_X - 2.01 - 0.165 - 1.00
_ORIGIN_Y     = ROOM_Y - 1.44

DRONE_TAKEOFF = "LIGHTHOUSE"

DRONE_ORIGIN_XY = np.array([
    {"LIGHTHOUSE": _LIGHTHOUSE_X, "HOMEPAD": _HOMEPAD_X}[DRONE_TAKEOFF],
    _ORIGIN_Y,
])
HOME_XY = np.array([_HOMEPAD_X, _ORIGIN_Y], dtype=float)

GATE_NOMINAL_SIZES = {
    "50x40 cm": (0.50, 0.40),
    "40x40 cm": (0.40, 0.40),
    "29x40 cm": (0.29, 0.40),
}

GATES_CSV = os.path.join(os.path.dirname(__file__), "gates_test.csv")

_HOME_DRONE_XY = tuple(HOME_XY - DRONE_ORIGIN_XY)

# =========================
# Waypoint type colours
# =========================
WP_COLORS = {
    "home":   "orange",
    "pre":    "#2196F3",    # blue
    "center": "#E91E63",    # pink/red
    "post":   "#4CAF50",    # green
    "land":   "orange",
}
WP_MARKERS = {
    "home":   "s",
    "pre":    "v",
    "center": "o",
    "post":   "^",
    "land":   "s",
}
WP_LABELS = {
    "home":   "Home",
    "pre":    "Pre-gate",
    "center": "Gate center",
    "post":   "Post-gate",
    "land":   None,
}


@dataclass
class Gate:
    idx: int
    size_label: str
    width: float
    height: float
    frame: float
    center: np.ndarray
    yaw: float
    bar_yaw: float

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
    p = np.asarray(p, dtype=float)
    return np.array([p[1], p[0]])


def sheet_xyz(p):
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
    gates = []
    for i, row in enumerate(poses):
        x, y, z, yaw = row[:4]
        width  = float(row[4]) if poses.shape[1] > 4 else 0.50
        height = float(row[5]) if poses.shape[1] > 5 else 0.40
        x_room   = DRONE_ORIGIN_XY[0] + x
        y_room   = DRONE_ORIGIN_XY[1] + y
        yaw_room = float(yaw)
        label    = f"{int(round(width * 100))}x{int(round(height * 100))} cm"
        gates.append(Gate(
            idx=i + 1,
            size_label=label,
            width=width,
            height=height,
            frame=FRAME_WIDTH,
            center=np.array([x_room, y_room, z]),
            yaw=yaw_room,
            bar_yaw=float(wrap(yaw_room + np.pi / 2)),
        ))
    return gates


# =========================
# Trajectory builder — now returns (wps array, type list)
# Types: 'home', 'pre', 'center', 'post', 'land'
# =========================
def build_trajectory_room_coords(poses: np.ndarray):
    n = len(poses)
    lapped_idx = list(range(n)) * 2     # gate indices for 2 laps

    home_xyz = [HOME_XY[0], HOME_XY[1], HOME_HOVER_Z]

    wps   = [home_xyz]
    types = ["home"]

    for lap_i, gi in enumerate(lapped_idx):
        x, y, z, yaw = poses[gi, :4]

        prev_gi = lapped_idx[lap_i - 1] if lap_i > 0 else None
        next_gi = lapped_idx[lap_i + 1] if lap_i < len(lapped_idx) - 1 else None

        prev_xyz = (
            [poses[prev_gi, 0], poses[prev_gi, 1], poses[prev_gi, 2]]
            if prev_gi is not None else home_xyz
        )
        next_xyz = (
            [poses[next_gi, 0], poses[next_gi, 1], poses[next_gi, 2]]
            if next_gi is not None else home_xyz
        )

        z_shift = compute_z_shift(prev_xyz, next_xyz)

        dx, dy  = np.cos(yaw), np.sin(yaw)
        x_room  = DRONE_ORIGIN_XY[0] + x
        y_room  = DRONE_ORIGIN_XY[1] + y

        wps.append([x_room - APPROACH_DIST * dx, y_room - APPROACH_DIST * dy, z - 0.5 * z_shift])
        types.append("pre")

        wps.append([x_room, y_room, z])
        types.append("center")

        wps.append([x_room + APPROACH_DIST * dx, y_room + APPROACH_DIST * dy, z + 0.5 * z_shift])
        types.append("post")

    cruise_z = float(poses[-1][2])
    wps.append([HOME_XY[0], HOME_XY[1], cruise_z])
    types.append("land")
    wps.append([HOME_XY[0], HOME_XY[1], LAND_HOVER_Z])
    types.append("land")

    return np.array(wps), types


# =========================
# Drawing helpers
# =========================
def draw_trajectory_3d(ax, wps: np.ndarray, types: list):
    s = np.array([sheet_xyz(wp) for wp in wps])

    # Connect all waypoints with a dashed line
    ax.plot(s[:, 0], s[:, 1], s[:, 2], '--', color='#FF69B4', lw=1.4, alpha=0.6, zorder=3)

    # Scatter each type separately for the legend
    plotted = set()
    for wp, t, sp in zip(wps, types, s):
        label = WP_LABELS.get(t) if t not in plotted else None
        ax.scatter(sp[0], sp[1], sp[2],
                   color=WP_COLORS[t], marker=WP_MARKERS[t],
                   s=45, zorder=5, label=label)
        if WP_LABELS.get(t):
            plotted.add(t)

    # Vertical lines from gate center z to pre/post z to show the offset
    i = 0
    while i < len(types):
        if types[i] == "pre" and i + 1 < len(types) and types[i + 1] == "center":
            pre_wp    = wps[i]
            center_wp = wps[i + 1]
            pre_s     = sheet_xyz(pre_wp)
            # vertical line at pre xy from center_z down/up to pre_z
            ax.plot(
                [pre_s[0], pre_s[0]],
                [pre_s[1], pre_s[1]],
                [center_wp[2], pre_wp[2]],
                color=WP_COLORS["pre"], lw=1.2, linestyle=":", alpha=0.8
            )
            i += 1
        elif types[i] == "post" and i > 0 and types[i - 1] == "center":
            post_wp   = wps[i]
            center_wp = wps[i - 1]
            post_s    = sheet_xyz(post_wp)
            ax.plot(
                [post_s[0], post_s[0]],
                [post_s[1], post_s[1]],
                [center_wp[2], post_wp[2]],
                color=WP_COLORS["post"], lw=1.2, linestyle=":", alpha=0.8
            )
            i += 1
        else:
            i += 1


def draw_trajectory_2d(ax, wps: np.ndarray, types: list):
    s = np.array([sheet_xy(wp[:2]) for wp in wps])

    ax.plot(s[:, 0], s[:, 1], '--', color='#FF69B4', lw=1.4, alpha=0.6, zorder=3,
            label='Planned trajectory')

    plotted = set()
    for wp, t, sp in zip(wps, types, s):
        label = WP_LABELS.get(t) if t not in plotted else None
        ax.scatter(sp[0], sp[1],
                   color=WP_COLORS[t], marker=WP_MARKERS[t],
                   s=45, zorder=5, label=label)
        if WP_LABELS.get(t):
            plotted.add(t)

        # Annotate z value on pre/post so the height shift is readable in 2D
        if t in ("pre", "center", "post"):
            dz_str = f"z={wp[2]:.2f}"
            ax.annotate(dz_str, xy=(sp[0], sp[1]), xytext=(4, 4),
                        textcoords="offset points", fontsize=6,
                        color=WP_COLORS[t], alpha=0.85)


# =========================
# Gate geometry helpers (unchanged from v1.0)
# =========================
def gate_bar_polys(g):
    c = g.center
    axis  = np.array([np.cos(g.bar_yaw), np.sin(g.bar_yaw), 0.0])
    zaxis = np.array([0.0, 0.0, 1.0])

    iw, ih = g.width, g.height
    ow, oh = g.outer_width, g.outer_height

    def p(u, z):
        return c + u * axis + z * zaxis

    specs = [
        (-ow / 2,  ow / 2,  ih / 2,  oh / 2),
        (-ow / 2,  ow / 2, -oh / 2, -ih / 2),
        (-ow / 2, -iw / 2, -ih / 2,  ih / 2),
        ( iw / 2,  ow / 2, -ih / 2,  ih / 2),
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
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        ax.plot([s[a,0], s[b,0]], [s[a,1], s[b,1]], [s[a,2], s[b,2]],
                color="black", lw=1, alpha=0.35)


def draw_sector_guides_2d(ax):
    c = sheet_xy(CENTER)
    home_angle = np.arctan2(HOME_XY[1] - CENTER[1], HOME_XY[0] - CENTER[0])
    home_start = home_angle - np.deg2rad(HOME_HALF_ANGLE_DEG)
    home_end   = home_angle + np.deg2rad(HOME_HALF_ANGLE_DEG)

    angles_home = np.linspace(home_start, home_end, 30)
    wedge = [CENTER] + [ray_to_wall(CENTER, uvec(a)) for a in angles_home] + [CENTER]
    ws = np.array([sheet_xy(p) for p in wedge])
    ax.fill(ws[:, 0], ws[:, 1], color="#8CBDF2", alpha=0.35, zorder=1, label="Home zone (90°)")

    for k in range(9):
        zone_start = home_end + k * np.deg2rad(30)
        zone_end   = zone_start + np.deg2rad(30)
        if k % 2 != 0:
            angles_zone = np.linspace(zone_start, zone_end, 10)
            wedge = [CENTER] + [ray_to_wall(CENTER, uvec(a)) for a in angles_zone] + [CENTER]
            ws = np.array([sheet_xy(p) for p in wedge])
            ax.fill(ws[:, 0], ws[:, 1], color="#C8C8C8", alpha=0.45, zorder=1)

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

    angles_home = np.linspace(home_start, home_end, 30)
    wedge = [[CENTER[0], CENTER[1], 0]]
    wedge += [[*ray_to_wall(CENTER, uvec(a)), 0] for a in angles_home]
    wedge += [[CENTER[0], CENTER[1], 0]]
    ws = np.array([sheet_xyz(p) for p in wedge])
    ax.add_collection3d(Poly3DCollection([ws], facecolor="#8CBDF2", edgecolor="none", alpha=0.25))

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

    ow, oh = g.outer_width, g.outer_height
    axis = np.array([np.cos(g.bar_yaw), np.sin(g.bar_yaw), 0.0])
    for side in [-1, 1]:
        top    = g.center + side * ow / 2 * axis + np.array([0, 0, -oh / 2])
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

    ax.annotate(f"G{g.idx}", xy=(c[0], c[1]), xytext=(10, 6),
                textcoords="offset points", fontsize=10, weight="bold")


def print_gate_table(gates, wps, types):
    print("\nMeasured real-arena gates:")
    print("gate | size       | x [m] | y [m] | z [m] | yaw [deg] | pre_dz [m] | post_dz [m]")
    print("-----+------------+-------+-------+-------+-----------+------------+------------")

    # Collect per-gate center/pre/post from first lap only
    wp_records = []
    for wp, t in zip(wps, types):
        if t in ("pre", "center", "post"):
            wp_records.append((wp, t))

    # Group into triplets: (pre, center, post)
    triplets = []
    i = 0
    while i + 2 < len(wp_records):
        if wp_records[i][1] == "pre" and wp_records[i+1][1] == "center" and wp_records[i+2][1] == "post":
            triplets.append((wp_records[i][0], wp_records[i+1][0], wp_records[i+2][0]))
            i += 3
        else:
            i += 1

    # First lap only = first n_gates triplets
    n_gates = len(gates)
    first_lap = triplets[:n_gates]

    for g, (pre, center, post) in zip(gates, first_lap):
        x, y, z = g.center
        pre_dz  = pre[2]  - center[2]
        post_dz = post[2] - center[2]
        print(
            f"G{g.idx:<3d} | {g.size_label:<10s} | {x:5.2f} | {y:5.2f} | {z:5.2f} | "
            f"{deg(g.yaw):>9.1f} | {pre_dz:>+10.3f} | {post_dz:>+10.3f}"
        )


def plot_arena(gates, wps: np.ndarray, types: list):
    fig = plt.figure(figsize=(15, 7.5))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax2d = fig.add_subplot(gs[0, 1])

    # --- 3D ---
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
                 color="orange", s=1, label="Home pad")

    for g in gates:
        draw_gate_3d(ax3d, g)

    draw_trajectory_3d(ax3d, wps, types)

    ax3d.set_xlim(-PLOT_MARGIN, ROOM_Y + PLOT_MARGIN)
    ax3d.set_ylim(-PLOT_MARGIN, ROOM_X + PLOT_MARGIN)
    ax3d.set_zlim(0, ROOM_Z)
    ax3d.invert_xaxis()
    ax3d.set_box_aspect((ROOM_Y, ROOM_X, ROOM_Z))
    ax3d.set_xlabel("Y [m]")
    ax3d.set_ylabel("X [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_title("3D arena — z offsets on pre (▼) / post (▲) waypoints")
    ax3d.view_init(elev=24, azim=-60)
    ax3d.legend(loc="upper left", fontsize=8)

    # --- 2D ---
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
    ax2d.fill(_h2s[:, 0], _h2s[:, 1], color="orange", alpha=0.75, zorder=5, label="Home pad")
    ax2d.plot(np.append(_h2s[:, 0], _h2s[0, 0]),
              np.append(_h2s[:, 1], _h2s[0, 1]),
              color="darkorange", lw=1.5, zorder=5)

    for g in gates:
        draw_gate_2d(ax2d, g)

    draw_trajectory_2d(ax2d, wps, types)

    ax2d.set_xlim(-PLOT_MARGIN, ROOM_Y + PLOT_MARGIN)
    ax2d.set_ylim(-PLOT_MARGIN, ROOM_X + PLOT_MARGIN)
    ax2d.invert_xaxis()
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.set_xlabel("Y [m]")
    ax2d.set_ylabel("X [m]")
    ax2d.set_title("Top view — z annotated on each waypoint")

    ax2d.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.95)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(f"Loading gates from: {GATES_CSV}  (GATE_SOURCE={GATE_SOURCE})")
    poses = load_gates_csv(GATES_CSV, home_xy=_HOME_DRONE_XY)
    gates = poses_to_gates(poses)
    wps, types = build_trajectory_room_coords(poses)
    print_gate_table(gates, wps, types)
    plot_arena(gates, wps, types)
