import ast
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from pyglm import glm
from app.planner.racer_polynom import RacerPolynom
from app.io import Measurement, Command, Setpoint
from app.telemetry import Telemetry

from csv_gates import load_gates_csv, GATE_SOURCE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "planner"))
from racer_z_params import compute_z_shift

# Read DEFAULT_RACE_TIME directly from racer_polynom.py source — no pyglm import needed
def _read_default_race_time() -> float:
    src_path = os.path.join(os.path.dirname(__file__), "..", "app", "planner", "racer_polynom.py")
    tree = ast.parse(open(src_path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DEFAULT_RACE_TIME":
                    if isinstance(node.value, ast.Constant):
                        return float(node.value.value)
    return 20.0  # fallback

RACE_TIME = _read_default_race_time()

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
PLOT_MARGIN    = 0.50

_LIGHTHOUSE_X = ROOM_X - 2.01 - 0.165
_HOMEPAD_X    = ROOM_X - 2.01 - 0.165 - 1.00
_ORIGIN_Y     = ROOM_Y - 1.44

DRONE_TAKEOFF = "HOMEPAD"

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

GATES_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gates", "gates_info.csv"))

_HOME_DRONE_XY = tuple(HOME_XY - DRONE_ORIGIN_XY)

# =========================
# Polynomial trajectory settings
# =========================
# RACE_TIME is read from app/planner/racer_polynom.py DEFAULT_RACE_TIME (see top of file)
POLY_STEPS = 800    # number of evaluation points for the smooth curve

# =========================
# Waypoint type colours
# =========================
WP_COLORS = {
    "home":   "orange",
    "pre":    "#2196F3",
    "center": "#E91E63",
    "post":   "#4CAF50",
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


def build_trajectory_room_coords(poses: np.ndarray):
    n = len(poses)
    lapped_idx = list(range(n)) * 2

    # Match racer_polynom.py: hover z = first gate z, not a hardcoded constant
    home_hover_z = float(poses[0][2])
    home_xyz = [HOME_XY[0], HOME_XY[1], home_hover_z]

    wps   = [home_xyz]
    types = ["home"]

    for lap_i, gi in enumerate(lapped_idx):
        x, y, z, yaw = poses[gi, :4]

        prev_gi = lapped_idx[lap_i - 1] if lap_i > 0 else None
        next_gi = lapped_idx[lap_i + 1] if lap_i < len(lapped_idx) - 1 else None

        gate_xyz = [x, y, z]
        prev_xyz = (
            [poses[prev_gi, 0], poses[prev_gi, 1], poses[prev_gi, 2]]
            if prev_gi is not None else home_xyz
        )
        next_xyz = (
            [poses[next_gi, 0], poses[next_gi, 1], poses[next_gi, 2]]
            if next_gi is not None else home_xyz
        )

        pre_shift  = compute_z_shift(prev_xyz, gate_xyz)
        post_shift = compute_z_shift(gate_xyz, next_xyz)

        dx, dy  = np.cos(yaw), np.sin(yaw)
        x_room  = DRONE_ORIGIN_XY[0] + x
        y_room  = DRONE_ORIGIN_XY[1] + y


        # pre_shift  = pre_shift  if pre_shift  > 0 else 0.0


        wps.append([x_room - APPROACH_DIST * dx, y_room - APPROACH_DIST * dy, z - 0.5 * pre_shift])
        types.append("pre")

        wps.append([x_room, y_room, z])
        types.append("center")

        wps.append([x_room + APPROACH_DIST * dx, y_room + APPROACH_DIST * dy, z + 0.5 * post_shift])
        types.append("post")

    cruise_z = float(poses[-1][2])
    wps.append([HOME_XY[0], HOME_XY[1], cruise_z])
    types.append("land")
    wps.append([HOME_XY[0], HOME_XY[1], LAND_HOVER_Z])
    types.append("land")

    return np.array(wps), types


# =========================
# Polynomial trajectory (minimum-jerk, same logic as racer_polynom.py)
# =========================

def _poly_matrix(t: float) -> np.ndarray:
    """5th-order constraint matrix at local segment time t. Shape (5, 6)."""
    return np.array([
        [1,  t,   t**2,    t**3,    t**4,    t**5],
        [0,  1,  2*t,   3*t**2,  4*t**3,  5*t**4],
        [0,  0,    2,     6*t,  12*t**2, 20*t**3],
        [0,  0,    0,       6,    24*t,  60*t**2],
        [0,  0,    0,       0,      24,    120*t],
    ])


def _solve_poly_1d(positions: np.ndarray, seg_durations: np.ndarray) -> np.ndarray:
    """Solve minimum-jerk polynomial for one dimension. Returns coeffs shape (6*(m-1),)."""
    m = len(positions)
    n = 6 * (m - 1)
    A = np.zeros((n, n))
    b = np.zeros(n)
    A0 = _poly_matrix(0.0)
    row = 0

    for i in range(m - 1):
        Af = _poly_matrix(seg_durations[i])
        c  = slice(6 * i, 6 * (i + 1))

        if i == 0:
            A[row, c] = A0[0]; b[row] = positions[0]; row += 1
            A[row, c] = A0[1]; b[row] = 0.0;          row += 1
            A[row, c] = A0[2]; b[row] = 0.0;          row += 1
            A[row, c] = Af[0]; b[row] = positions[1]; row += 1
            cn = slice(6 * (i + 1), 6 * (i + 2))
            for k in range(1, 5):
                A[row, c] = Af[k]; A[row, cn] = -A0[k]; b[row] = 0.0; row += 1

        elif i < m - 2:
            A[row, c] = A0[0]; b[row] = positions[i];     row += 1
            A[row, c] = Af[0]; b[row] = positions[i + 1]; row += 1
            cn = slice(6 * (i + 1), 6 * (i + 2))
            for k in range(1, 5):
                A[row, c] = Af[k]; A[row, cn] = -A0[k]; b[row] = 0.0; row += 1

        else:
            A[row, c] = A0[0]; b[row] = positions[i];     row += 1
            A[row, c] = Af[0]; b[row] = positions[i + 1]; row += 1
            A[row, c] = Af[1]; b[row] = 0.0;              row += 1
            A[row, c] = Af[2]; b[row] = 0.0;              row += 1

    return np.linalg.solve(A, b)


def compute_poly_trajectory(wps: np.ndarray, race_time: float, n_steps: int = POLY_STEPS) -> np.ndarray:
    """
    Compute a smooth minimum-jerk polynomial trajectory through the given waypoints.
    wps:       shape (m, 3) — x, y, z positions in room coordinates
    race_time: total duration [s]
    Returns:   shape (n_steps, 3) dense trajectory
    """
    pts    = wps[:, :3].astype(float)
    m      = len(pts)
    n_seg  = m - 1

    seg_lengths  = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total_length = float(np.sum(seg_lengths))
    BASE_FRAC    = 0.10
    base_dur     = BASE_FRAC * race_time / n_seg
    remaining    = race_time - n_seg * base_dur
    seg_durations = (base_dur + remaining * seg_lengths / total_length
                     if total_length > 0 else np.full(n_seg, race_time / n_seg))

    wp_times    = np.concatenate(([0.0], np.cumsum(seg_durations)))
    poly_coeffs = np.column_stack([
        _solve_poly_1d(pts[:, 0], seg_durations),
        _solve_poly_1d(pts[:, 1], seg_durations),
        _solve_poly_1d(pts[:, 2], seg_durations),
    ])   # shape (6*(m-1), 3)

    ts   = np.linspace(0.0, wp_times[-1], n_steps)
    traj = np.zeros((n_steps, 3))
    for idx, t in enumerate(ts):
        seg = int(np.searchsorted(wp_times, t, side='right')) - 1
        seg = int(np.clip(seg, 0, n_seg - 1))
        t_local    = t - wp_times[seg]
        row        = _poly_matrix(t_local)[0]
        traj[idx]  = poly_coeffs[6 * seg: 6 * (seg + 1), :].T @ row

    return traj


# =========================
# Drawing helpers
# =========================

def draw_trajectory_3d(ax, wps: np.ndarray, types: list):
    s = np.array([sheet_xyz(wp) for wp in wps])
    ax.plot(s[:, 0], s[:, 1], s[:, 2], '--', color='#FF69B4', lw=1.2, alpha=0.45, zorder=3)

    plotted = set()
    for wp, t, sp in zip(wps, types, s):
        label = WP_LABELS.get(t) if t not in plotted else None
        ax.scatter(sp[0], sp[1], sp[2],
                   color=WP_COLORS[t], marker=WP_MARKERS[t],
                   s=45, zorder=5, label=label)
        if WP_LABELS.get(t):
            plotted.add(t)

    i = 0
    while i < len(types):
        if types[i] == "pre" and i + 1 < len(types) and types[i + 1] == "center":
            pre_wp    = wps[i]
            center_wp = wps[i + 1]
            pre_s     = sheet_xyz(pre_wp)
            ax.plot(
                [pre_s[0], pre_s[0]], [pre_s[1], pre_s[1]], [center_wp[2], pre_wp[2]],
                color=WP_COLORS["pre"], lw=1.2, linestyle=":", alpha=0.8
            )
            i += 1
        elif types[i] == "post" and i > 0 and types[i - 1] == "center":
            post_wp   = wps[i]
            center_wp = wps[i - 1]
            post_s    = sheet_xyz(post_wp)
            ax.plot(
                [post_s[0], post_s[0]], [post_s[1], post_s[1]], [center_wp[2], post_wp[2]],
                color=WP_COLORS["post"], lw=1.2, linestyle=":", alpha=0.8
            )
            i += 1
        else:
            i += 1


def draw_trajectory_2d(ax, wps: np.ndarray, types: list):
    s = np.array([sheet_xy(wp[:2]) for wp in wps])
    ax.plot(s[:, 0], s[:, 1], '--', color='#FF69B4', lw=1.2, alpha=0.45, zorder=3,
            label='Waypoints (dashed)')

    plotted = set()
    for wp, t, sp in zip(wps, types, s):
        label = WP_LABELS.get(t) if t not in plotted else None
        ax.scatter(sp[0], sp[1],
                   color=WP_COLORS[t], marker=WP_MARKERS[t],
                   s=45, zorder=5, label=label)
        if WP_LABELS.get(t):
            plotted.add(t)

        if t in ("pre", "center", "post"):
            dz_str = f"z={wp[2]:.2f}"
            ax.annotate(dz_str, xy=(sp[0], sp[1]), xytext=(4, 4),
                        textcoords="offset points", fontsize=6,
                        color=WP_COLORS[t], alpha=0.85)


def draw_poly_trajectory_3d(ax, poly_traj: np.ndarray):
    """Draw the smooth polynomial curve in 3D."""
    s = np.array([sheet_xyz(p) for p in poly_traj])
    ax.plot(s[:, 0], s[:, 1], s[:, 2],
            '-', color='#FF6600', lw=2.0, alpha=0.90, zorder=4,
            label=f'Polynomial (t={RACE_TIME:.0f}s)')


def draw_poly_trajectory_2d(ax, poly_traj: np.ndarray):
    """Draw the smooth polynomial curve in 2D."""
    s = np.array([sheet_xy(p[:2]) for p in poly_traj])
    ax.plot(s[:, 0], s[:, 1],
            '-', color='#FF6600', lw=2.0, alpha=0.90, zorder=4,
            label=f'Polynomial (t={RACE_TIME:.0f}s)')


# =========================
# Gate geometry helpers
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

    wp_records = []
    for wp, t in zip(wps, types):
        if t in ("pre", "center", "post"):
            wp_records.append((wp, t))

    triplets = []
    i = 0
    while i + 2 < len(wp_records):
        if wp_records[i][1] == "pre" and wp_records[i+1][1] == "center" and wp_records[i+2][1] == "post":
            triplets.append((wp_records[i][0], wp_records[i+1][0], wp_records[i+2][0]))
            i += 3
        else:
            i += 1

    n_gates   = len(gates)
    first_lap = triplets[:n_gates]

    for g, (pre, center, post) in zip(gates, first_lap):
        x, y, z = g.center
        pre_dz  = pre[2]  - center[2]
        post_dz = post[2] - center[2]
        print(
            f"G{g.idx:<3d} | {g.size_label:<10s} | {x:5.2f} | {y:5.2f} | {z:5.2f} | "
            f"{deg(g.yaw):>9.1f} | {pre_dz:>+10.3f} | {post_dz:>+10.3f}"
        )


def plot_arena(gates, wps: np.ndarray, types: list, sim_pos: np.ndarray = None, plan_pos: np.ndarray = None):
    print(f"Computing polynomial trajectory (race_time={RACE_TIME}s, {POLY_STEPS} steps)...")
    poly_traj = compute_poly_trajectory(wps, RACE_TIME)
    print("Done.")

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
    draw_poly_trajectory_3d(ax3d, poly_traj)

    # Overlay planner-produced setpoints and simulated measurements if provided
    if plan_pos is not None and plan_pos.size:
        try:
            pp = np.array([sheet_xyz(p) for p in plan_pos])
            ax3d.plot(pp[:, 0], pp[:, 1], pp[:, 2], ':', color='C2', lw=1.6, alpha=0.9, zorder=6, label='Planner setpoints')
        except Exception:
            pass

    if sim_pos is not None and sim_pos.size:
        try:
            sp = np.array([sheet_xyz(p) for p in sim_pos])
            ax3d.plot(sp[:, 0], sp[:, 1], sp[:, 2], '--', color='C1', lw=1.8, alpha=0.95, zorder=7, label='Simulated path')
        except Exception:
            pass

    ax3d.set_xlim(-PLOT_MARGIN, ROOM_Y + PLOT_MARGIN)
    ax3d.set_ylim(-PLOT_MARGIN, ROOM_X + PLOT_MARGIN)
    ax3d.set_zlim(0, ROOM_Z)
    ax3d.invert_xaxis()
    ax3d.set_box_aspect((ROOM_Y, ROOM_X, ROOM_Z))
    ax3d.set_xlabel("Y [m]")
    ax3d.set_ylabel("X [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_title(f"3D arena — polynomial trajectory (race_time={RACE_TIME}s)")
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
    draw_poly_trajectory_2d(ax2d, poly_traj)

    # 2D overlays for planner and simulation
    if plan_pos is not None and plan_pos.size:
        try:
            pp2 = np.array([sheet_xy(p[:2]) for p in plan_pos])
            ax2d.plot(pp2[:, 0], pp2[:, 1], ':', color='C2', lw=1.6, alpha=0.9, label='Planner setpoints')
        except Exception:
            pass

    if sim_pos is not None and sim_pos.size:
        try:
            sp2 = np.array([sheet_xy(p[:2]) for p in sim_pos])
            ax2d.plot(sp2[:, 0], sp2[:, 1], '--', color='C1', lw=1.8, alpha=0.95, label='Simulated path')
        except Exception:
            pass

    ax2d.set_xlim(-PLOT_MARGIN, ROOM_Y + PLOT_MARGIN)
    ax2d.set_ylim(-PLOT_MARGIN, ROOM_X + PLOT_MARGIN)
    ax2d.invert_xaxis()
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.set_xlabel("Y [m]")
    ax2d.set_ylabel("X [m]")
    ax2d.set_title(f"Top view — polynomial trajectory (race_time={RACE_TIME}s)")

    ax2d.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.95)

    plt.tight_layout()
    plt.show()


# =========================
# Simulation
# =========================

def run_simulation(planner: RacerPolynom, poses: np.ndarray, duration: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the planner + Telemetry-style simulation loop.

    Returns
    sim_pos : (N,3) numpy array of simulated positions (Measurement.simulate world coords)
    plan_pos: (N,3) numpy array of planner setpoints returned by planner.update (in room coords)
    """
    # Match app timing
    SIM_DT = Telemetry.UPDATE_PERIOD / 1000.0

    # Initialize planner then override its gates with the poses we loaded for plotting
    planner.reload()

    # Build planner.gates from poses (poses are in CSV/planner frame: x,y,z,yaw,...)
    positions = [glm.vec3(float(row[0]), float(row[1]), float(row[2])) for row in poses]
    yaws = [float(row[3]) for row in poses]
    planner.gates = [Setpoint(p, y) for p, y in zip(positions, yaws)]

    # Rebuild planner internal state (waypoints, wp_times, poly_coeffs) using same logic as reload()
    planner.started = False
    planner.elapsed = 0.0
    planner.hover_setpoint = Setpoint(glm.vec3(0.0, 0.0, planner.gates[0].position.z), RacerPolynom.HOME_YAW)
    planner.waypoints.clear()

    home_to_first = planner.gates[0].position - planner.hover_setpoint.position
    home_yaw = float(np.atan2(home_to_first.y, home_to_first.x))
    planner.waypoints.append(Setpoint(planner.hover_setpoint.position, home_yaw))

    lapped_gates = planner.gates * 2
    home_pos = planner.hover_setpoint.position

    for i, gate in enumerate(lapped_gates):
        prev_pos = lapped_gates[i - 1].position if i > 0                    else home_pos
        next_pos = lapped_gates[i + 1].position if i < len(lapped_gates) - 1 else home_pos

        g          = [gate.position.x, gate.position.y, gate.position.z]
        pre_shift  = compute_z_shift([prev_pos.x, prev_pos.y, prev_pos.z], g)
        post_shift = compute_z_shift(g, [next_pos.x, next_pos.y, next_pos.z])

        normal = glm.vec3(
            RacerPolynom.APPROACH_DIST * np.cos(gate.yaw),
            RacerPolynom.APPROACH_DIST * np.sin(gate.yaw),
            0.0,
        )
        pre_pos  = gate.position - normal
        post_pos = gate.position + normal

        planner.waypoints.append(Setpoint(glm.vec3(pre_pos.x,  pre_pos.y,  gate.position.z - 0.5 * pre_shift),  gate.yaw))
        planner.waypoints.append(Setpoint(gate.position, gate.yaw))
        planner.waypoints.append(Setpoint(glm.vec3(post_pos.x, post_pos.y, gate.position.z + 0.5 * post_shift), gate.yaw))

    planner.waypoints.append(planner.hover_setpoint)

    # Compute segment durations and polynomial coefficients
    pts = np.array([[wp.position.x, wp.position.y, wp.position.z] for wp in planner.waypoints])
    planner.wp_yaws = np.array([wp.yaw for wp in planner.waypoints])

    seg_lengths   = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total_length  = float(np.sum(seg_lengths))
    n_seg         = len(planner.waypoints) - 1
    BASE_FRAC     = 0.10
    base_dur      = BASE_FRAC * planner.race_time / n_seg
    remaining     = planner.race_time - n_seg * base_dur
    seg_durations = base_dur + remaining * seg_lengths / total_length if total_length > 0 else np.full(n_seg, planner.race_time / n_seg)

    planner.wp_times = np.concatenate(([0.0], np.cumsum(seg_durations)))

    cx = planner._solve_poly_1d(pts[:, 0], seg_durations)
    cy = planner._solve_poly_1d(pts[:, 1], seg_durations)
    cz = planner._solve_poly_1d(pts[:, 2], seg_durations)
    planner.poly_coeffs = np.column_stack([cx, cy, cz])

    # initial measurement: start above home pad at first gate height (same as planner.hover_setpoint)
    if planner.gates is None or len(planner.gates) == 0:
        raise RuntimeError("Planner has no gates; call planner.reload() or ensure gates directory is correct")

    start_z = float(planner.gates[0].position.z)
    # Ensure we start at or above takeoff height used by Telemetry
    start_z = max(start_z, Telemetry.TKOF_HEIGHT)
    start_pos = glm.vec3(float(HOME_XY[0]), float(HOME_XY[1]), start_z)
    print(f"DEBUG: sim start_pos={start_pos}, planner.hover_setpoint={planner.hover_setpoint}")
    meas = Measurement(timestamp=0.0, position=start_pos, rotation=glm.vec3(0.0, 0.0, 0.0), battery=1.0)
 
    total_time = duration if duration is not None else planner.race_time
    total_ticks = int(np.ceil(total_time / SIM_DT)) + 1

    sim_positions = []
    plan_positions = []

    flags = Telemetry.Flags.START

    for tick in range(total_ticks):
        # Convert measurement into planner frame (planner uses CSV/drone-origin coords)
        meas_planner = Measurement(
            timestamp=meas.timestamp,
            position=glm.vec3(float(meas.position.x - DRONE_ORIGIN_XY[0]), float(meas.position.y - DRONE_ORIGIN_XY[1]), float(meas.position.z)),
            rotation=meas.rotation,
            battery=meas.battery,
        )

        # planner produces an intermediate setpoint (as in Telemetry loop) in planner frame
        setpt = planner.update(meas_planner, None, flags, SIM_DT)
        flags = Telemetry.Flags.NEITHER

        # convert planner setpoint back to room coordinates for plotting
        setpt_room = glm.vec3(float(setpt.position.x + DRONE_ORIGIN_XY[0]), float(setpt.position.y + DRONE_ORIGIN_XY[1]), float(setpt.position.z))
        plan_positions.append([float(setpt_room.x), float(setpt_room.y), float(setpt_room.z)])

        # convert setpoint -> command using same semantics as Telemetry.send_setpoint
        # (use planner-frame measurement when producing the Command)
        cmd = setpt.to_command(meas_planner)

        # run Telemetry-like simulation: Measurement.simulate(command, dt)
        meas_next = meas.simulate(cmd, SIM_DT)

        # Compact diagnostics for the first few ticks
        if tick < 6:
            print(
                f"T={tick:02d}: meas(room)={tuple(map(float, (meas.position.x, meas.position.y, meas.position.z)))} |",
                f"meas_planner={tuple(map(float, (meas_planner.position.x, meas_planner.position.y, meas_planner.position.z)))} |",
                f"setpt_room={tuple(map(float, (setpt_room.x, setpt_room.y, setpt_room.z)))} |",
                f"cmd_vel={tuple(map(float, (cmd.velocity.x, cmd.velocity.y, cmd.velocity.z)))} |",
                f"meas_next(room)={tuple(map(float, (meas_next.position.x, meas_next.position.y, meas_next.position.z)))}"
            )

        meas = meas_next

        # record simulated measurement position (world/room coords)
        sim_positions.append([float(meas.position.x), float(meas.position.y), float(meas.position.z)])

    return np.array(sim_positions), np.array(plan_positions)


def plot_plan_and_sim(sim_pos: np.ndarray, plan_pos: np.ndarray, show_3d: bool = True) -> None:
    """Simple 2D/3D plotting of simulated vs planned paths."""
    if sim_pos.shape[0] == 0 or plan_pos.shape[0] == 0:
        print("No data to plot")
        return

    # 2D XY plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(plan_pos[:, 0], plan_pos[:, 1], '-C0', label='Planned (planner.update)')
    ax.plot(sim_pos[:, 0], sim_pos[:, 1], '--C1', label='Simulated (Measurement.simulate)')
    ax.scatter(plan_pos[0,0], plan_pos[0,1], c='C2', marker='o', label='Planned start')
    ax.scatter(sim_pos[0,0], sim_pos[0,1], c='C3', marker='x', label='Sim start')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.legend()
    ax.grid(True)

    if show_3d:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig3 = plt.figure(figsize=(8, 6))
        ax3 = fig3.add_subplot(111, projection='3d')
        ax3.plot(plan_pos[:, 0], plan_pos[:, 1], plan_pos[:, 2], '-C0', label='Planned')
        ax3.plot(sim_pos[:, 0], sim_pos[:, 1], sim_pos[:, 2], '--C1', label='Simulated')
        ax3.set_xlabel('X [m]')
        ax3.set_ylabel('Y [m]')
        ax3.set_zlabel('Z [m]')
        ax3.legend()
        ax3.set_box_aspect((np.ptp(plan_pos[:,0]), np.ptp(plan_pos[:,1]), np.ptp(plan_pos[:,2]) if np.ptp(plan_pos[:,2])>0 else 1.0))

    plt.show()


if __name__ == "__main__":
    print(f"Loading gates from: {GATES_CSV}  (GATE_SOURCE={GATE_SOURCE})")
    poses = load_gates_csv(GATES_CSV, home_xy=_HOME_DRONE_XY)
    gates = poses_to_gates(poses)
    wps, types = build_trajectory_room_coords(poses)
    # print_gate_table(gates, wps, types)
    # plot_arena(gates, wps, types)

    planner = RacerPolynom(race_time=10.0)
    sim_pos, plan_pos = run_simulation(planner, poses)
    # show simulation overlay on the same arena figure
    plot_arena(gates, wps, types, sim_pos=sim_pos, plan_pos=plan_pos)
