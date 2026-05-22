import os
import numpy as np
from pyglm import glm
from cv2.typing import MatLike
from overrides import overrides
from app import wrap
from app.io import Measurement
from app.io import Setpoint
from app.planner import Planner
from app.telemetry import Telemetry
from app.planner.racer_z_params import compute_z_shift


DEFAULT_RACE_TIME = 30.0  # seconds — single source of truth, also read by visulazitaion_v3.0.py


class RacerPolynom(Planner):

    GATES_DIRECTORY = "gates"

    def __init__(self, race_time: float = DEFAULT_RACE_TIME):
        super().__init__()
        self.race_time = race_time
        self.started = False
        self.elapsed = 0.0
        self.poly_coeffs: np.ndarray | None = None  # shape (6*(m-1), 3)
        self.wp_times: np.ndarray | None = None     # cumulative time at each waypoint, shape (m,)
        self.wp_yaws: np.ndarray | None = None      # yaw at each waypoint, shape (m,)
        self.hover_setpoint = RacerPolynom.HOME_SETPOINT

    # ------------------------------------------------------------------
    # Polynomial math
    # ------------------------------------------------------------------

    def _poly_matrix(self, t: float) -> np.ndarray:
        """5th-order constraint matrix at local time t. Shape (5, 6)."""
        return np.array([
            [1,  t,   t**2,    t**3,    t**4,    t**5],
            [0,  1,  2*t,   3*t**2,  4*t**3,  5*t**4],
            [0,  0,    2,     6*t,  12*t**2, 20*t**3],
            [0,  0,    0,       6,    24*t,  60*t**2],
            [0,  0,    0,       0,      24,    120*t],
        ])

    def _solve_poly_1d(self, positions: np.ndarray, seg_durations: np.ndarray) -> np.ndarray:
        """
        Minimum-jerk polynomial for one dimension.
        positions:     shape (m,)   — position at each waypoint
        seg_durations: shape (m-1,) — duration of each segment
        Returns coefficients: shape (6*(m-1),)
        """
        m = len(positions)
        n = 6 * (m - 1)
        A = np.zeros((n, n))
        b = np.zeros(n)
        A0 = self._poly_matrix(0.0)
        row = 0

        for i in range(m - 1):
            Af = self._poly_matrix(seg_durations[i])
            c = slice(6 * i, 6 * (i + 1))

            if i == 0:
                # Initial constraints: pos, vel=0, acc=0
                A[row, c] = A0[0]; b[row] = positions[0]; row += 1
                A[row, c] = A0[1]; b[row] = 0.0;          row += 1
                A[row, c] = A0[2]; b[row] = 0.0;          row += 1
                # Final position of this segment
                A[row, c] = Af[0]; b[row] = positions[1]; row += 1
                # Continuity of vel, acc, jerk, snap with next segment
                cn = slice(6 * (i + 1), 6 * (i + 2))
                for k in range(1, 5):
                    A[row, c] = Af[k]; A[row, cn] = -A0[k]; b[row] = 0.0; row += 1

            elif i < m - 2:
                # Intermediate: initial and final position
                A[row, c] = A0[0]; b[row] = positions[i];     row += 1
                A[row, c] = Af[0]; b[row] = positions[i + 1]; row += 1
                # Continuity with next segment
                cn = slice(6 * (i + 1), 6 * (i + 2))
                for k in range(1, 5):
                    A[row, c] = Af[k]; A[row, cn] = -A0[k]; b[row] = 0.0; row += 1

            else:
                # Final segment: initial pos, final pos, final vel=0, final acc=0
                A[row, c] = A0[0]; b[row] = positions[i];     row += 1
                A[row, c] = Af[0]; b[row] = positions[i + 1]; row += 1
                A[row, c] = Af[1]; b[row] = 0.0;              row += 1
                A[row, c] = Af[2]; b[row] = 0.0;              row += 1

        return np.linalg.solve(A, b)

    def _eval(self, t: float) -> Setpoint:
        """Evaluate position and yaw at global elapsed time t."""
        t = float(np.clip(t, self.wp_times[0], self.wp_times[-1]))
        seg = int(np.searchsorted(self.wp_times, t, side='right')) - 1
        seg = int(np.clip(seg, 0, len(self.wp_times) - 2))

        t_local = t - self.wp_times[seg]
        row = self._poly_matrix(t_local)[0]

        x = float(np.dot(row, self.poly_coeffs[6 * seg: 6 * (seg + 1), 0]))
        y = float(np.dot(row, self.poly_coeffs[6 * seg: 6 * (seg + 1), 1]))
        z = float(np.dot(row, self.poly_coeffs[6 * seg: 6 * (seg + 1), 2]))

        # Linear yaw interpolation within the segment
        dur = self.wp_times[seg + 1] - self.wp_times[seg]
        alpha = (t_local / dur) if dur > 0 else 1.0
        yaw = wrap(self.wp_yaws[seg] + alpha * wrap(self.wp_yaws[seg + 1] - self.wp_yaws[seg]))

        return Setpoint(glm.vec3(x, y, z), yaw)

    # ------------------------------------------------------------------
    # Planner interface
    # ------------------------------------------------------------------

    @overrides
    def reload(self) -> None:

        # Load gates — identical to racer.py
        gates_directory = os.path.join(RacerPolynom.GATES_DIRECTORY)
        file_name = os.listdir(gates_directory)[0]
        file_path = os.path.join(gates_directory, file_name)

        raw_csv_data: np.ndarray  = np.genfromtxt(file_path, delimiter=',')[1:]
        positions: list[glm.vec3] = [glm.vec3(row[1], row[2], row[3]) for row in raw_csv_data]
        yaws: list[float]         = [wrap(row[4]) for row in raw_csv_data]
        self.gates                = [Setpoint(position, yaw) for position, yaw in zip(positions, yaws)]

        # Reset state
        self.started = False
        self.elapsed = 0.0
        self.hover_setpoint = Setpoint(glm.vec3(0.0, 0.0, self.gates[0].position.z), RacerPolynom.HOME_YAW)
        self.waypoints.clear()

        # Build waypoint list — identical to racer.py
        home_to_first = self.gates[0].position - self.hover_setpoint.position
        home_yaw = float(np.atan2(home_to_first.y, home_to_first.x))
        self.waypoints.append(Setpoint(self.hover_setpoint.position, home_yaw))

        lapped_gates = self.gates * 2
        home_pos = self.hover_setpoint.position

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

            self.waypoints.append(Setpoint(glm.vec3(pre_pos.x,  pre_pos.y,  gate.position.z - 0.5 * pre_shift),  gate.yaw))
            self.waypoints.append(Setpoint(gate.position, gate.yaw))
            self.waypoints.append(Setpoint(glm.vec3(post_pos.x, post_pos.y, gate.position.z + 0.5 * post_shift), gate.yaw))

        self.waypoints.append(self.hover_setpoint)

        # Compute segment durations proportional to arc length (with a base share)
        pts = np.array([[wp.position.x, wp.position.y, wp.position.z] for wp in self.waypoints])
        self.wp_yaws = np.array([wp.yaw for wp in self.waypoints])

        seg_lengths   = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total_length  = float(np.sum(seg_lengths))
        n_seg         = len(self.waypoints) - 1
        BASE_FRAC     = 0.10   # 10 % of race_time shared equally across all segments
        base_dur      = BASE_FRAC * self.race_time / n_seg
        remaining     = self.race_time - n_seg * base_dur
        seg_durations = base_dur + remaining * seg_lengths / total_length if total_length > 0 \
                        else np.full(n_seg, self.race_time / n_seg)

        self.wp_times = np.concatenate(([0.0], np.cumsum(seg_durations)))

        # Solve polynomial coefficients for x, y, z independently
        cx = self._solve_poly_1d(pts[:, 0], seg_durations)
        cy = self._solve_poly_1d(pts[:, 1], seg_durations)
        cz = self._solve_poly_1d(pts[:, 2], seg_durations)
        self.poly_coeffs = np.column_stack([cx, cy, cz])   # (6*(m-1), 3)

    @overrides
    def update(self, measurement: Measurement, frame: MatLike, flags: Telemetry.Flags, dt: float) -> Setpoint:

        if Telemetry.Flags.START in flags:
            self.started = True

        if not self.started or self.poly_coeffs is None:
            return self.hover_setpoint

        self.elapsed += dt

        if self.elapsed >= self.wp_times[-1]:
            return self.hover_setpoint

        return self._eval(self.elapsed)
