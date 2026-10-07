"""Construction of the MPC problem (11) for the planar single integrator of Sec. VI-A.

Model (Eq. (6) with ``A = I``, ``B = dt I``)::

    z_{k+1} = z_k + dt * u_k,   z in R^2, u in R^2.

Stacking ``u = [u_0; ...; u_{N-1}] in R^{2N}`` the predicted states ``z_hat_k`` for ``k = 1..N`` are
``z_hat = 1 (x) z_t + S u`` (Eq. (9)) with ``S`` block lower-triangular with blocks ``dt I``.

Tightened convex inner approximations (Eq. (7)):

* obstacles: for every obstacle a separating halfspace computed around the *current* state ``z_t``
  (the radical axis of the robot disk and the inflated obstacle disk, shifted by the robot radius
  towards the robot). It is fixed over the horizon,
  so that the constraint set ``U(z_t, d)`` of Eq. (12) is polyhedral, as stated in Sec. III-C.
* walls: the four arena walls, tightened by ``wall_margin + extra margin``.
* inputs: the box ``|u_{k,j}| <= v_max``.

All halfspace rows are normalised to unit-norm normals so that constraint values are in metres
(state rows) or m/s (input rows); this does not change the feasible set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def prediction_matrix(horizon: int, dt: float, dim: int = 2) -> np.ndarray:
    """``S`` such that ``z_hat_{1..N} = 1 (x) z_t + S u`` (shape ``(dim N, dim N)``)."""
    S = np.zeros((dim * horizon, dim * horizon))
    eye = dt * np.eye(dim)
    for k in range(horizon):            # row block -> state z_hat_{k+1}
        for i in range(k + 1):          # depends on inputs u_0..u_k
            S[dim * k:dim * (k + 1), dim * i:dim * (i + 1)] = eye
    return S


def obstacle_halfspaces(z: np.ndarray, obstacles: np.ndarray, robot_radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Separating halfspaces ``a_i^T p <= b_i`` between the robot (radius ``r``) at ``z`` and each obstacle.

    For obstacle centre ``o_i`` and radius ``r_i`` let ``d = ||o_i - z||`` and ``n = (o_i - z)/d``.
    The radical axis of the two circles is at distance ``s = d/2 - (r_i^2 - r^2)/(2d)`` from ``z`` along ``n``.
    The constraint keeps the robot centre at least ``r`` behind that line:

        n^T p <= n^T z + s - r.

    The current state ``z`` satisfies the constraint iff ``d >= r + r_i`` (no overlap), and every point
    of the halfspace keeps the robot disk disjoint from the obstacle disk, so the set is an inner
    approximation of the collision-free set (``Z_tilde subset Z``).
    """
    z = np.asarray(z, dtype=np.float64)
    obstacles = np.asarray(obstacles, dtype=np.float64).reshape(-1, 3)
    diff = obstacles[:, :2] - z[None, :]
    d = np.linalg.norm(diff, axis=1)
    d = np.maximum(d, 1e-9)
    n = diff / d[:, None]
    r_i = obstacles[:, 2]
    s = 0.5 * d - (r_i**2 - robot_radius**2) / (2.0 * d)
    b = n @ z + s - robot_radius
    return n, b


def wall_halfspaces(x_min: float, x_max: float, y_min: float, y_max: float, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Four halfspaces keeping the robot centre inside the arena shrunk by ``margin``."""
    A = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    b = np.array([-(x_min + margin), x_max - margin, -(y_min + margin), y_max - margin])
    return A, b


@dataclass
class StackedConstraints:
    """All constraints of problem (11) in the compact form ``H u + g <= 0`` (Eq. (12)).

    ``kind`` labels each row (0 = input box, 1 = obstacle, 2 = wall) for diagnostics.
    """

    H: np.ndarray
    g: np.ndarray
    kind: np.ndarray

    @property
    def n_constraints(self) -> int:
        return int(self.H.shape[0])

    def values(self, u: np.ndarray) -> np.ndarray:
        return self.H @ u + self.g


def stack_constraints(
    z: np.ndarray,
    S: np.ndarray,
    horizon: int,
    v_max: float,
    obstacles: np.ndarray,
    robot_radius: float,
    walls: tuple[float, float, float, float, float] | None,
) -> StackedConstraints:
    """Build ``H u + g <= 0`` for the horizon (states ``z_hat_1..z_hat_N``, inputs ``u_0..u_{N-1}``)."""
    dim = 2
    rows_H, rows_g, kinds = [], [], []

    # input box: e_j^T u_k - v_max <= 0 and -e_j^T u_k - v_max <= 0
    for k in range(horizon):
        for j in range(dim):
            for sign in (1.0, -1.0):
                h = np.zeros(dim * horizon)
                h[dim * k + j] = sign
                rows_H.append(h)
                rows_g.append(-v_max)
                kinds.append(0)

    # state halfspaces, applied to every predicted state
    A_parts, b_parts, k_parts = [], [], []
    if obstacles is not None and len(obstacles) > 0:
        A_o, b_o = obstacle_halfspaces(z, obstacles, robot_radius)
        A_parts.append(A_o)
        b_parts.append(b_o)
        k_parts.append(np.full(len(b_o), 1))
    if walls is not None:
        A_w, b_w = wall_halfspaces(*walls)
        A_parts.append(A_w)
        b_parts.append(b_w)
        k_parts.append(np.full(len(b_w), 2))
    H = np.asarray(rows_H)
    g = np.asarray(rows_g, dtype=np.float64)
    kind = np.asarray(kinds)
    if A_parts:
        A = np.vstack(A_parts)
        b = np.concatenate(b_parts)
        kk = np.concatenate(k_parts)
        H_state, g_state, kind_state = [], [], []
        for k in range(horizon):
            Sk = S[dim * k:dim * (k + 1), :]          # z_hat_{k+1} = z + Sk u
            H_state.append(A @ Sk)
            g_state.append(A @ z - b)
            kind_state.append(kk)
        H = np.vstack([H] + H_state)
        g = np.concatenate([g] + g_state)
        kind = np.concatenate([kind] + kind_state)
    return StackedConstraints(H=H, g=g, kind=kind)
