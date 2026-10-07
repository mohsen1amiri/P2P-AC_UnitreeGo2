"""Anytime-feasible MPC planner (REAP) as described in Sec. IV of the paper.

Problem (11) for the planar single integrator of Sec. VI-A is solved approximately by evolving the
primal-dual gradient flow (14) of the modified barrier function (13)::

    B(z, d, u, lam) = J(z, u, d) - sum_i lam_i * log(1 - beta * (eta_i^T u + g_i + 1/omega)),
    du/drho   = -zeta * grad_u B,
    dlam/drho =  zeta * (grad_lam B + Psi),

and returning the first input of the sequence reached after the computation budget (Eq. (15)).

Discretisation and computation budget
-------------------------------------
The flow is integrated in discrete time; the computation budget is the number of iterations ``n_it``
(``iterations``), i.e. the virtual time ``t_cpu = n_it * delta_rho``. An iteration count (instead of a
wall-clock deadline) makes runs deterministic and machine independent; the measured wall-clock time of
every query is logged, so the budget can also be reported in milliseconds (Fig. 3). ``time_budget_ms``
optionally adds a hard wall-clock cap.

One iteration is
  1. primal step  ``u <- u - h_p * grad_u B(u, lam)``   (``step_size``), safeguarded by backtracking (see below);
  2. dual step    ``lam <- max(0, lam + h_d * (grad_lam B(u, lam) + Psi))``  (``dual_step_size``),
     evaluated at the *new* primal iterate (Arrow-Hurwicz / Gauss-Seidel, ``dual_update="gauss_seidel"``)
     or at the old one (``"jacobi"``, plain forward Euler of (14)).

Why safeguards are needed (validated in tests/test_planners.py and docs/planner_validation.md): plain forward Euler of the saddle
flow with a fixed step is not stable once a constraint becomes active (the barrier Hessian scales with
``beta * mu``), and it can produce limit cycles instead of the exponential convergence of Proposition 1.
The backtracking safeguard halves the primal step until (i) all constraints are strictly satisfied and the
barrier is defined, and (ii) if ``armijo``, ``B(., lam)`` decreases sufficiently. Both are properties of the
continuous flow (invariance of ``U``, Proposition 2, and monotone decrease of ``B`` in ``u``), so the discrete
iteration is a faithful, convergent discretisation.

Anytime feasibility in discrete time
------------------------------------
Starting from a strictly feasible point, every iterate -- hence every returned action, *whatever the budget*
-- satisfies the constraints of the planning problem (Definition 1(i)). This is checked and logged at every
query (``max_constraint <= 0``).

Initial condition
-----------------
``init="cold"`` (default) starts every query from ``u = 0`` and ``lam = lambda_init * 1`` (Assumption 1: a fixed
feasible initial condition; any ``lam_0 >= 0`` is admissible, a small positive value switches the barrier on from
the first iteration). ``u = 0`` keeps the robot at ``z_t``; it is strictly feasible exactly when ``z_t`` lies in the
interior of the tightened state set, i.e. ``z_t in S_P``. If it is not (the true system can enter the tightening
margin because of the model mismatch ``w_t``), the violated rows are relaxed just enough to make the start strictly
feasible, the query is flagged ``outside_domain`` (logged: the empirical frequency of Assumption-1 violations) and
``max_constraint`` reports the violation of the unrelaxed constraints. ``init="warm"`` reuses the shifted previous
primal solution when it is strictly feasible (falls back to cold otherwise).

Terminal ingredients
--------------------
``terminal_cost="dare"`` uses the LQR (DARE) weight for ``(A, B, Q, R)``; ``"q"`` uses ``L_N = L_x``. No terminal
constraint set ``Omega`` is imposed: the constraints are re-linearised around ``z_t`` at every step and the goal is
usually more than ``N`` steps away, so a terminal set around the goal would make the problem infeasible almost
everywhere (docs/paper_vs_code.md).

Default parameters: ``beta = 100`` (Sec. VI-A); ``omega = 200``, ``h_p = 1``, ``h_d = 0.05`` and ``lambda_init = 0.01``
make the discrete iteration converge on the kinematic test model (docs/planner_validation.md). They are starting
values, not tuned on the Isaac task.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from p2pac.planners.base import PlannerOutput
from p2pac.planners.constraints import StackedConstraints, prediction_matrix, stack_constraints


@dataclass
class ReapConfig:
    """Hyper-parameters of the anytime-feasible planner (all exposed on the command line)."""

    horizon: int = 15                 # N
    dt: float = 0.02                  # model sampling time (single integrator z+ = z + dt u)
    v_max: float = 0.5                # input box |u_j| <= v_max (planner speed bound, Sec. VI-A)
    q_weight: float = 1.0             # L_x = q I
    r_weight: float = 0.5             # L_u = r I
    terminal_cost: str = "dare"       # "dare" | "q"
    beta: float = 100.0               # barrier parameter beta
    omega: float = 200.0              # tightening parameter omega (activation margin 1/omega)
    step_size: float = 1.0            # h_p = zeta * delta_rho (initial primal step, backtracked)
    dual_step_size: Optional[float] = 0.05  # h_d (None -> same as step_size)
    iterations: int = 20              # computation budget n_it (virtual time t_cpu = n_it * delta_rho)
    time_budget_ms: Optional[float] = None  # optional wall-clock cap (non-deterministic)
    init: str = "cold"                # "cold" | "warm"
    obstacle_margin: float = 0.05     # planner inflation beyond the collision radius (tightening of Z)
    wall_margin_extra: float = 0.05   # planner tightening of the walls beyond the task wall margin
    use_walls: bool = True
    max_backtracks: int = 40
    lambda_init: float = 0.01         # initial dual variables lam_0 >= 0 (any nonnegative value is admissible)
    armijo: bool = True               # sufficient-decrease safeguard on the primal step (see module docstring)
    dual_update: str = "gauss_seidel" # "gauss_seidel" (Arrow-Hurwicz, uses new u) | "jacobi" (simultaneous)
    recovery_margin: float = 1e-3     # strict-feasibility margin used when relaxing an infeasible start

    def __post_init__(self) -> None:
        if self.terminal_cost not in ("dare", "q"):
            raise ValueError("terminal_cost must be 'dare' or 'q'")
        if self.init not in ("cold", "warm"):
            raise ValueError("init must be 'cold' or 'warm'")
        if self.dual_update not in ("gauss_seidel", "jacobi"):
            raise ValueError("dual_update must be 'gauss_seidel' or 'jacobi'")
        if self.lambda_init < 0:
            raise ValueError("lambda_init must be >= 0")
        if self.iterations < 0:
            raise ValueError("iterations must be >= 0")


def dare_terminal_weight(dt: float, q: float, r: float) -> np.ndarray:
    """Solution ``P`` of the discrete algebraic Riccati equation for ``A = I, B = dt I, Q = qI, R = rI``."""
    from scipy.linalg import solve_discrete_are

    A = np.eye(2)
    B = dt * np.eye(2)
    return solve_discrete_are(A, B, q * np.eye(2), r * np.eye(2))


class ReapPlanner:
    """Paper-consistent REAP planner (see module docstring)."""

    name = "reap"

    def __init__(
        self,
        cfg: ReapConfig,
        goal: np.ndarray,
        obstacles: np.ndarray,
        robot_radius: float,
        leg_reach_margin: float,
        arena: tuple[float, float, float, float],
        wall_margin: float,
    ):
        self.cfg = cfg
        self.goal = np.asarray(goal, dtype=np.float64)
        obs = np.asarray(obstacles, dtype=np.float64).reshape(-1, 3).copy()
        # Planner obstacle radius = true radius + leg margin + planner tightening (robot radius is separate).
        obs[:, 2] = obs[:, 2] + leg_reach_margin + cfg.obstacle_margin
        self.obstacles = obs
        self.robot_radius = float(robot_radius)
        self.walls = (*arena, wall_margin + cfg.wall_margin_extra) if cfg.use_walls else None

        N = cfg.horizon
        self.S = prediction_matrix(N, cfg.dt)
        Q = cfg.q_weight * np.eye(2)
        P = dare_terminal_weight(cfg.dt, cfg.q_weight, cfg.r_weight) if cfg.terminal_cost == "dare" else Q.copy()
        self.P = P
        Qbar = np.zeros((2 * N, 2 * N))
        for k in range(N):
            Qbar[2 * k:2 * k + 2, 2 * k:2 * k + 2] = P if k == N - 1 else Q
        self.Qbar = Qbar
        self.Rbar = cfg.r_weight * np.eye(2 * N)
        # J(u) = (e + S u)^T Qbar (e + S u) + u^T Rbar u,  e = 1 (x) (z - goal)
        self.M = self.S.T @ Qbar @ self.S + self.Rbar
        self._u_prev: Optional[np.ndarray] = None
        self._lam_prev: Optional[np.ndarray] = None
        self._inv_omega = 1.0 / cfg.omega
        # Barrier domain: 1 - beta (c + 1/omega) > 0  <=>  c < 1/beta - 1/omega.
        self._c_dom = 1.0 / cfg.beta - self._inv_omega

    # ------------------------------------------------------------------------------------------
    def reset(self) -> None:
        self._u_prev = None
        self._lam_prev = None

    def constraints(self, z: np.ndarray) -> StackedConstraints:
        c = self.cfg
        return stack_constraints(z, self.S, c.horizon, c.v_max, self.obstacles, self.robot_radius, self.walls)

    def _linear_term(self, z: np.ndarray) -> np.ndarray:
        e = np.tile(np.asarray(z, dtype=np.float64)[:2] - self.goal, self.cfg.horizon)
        return self.S.T @ self.Qbar @ e

    def cost(self, z: np.ndarray, u: np.ndarray) -> float:
        e = np.tile(np.asarray(z, dtype=np.float64)[:2] - self.goal, self.cfg.horizon)
        zh = e + self.S @ u
        return float(zh @ self.Qbar @ zh + u @ self.Rbar @ u)

    # ------------------------------------------------------------------------------------------
    def plan(self, z: np.ndarray, iterations: Optional[int] = None) -> PlannerOutput:
        """One planner query ``pi_P^{(t_cpu)}(s)`` at ``z = h_z(s)``; ``iterations`` overrides the budget."""
        cfg = self.cfg
        t0 = time.perf_counter()
        z = np.asarray(z, dtype=np.float64)[:2]
        n_it = cfg.iterations if iterations is None else int(iterations)
        cons = self.constraints(z)
        H, g_orig = cons.H, cons.g
        n_u = 2 * cfg.horizon
        c_lim = min(0.0, self._c_dom)          # iterates must satisfy c < c_lim (feasible and in the domain)

        # ---- initial condition -------------------------------------------------------------
        u = np.zeros(n_u)
        lam = np.full(H.shape[0], cfg.lambda_init)
        used_warm = False
        if cfg.init == "warm" and self._u_prev is not None:
            u_w = np.concatenate([self._u_prev[2:], np.zeros(2)])
            # Only the primal sequence is warm-started; the multipliers restart from zero because the
            # constraint rows (one block per stage) do not align after the shift.
            if np.all(H @ u_w + g_orig < c_lim - cfg.recovery_margin):
                u = u_w
                used_warm = True
        g = g_orig.copy()
        c0 = H @ u + g
        bad = c0 >= c_lim - cfg.recovery_margin
        outside = bool(np.any(bad))
        if outside:
            # z_t not in S_P: relax the violated rows so that the start is strictly feasible.
            g[bad] -= c0[bad] - (c_lim - cfg.recovery_margin) + cfg.recovery_margin
        n_relaxed = int(np.count_nonzero(bad))

        # ---- primal-dual flow ---------------------------------------------------------------
        h_lin = self._linear_term(z)
        beta, inv_omega = cfg.beta, self._inv_omega
        eta_p = cfg.step_size
        eta_d = cfg.dual_step_size if cfg.dual_step_size is not None else cfg.step_size
        n_backtracks = 0
        n_stalls = 0
        it_done = 0
        deadline = None if cfg.time_budget_ms is None else t0 + cfg.time_budget_ms * 1e-3
        def barrier_value(u_: np.ndarray, c_: np.ndarray) -> float:
            s_ = 1.0 - beta * (c_ + inv_omega)
            return float(u_ @ self.M @ u_ + 2.0 * h_lin @ u_ - lam @ np.log(s_))

        for _ in range(n_it):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            c = H @ u + g
            s = 1.0 - beta * (c + inv_omega)                        # > 0 by invariance
            grad_u = 2.0 * (self.M @ u + h_lin) + H.T @ (beta * lam / s)
            if cfg.dual_update == "jacobi":
                grad_lam = -np.log(s)                                # dual gradient at the *current* u
            # ---- primal step: Euler step of du/drho = -zeta grad_u B, safeguarded by backtracking.
            #      Accepted only if the new point is strictly feasible, inside the barrier domain and
            #      (if armijo) decreases B(., lam) sufficiently.
            step = eta_p
            accepted = False
            b0 = barrier_value(u, c) if cfg.armijo else 0.0
            gn = float(grad_u @ grad_u)
            for _bt in range(cfg.max_backtracks + 1):
                u_new = u - step * grad_u
                c_new = H @ u_new + g
                if np.all(c_new < c_lim) and (not cfg.armijo or barrier_value(u_new, c_new) <= b0 - 1e-4 * step * gn):
                    accepted = True
                    break
                step *= 0.5
                n_backtracks += 1
            if accepted:
                u = u_new
            else:
                n_stalls += 1
            # ---- dual step: Euler step of dlam/drho = zeta (grad_lam B + Psi), projected on lam >= 0.
            if cfg.dual_update == "gauss_seidel":                   # Arrow-Hurwicz: use the new primal iterate
                grad_lam = -np.log(1.0 - beta * (H @ u + g + inv_omega))
            d_lam = np.where((lam > 0.0) | (grad_lam >= 0.0), grad_lam, 0.0)   # projection Psi
            lam = np.maximum(0.0, lam + eta_d * d_lam)
            it_done += 1

        self._u_prev, self._lam_prev = u.copy(), lam.copy()
        c_final = H @ u + g_orig
        wall = time.perf_counter() - t0
        active = np.count_nonzero(lam > 0.0)
        return PlannerOutput(
            u_world=u[:2].copy(),
            iterations=it_done,
            wall_time_s=wall,
            max_constraint=float(np.max(c_final)),
            outside_domain=outside,
            info={
                "n_relaxed": n_relaxed,
                "n_backtracks": n_backtracks,
                "n_stalls": n_stalls,
                "n_active": int(active),
                "cost": self.cost(z, u),
                "lambda_max": float(np.max(lam)) if lam.size else 0.0,
                "warm": int(used_warm),
            },
        )

    def describe(self) -> dict:
        d = asdict(self.cfg)
        d.update({"terminal_weight_P": self.P.tolist(), "planner_obstacles": self.obstacles.tolist(),
                  "robot_radius": self.robot_radius, "walls": list(self.walls) if self.walls else None})
        return d


def planner_horizon_sequence(planner: ReapPlanner, z: np.ndarray, iterations: int) -> np.ndarray:
    """Return the whole input sequence (N, 2) after ``iterations`` (used by the gap analysis)."""
    planner.reset()
    planner.plan(z, iterations=iterations)
    return planner._u_prev.reshape(-1, 2).copy()


def first_action_after(planner: ReapPlanner, z: np.ndarray, iterations: int) -> np.ndarray:
    planner.reset()
    return planner.plan(z, iterations=iterations).u_world


__all__ = ["ReapConfig", "ReapPlanner", "dare_terminal_weight", "first_action_after", "planner_horizon_sequence"]
