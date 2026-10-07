"""Reference solutions used to measure the suboptimality gap of Corollary 1.

* :func:`exact_mpc_solution` -- minimiser ``u*`` of the MPC problem (11) (same constraints as the planner,
  including any relaxation applied when ``z_t`` is outside ``S_P``), solved with SLSQP.
* :func:`converged_barrier_solution` -- ``u^dagger``, the limit of the primal-dual flow (minimiser of the
  modified barrier problem), approximated by running the flow for many iterations.

Both are offline tools (``python -m p2pac.analysis.planner_gap``); they are never used during training.
"""

from __future__ import annotations

import numpy as np

from p2pac.planners.reap import ReapPlanner


def exact_mpc_solution(planner: ReapPlanner, z: np.ndarray, tol: float = 1e-10, maxiter: int = 500) -> dict:
    """Solve ``min_u J(z, u)  s.t.  H u + g <= 0`` with SLSQP and return the full sequence and diagnostics."""
    from scipy.optimize import minimize

    z = np.asarray(z, dtype=np.float64)[:2]
    cons = planner.constraints(z)
    H, g = cons.H, cons.g.copy()
    # Apply the same relaxation as the planner when the start is infeasible (outside S_P).
    c0 = g  # value at u = 0
    c_lim = min(0.0, planner._c_dom)
    bad = c0 >= c_lim - planner.cfg.recovery_margin
    if np.any(bad):
        g[bad] -= c0[bad] - (c_lim - planner.cfg.recovery_margin) + planner.cfg.recovery_margin
    h = planner._linear_term(z)
    M = planner.M

    def fun(u):
        return float(u @ M @ u + 2.0 * h @ u)

    def jac(u):
        return 2.0 * (M @ u + h)

    res = minimize(
        fun,
        np.zeros(H.shape[1]),
        jac=jac,
        method="SLSQP",
        constraints=[{"type": "ineq", "fun": lambda u: -(H @ u + g), "jac": lambda u: -H}],
        options={"ftol": tol, "maxiter": maxiter},
    )
    u = res.x
    return {
        "u_seq": u.reshape(-1, 2),
        "u0": u[:2].copy(),
        "success": bool(res.success),
        "message": str(res.message),
        "max_constraint": float(np.max(H @ u + g)),
        "relaxed": bool(np.any(bad)),
    }


def converged_barrier_solution(planner: ReapPlanner, z: np.ndarray, iterations: int = 5000) -> np.ndarray:
    """First input of the flow after a very large budget (approximates ``Proj_p(u^dagger)``)."""
    planner.reset()
    return planner.plan(z, iterations=iterations).u_world
