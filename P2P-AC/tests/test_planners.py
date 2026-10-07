"""Planner tests: anytime feasibility (Definition 1(i)), convergence (Corollary 1), constraint construction and
the action map of Sec. VI-A."""

import numpy as np
import pytest

from p2pac.planners import PlannerActionMap, ReapConfig, make_planner, world_to_body
from p2pac.planners.constraints import obstacle_halfspaces, prediction_matrix
from p2pac.planners.reference import converged_barrier_solution, exact_mpc_solution
from p2pac.task import NavigationTask, TaskConfig

TASK = TaskConfig()


def _states(n=20, seed=0):
    task = NavigationTask(TASK)
    rng = np.random.default_rng(seed)
    extra = [np.array([0.5, 1.0]), np.array([-0.6, 0.4]), np.array([0.05, -0.5]), np.array([0.9, 2.0])]
    return [task.sample_start(rng) for _ in range(n)] + extra


def test_prediction_matrix_single_integrator():
    S = prediction_matrix(3, 0.1)
    u = np.array([1.0, 0.0, 0.0, 2.0, -1.0, 0.0])
    z = S @ u
    np.testing.assert_allclose(z, [0.1, 0.0, 0.1, 0.2, 0.0, 0.2])


def test_obstacle_halfspace_separates_disks():
    z = np.array([0.0, -1.0])
    obs = np.array([[0.0, 0.15, 0.30]])
    n, b = obstacle_halfspaces(z, obs, robot_radius=0.3)
    assert n @ z <= b[0]                                    # current state feasible (no overlap)
    rng = np.random.default_rng(1)
    for _ in range(2000):                                   # every feasible point keeps the disks disjoint
        p = z + rng.uniform(-2, 2, 2)
        if n @ p <= b[0]:
            assert np.linalg.norm(p - obs[0, :2]) >= 0.3 + 0.3 - 1e-9


@pytest.mark.parametrize("iterations", [0, 1, 2, 5, 20, 100])
def test_anytime_feasibility_every_budget(iterations):
    """Definition 1(i): for every z in S_P the returned action is feasible, whatever the budget.

    Spawn states within ``planner_wall_margin`` of the task wall margin are outside S_P (Assumption 1 is
    violated there; docs/paper_vs_code.md): they must be flagged, and the planner must still return an action.
    """
    pl = make_planner("reap", ReapConfig(iterations=iterations), TASK)
    n_inside = 0
    for z in _states(40):
        pl.reset()
        out = pl.plan(z)
        assert np.all(np.abs(out.u_world) <= pl.cfg.v_max + 1e-9)
        if out.outside_domain:
            assert out.info["n_relaxed"] > 0
            continue
        n_inside += 1
        assert out.max_constraint < 1e-9, f"infeasible planner action at z={z}, budget={iterations}"
    assert n_inside >= 30


def test_gap_decreases_with_budget():
    pl = make_planner("reap", ReapConfig(), TASK)
    budgets = [1, 20, 200]
    gaps = {k: [] for k in budgets}
    for z in _states(10):
        u_star = exact_mpc_solution(pl, z)["u0"]
        for k in budgets:
            pl.reset()
            gaps[k].append(np.linalg.norm(pl.plan(z, iterations=k).u_world - u_star))
    med = [np.median(gaps[k]) for k in budgets]
    assert med[0] > med[1] > med[2]
    assert med[2] < 0.1


def test_converged_flow_is_close_to_mpc_optimum():
    pl = make_planner("reap", ReapConfig(), TASK)
    diffs = []
    for z in _states(8):
        diffs.append(np.linalg.norm(converged_barrier_solution(pl, z, 4000) - exact_mpc_solution(pl, z)["u0"]))
    assert np.median(diffs) < 0.05      # epsilon_omega of Corollary 1 is small for omega = 200


def test_outside_domain_is_flagged_and_relaxed():
    pl = make_planner("reap", ReapConfig(), TASK)
    z = np.array([0.0, -0.42])          # inside the planner's tightening margin of the entry obstacle
    out = pl.plan(z)
    assert out.outside_domain
    assert out.info["n_relaxed"] > 0


def test_world_to_body_fixed_heading():
    np.testing.assert_allclose(world_to_body(np.array([0.2, 0.5]), np.pi / 2), [0.5, -0.2], atol=1e-12)
    amap = PlannerActionMap(np.pi / 2, [-0.7, -0.7], [0.7, 0.7])

    class Out:
        u_world = np.array([0.9, 0.1])

    u, clipped = amap(Out())
    np.testing.assert_allclose(u, [0.1, -0.7])
    assert clipped
