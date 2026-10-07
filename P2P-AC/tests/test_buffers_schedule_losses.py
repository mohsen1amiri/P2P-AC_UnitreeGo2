"""Dual buffer (Sec. V-B), removal schedule (Eq. (17)) and the logit-space imitation loss (Sec. V-C)."""

import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from p2pac.algos.buffers import AnnotatedReplayBuffer, PendingExtras, PlannerReplayBuffer
from p2pac.algos.schedule import P2PConfig

OBS = spaces.Box(-20, 20, (4,), np.float32)
ACT = spaces.Box(-0.7, 0.7, (2,), np.float32)


def _add(buf, t, to_planner, goal=False, planner_action=(0.1, -0.2)):
    buf.set_pending(PendingExtras(np.array([planner_action], np.float32), np.ones(1, np.float32),
                                  np.array([int(to_planner)], np.int8), t, to_planner))
    o = np.full((1, 4), t, np.float32)
    buf.add(o, o + 1, np.zeros((1, 2), np.float32), np.zeros(1), np.array([goal]), [{"outcome": 1 if goal else 0}])


def test_planner_buffer_never_overwritten():
    buf = PlannerReplayBuffer(110, OBS, ACT, device="cpu", planner_capacity=10)
    for t in range(10):
        _add(buf, t, True)
    with pytest.raises(BufferError):
        _add(buf, 10, True)
    for t in range(10, 300):                 # D_O is FIFO and may wrap
        _add(buf, t, False)
    assert buf.planner.size() == 10 and buf.online.full
    np.testing.assert_array_equal(buf.planner.step_index[:10, 0], np.arange(10))


def test_symmetric_sampling_and_fallback():
    buf = PlannerReplayBuffer(1000, OBS, ACT, device="cpu", planner_capacity=100)
    for t in range(50):
        _add(buf, t, True, goal=(t == 7))
    s = buf.sample(64)
    assert float(s.from_planner_buffer.sum()) == 64          # D_O empty -> all from D_P
    for t in range(50, 200):
        _add(buf, t, False)
    s = buf.sample(65)
    assert float(s.from_planner_buffer.sum()) == 32          # floor(n_b / 2)
    assert s.observations.shape == (65, 4) and s.planner_actions.shape == (65, 2)
    assert buf.n_goal_transitions_planner() == 1


def test_annotated_buffer_stores_planner_action_and_mask():
    buf = AnnotatedReplayBuffer(10, OBS, ACT, device="cpu")
    _add(buf, 0, False, planner_action=(0.5, -1.0))
    s = buf.sample(4)
    np.testing.assert_allclose(s.planner_actions.numpy(), np.tile([0.5, -1.0], (4, 1)))
    assert float(s.planner_mask.min()) == 1.0


def test_schedule_eq17():
    c = P2PConfig(t_p=100, t_d=50, kappa0=10.0)
    assert c.t_r == 150
    assert c.kappa(0) == 10 and c.kappa(99) == 10 and c.kappa(125) == pytest.approx(5.0) and c.kappa(150) == 0
    assert [c.phase(t) for t in (0, 99, 100, 149, 150)] == [0, 0, 1, 1, 2]
    assert c.executes_planner(99) and not c.executes_planner(100)
    assert c.queries_planner(149) and not c.queries_planner(150)       # removed at T_r (Prop. 3 (i))
    assert c.routes_to_planner_buffer(99) and not c.routes_to_planner_buffer(100)
    immediate = P2PConfig(t_p=100, t_d=0, kappa0=10.0)
    assert immediate.kappa(99) == 10 and immediate.kappa(100) == 0
    no_exec = P2PConfig(t_p=100, t_d=50, execute_planner=False)
    assert not no_exec.executes_planner(0) and no_exec.queries_planner(0)


def test_logit_gradient_does_not_vanish_at_the_boundary():
    """Sec. V-C: d/dxi of the logit loss is (2/p)(xi - xi_dag), independent of |xi|; the action loss vanishes."""
    eps = 1e-4
    target = th.tensor([[0.9999, -0.9999]])                              # planner action on the box boundary
    xi_dag = th.atanh(target.clamp(-(1 - eps), 1 - eps))
    xi = th.tensor([[3.0, -3.0]], requires_grad=True)                    # saturated logit, still short of the target
    ((xi - xi_dag) ** 2).mean().backward()
    g_logit = xi.grad.clone()
    np.testing.assert_allclose(g_logit.numpy(), (2 / 2) * (xi - xi_dag).detach().numpy(), rtol=1e-6)
    xi.grad = None
    ((th.tanh(xi) - target) ** 2).mean().backward()
    g_action = xi.grad.clone()
    assert th.all(g_logit.abs() > 50 * g_action.abs())
