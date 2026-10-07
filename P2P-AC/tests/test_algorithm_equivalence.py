"""P2P-AC with the planner disabled must be *bitwise* identical to Stable-Baselines3 (Proposition 3).

This is the strongest available check that ``P2PSAC/P2PTD3/P2PDDPG.train`` reproduce SB3 exactly, so that for
``t >= T_r`` (``kappa_t = 0``) the update is the underlying algorithm and nothing else.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from stable_baselines3 import DDPG, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

from p2pac.algos import P2PDDPG, P2PSAC, P2PTD3, AnnotatedReplayBuffer
from p2pac.algos.buffers import PlannerReplayBuffer
from p2pac.algos.schedule import P2PConfig

STEPS = 600


def _params(model):
    return th.cat([p.detach().flatten().cpu() for p in model.policy.parameters()])


def _make(cls, rb_cls=None, rb_kwargs=None, extra=None, **kw):
    env = gym.make("Pendulum-v1")
    kwargs = dict(learning_starts=100, batch_size=64, seed=3, device="cpu", policy_kwargs=dict(net_arch=[32, 32]))
    kwargs.update(extra or {})
    if rb_cls is not None:
        kwargs.update(replay_buffer_class=rb_cls, replay_buffer_kwargs=rb_kwargs or {})
    return cls("MlpPolicy", env, **kwargs, **kw)


@pytest.mark.parametrize(
    "ref_cls,p2p_cls,extra",
    [
        (SAC, P2PSAC, {}),
        (TD3, P2PTD3, {"action_noise": NormalActionNoise(np.zeros(1), 0.1 * np.ones(1))}),
        (DDPG, P2PDDPG, {"action_noise": NormalActionNoise(np.zeros(1), 0.1 * np.ones(1))}),
    ],
)
def test_disabled_p2p_equals_sb3(ref_cls, p2p_cls, extra):
    ref = _make(ref_cls, extra=dict(extra))
    ref.learn(STEPS)
    if "action_noise" in extra:
        extra = {"action_noise": NormalActionNoise(np.zeros(1), 0.1 * np.ones(1))}
    ours = _make(p2p_cls, rb_cls=AnnotatedReplayBuffer, extra=extra, p2p_config=P2PConfig.disabled())
    ours.learn(STEPS)
    assert th.equal(_params(ref), _params(ours)), "P2P-AC with the planner disabled deviates from SB3"


def test_dual_buffer_with_empty_planner_buffer_equals_sb3():
    """With D_P empty the dual buffer samples exactly like SB3's buffer (fallback rule of Sec. V-B)."""
    ref = _make(SAC)
    ref.learn(STEPS)
    ours = _make(P2PSAC, rb_cls=PlannerReplayBuffer, rb_kwargs=dict(planner_capacity=1000, planner_fraction=0.5),
                 p2p_config=P2PConfig(t_p=0, t_d=0, kappa0=0.0, execute_planner=False), buffer_size=1_001_000)
    ours.learn(STEPS)
    assert th.equal(_params(ref), _params(ours))
