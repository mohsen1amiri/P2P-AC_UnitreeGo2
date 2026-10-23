"""Replay buffers of P2P-AC (Sec. V-B).

Every stored tuple is ``(s_t, u_t, r_t, s_{t+1}, chi_{t+1}, u_t^dagger, iota_t)`` plus bookkeeping
(who executed the action, whether ``s_{t+1}`` is in ``G``, the global step index).

* :class:`AnnotatedReplayBuffer` -- a Stable-Baselines3 ``ReplayBuffer`` that additionally stores the
  planner action and its mask. Its sampling consumes the global numpy RNG exactly like SB3's buffer, so an
  algorithm using it without a planner is bitwise identical to plain SB3 (tests/test_algorithm_equivalence.py).
* :class:`PlannerReplayBuffer` -- the dual buffer ``D = (D_P, D_O)``: ``D_P`` (capacity ``C_P``) stores every
  transition of the planner phase and is never overwritten; ``D_O`` (capacity ``C - C_P``) is FIFO. Each
  minibatch of size ``n_b`` draws ``floor(n_b * f)`` samples (``f = 1/2`` in the paper) uniformly with replacement
  from ``D_P`` and the rest from ``D_O``; if one buffer is empty the whole minibatch comes from the other.

Actions are stored in the normalised space ``[-1, 1]`` used by SB3 (``D_u^{-1}(u - u_c)``), *including* the planner
action ``u^dagger`` (converted with ``policy.scale_action`` before it is stored).
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer

GOAL_OUTCOME_CODE = 1  # p2pac.task.Outcome.GOAL


class AnnotatedSamples(NamedTuple):
    """Minibatch with the SB3 fields (incl. ``discounts``) and the P2P-AC fields."""

    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    discounts: Optional[th.Tensor]
    planner_actions: th.Tensor        # u^dagger in normalised units, shape (B, p)
    planner_mask: th.Tensor           # iota, shape (B, 1)
    from_planner_buffer: th.Tensor    # 1 if the row was drawn from D_P, shape (B, 1)
    goal: th.Tensor                   # 1 if s_{t+1} in G, shape (B, 1)


class PendingExtras(NamedTuple):
    """P2P-AC fields of the transition about to be stored (one entry per parallel env)."""

    planner_actions: np.ndarray        # (n_envs, p) normalised
    planner_mask: np.ndarray           # (n_envs,)
    executed_by_planner: np.ndarray    # (n_envs,)
    step_index: int
    to_planner_buffer: bool


class AnnotatedReplayBuffer(ReplayBuffer):
    """SB3 replay buffer + planner action, mask, executor flag, goal flag and step index."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device="auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        never_overwrite: bool = False,
    ):
        if optimize_memory_usage:
            raise ValueError("optimize_memory_usage is not supported by AnnotatedReplayBuffer")
        super().__init__(buffer_size, observation_space, action_space, device, n_envs,
                         optimize_memory_usage=False, handle_timeout_termination=handle_timeout_termination)
        self.planner_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.planner_mask = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.executed_by_planner = np.zeros((self.buffer_size, self.n_envs), dtype=np.int8)
        self.goal = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.step_index = np.full((self.buffer_size, self.n_envs), -1, dtype=np.int64)
        self.never_overwrite = bool(never_overwrite)
        self._pending: Optional[PendingExtras] = None
        self.is_planner_buffer = False

    # ------------------------------------------------------------------------------------------
    def set_pending(self, extras: PendingExtras) -> None:
        self._pending = extras

    def add(self, obs, next_obs, action, reward, done, infos) -> None:  # type: ignore[override]
        if self.never_overwrite and self.full:
            raise BufferError(
                "the planner buffer D_P is full and must never be overwritten; increase its capacity (C_P >= T_p * n_envs)"
            )
        pos = self.pos
        ext = self._pending
        self._pending = None
        if ext is not None:
            self.planner_actions[pos] = np.asarray(ext.planner_actions, dtype=np.float32).reshape(self.n_envs, self.action_dim)
            self.planner_mask[pos] = np.asarray(ext.planner_mask, dtype=np.float32).reshape(self.n_envs)
            self.executed_by_planner[pos] = np.asarray(ext.executed_by_planner, dtype=np.int8).reshape(self.n_envs)
            self.step_index[pos] = ext.step_index
        else:
            self.planner_actions[pos] = 0.0
            self.planner_mask[pos] = 0.0
            self.executed_by_planner[pos] = 0
            self.step_index[pos] = -1
        self.goal[pos] = np.array(
            [float(info.get("outcome", 0) == GOAL_OUTCOME_CODE) for info in infos], dtype=np.float32
        )
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> AnnotatedSamples:  # type: ignore[override]
        # Identical to stable_baselines3 ReplayBuffer._get_samples (same RNG call), plus the extra fields.
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        obs_t, act_t, nobs_t, done_t, rew_t = tuple(map(self.to_torch, data))
        n = len(batch_inds)
        return AnnotatedSamples(
            observations=obs_t,
            actions=act_t,
            next_observations=nobs_t,
            dones=done_t,
            rewards=rew_t,
            discounts=None,
            planner_actions=self.to_torch(self.planner_actions[batch_inds, env_indices, :]),
            planner_mask=self.to_torch(self.planner_mask[batch_inds, env_indices].reshape(-1, 1)),
            from_planner_buffer=self.to_torch(np.full((n, 1), float(self.is_planner_buffer), dtype=np.float32)),
            goal=self.to_torch(self.goal[batch_inds, env_indices].reshape(-1, 1)),
        )

    # ------------------------------------------------------------------------------------------
    def n_goal_transitions(self) -> int:
        n = self.size()
        return int(self.goal[:n].sum())

    def export_npz(self, path: str) -> None:
        """Save the stored transitions (used to snapshot D_P at T_p and to train the BC baseline)."""
        n = self.size()
        np.savez_compressed(
            path,
            observations=self.observations[:n],
            next_observations=self.next_observations[:n],
            actions=self.actions[:n],
            rewards=self.rewards[:n],
            dones=self.dones[:n],
            timeouts=self.timeouts[:n],
            planner_actions=self.planner_actions[:n],
            planner_mask=self.planner_mask[:n],
            executed_by_planner=self.executed_by_planner[:n],
            goal=self.goal[:n],
            step_index=self.step_index[:n],
        )


def _cat(a: Optional[th.Tensor], b: Optional[th.Tensor]) -> Optional[th.Tensor]:
    if a is None or b is None:
        return None
    return th.cat((a, b), dim=0)


class PlannerReplayBuffer(BaseBuffer):
    """Dual replay buffer ``D = (D_P, D_O)`` with symmetric sampling (Sec. V-B)."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device="auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        planner_capacity: int = 1_000_000,
        planner_fraction: float = 0.5,
    ):
        if optimize_memory_usage:
            raise ValueError("optimize_memory_usage is not supported")
        if not (0 < planner_capacity < buffer_size):
            raise ValueError("need 0 < planner_capacity (C_P) < buffer_size (C)")
        if not (0.0 <= planner_fraction < 1.0):
            raise ValueError("planner_fraction must be in [0, 1)")
        super().__init__(buffer_size, observation_space, action_space, device, n_envs)
        self.optimize_memory_usage = False
        self.handle_timeout_termination = handle_timeout_termination
        self.planner_capacity = int(planner_capacity)
        self.planner_fraction = float(planner_fraction)
        # SB3 sizes are per env: a buffer of size B holds B * n_envs transitions.
        self.planner = AnnotatedReplayBuffer(
            math.ceil(self.planner_capacity / n_envs), observation_space, action_space, device, n_envs,
            handle_timeout_termination=handle_timeout_termination, never_overwrite=True,
        )
        self.planner.is_planner_buffer = True
        self.online = AnnotatedReplayBuffer(
            math.ceil((buffer_size - self.planner_capacity) / n_envs), observation_space, action_space, device, n_envs,
            handle_timeout_termination=handle_timeout_termination,
        )
        self._route_to_planner = False
        self.last_batch_stats: dict = {}

    # ------------------------------------------------------------------------------------------
    def set_pending(self, extras: PendingExtras) -> None:
        self._route_to_planner = bool(extras.to_planner_buffer)
        (self.planner if self._route_to_planner else self.online).set_pending(extras)

    def add(self, obs, next_obs, action, reward, done, infos) -> None:  # type: ignore[override]
        target = self.planner if self._route_to_planner else self.online
        target.add(obs, next_obs, action, reward, done, infos)
        self._route_to_planner = False

    def size(self) -> int:
        return self.planner.size() + self.online.size()

    def reset(self) -> None:
        self.planner.reset()
        self.online.reset()

    def _get_samples(self, batch_inds, env=None):  # pragma: no cover - sampling is done by sample()
        raise NotImplementedError

    def sample(self, batch_size: int, env=None) -> AnnotatedSamples:  # type: ignore[override]
        n_p, n_o = self.planner.size(), self.online.size()
        if n_p == 0 and n_o == 0:
            raise RuntimeError("cannot sample from an empty buffer")
        if n_p == 0:
            batch = self.online.sample(batch_size, env)
            k_p = 0
        elif n_o == 0:
            batch = self.planner.sample(batch_size, env)
            k_p = batch_size
        elif self.planner_fraction == 0.0:
            batch = self.online.sample(batch_size, env)
            k_p = 0
        else:
            k_p = int(math.floor(batch_size * self.planner_fraction))
            s_p = self.planner.sample(k_p, env)
            s_o = self.online.sample(batch_size - k_p, env)
            batch = AnnotatedSamples(*(_cat(a, b) for a, b in zip(s_p, s_o)))
        self.last_batch_stats = {"k_planner": k_p, "n_planner": n_p, "n_online": n_o}
        return batch

    def n_goal_transitions_planner(self) -> int:
        return self.planner.n_goal_transitions()
