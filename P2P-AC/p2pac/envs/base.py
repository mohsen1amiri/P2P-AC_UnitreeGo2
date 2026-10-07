"""Generic goal-navigation environment: gymnasium API + task logic on top of a robot backend.

The environment implements the MDP of Sec. III-A / VI-A:

* state / observation ``s_t`` (``obs_mode="paper"``)::

      s_t = [x_t, y_t, x_goal, y_goal]                                   (Eq. (18))

  ``obs_mode="extended"`` appends ``[cos(yaw), sin(yaw), v_x^b, v_y^b]`` (heading and body velocity). This option
  is NOT used in the paper; it exists because the paper's 4-D state omits the robot heading and velocity, which
  makes the transition kernel only approximately Markov (docs/paper_vs_code.md).

* action ``u_t = [v_x, v_y]`` body-frame velocity command in ``[-vx_max, vx_max] x [-vy_max, vy_max]`` m/s;
* termination in ``G`` or ``F`` and truncation after ``T_max`` steps (handled here, not by gymnasium wrappers);
* reward of Eq. (1) (:meth:`p2pac.task.NavigationTask.reward`).

The robot itself is provided by a *backend* (Isaac Lab Go2 or the kinematic test model) implementing
:class:`RobotBackend`. ``reset(options={"start_xy": (x, y)})`` places the robot at a given start position, which is
how the evaluation protocol uses a fixed set of start states.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from p2pac.task import EpisodeStats, NavigationTask, Outcome


@dataclass
class RobotState:
    """Planar state reported by a backend after reset/step (all in the task frame)."""

    xy: np.ndarray
    height: float
    yaw: float
    vel_body: np.ndarray = field(default_factory=lambda: np.zeros(2))
    contact: bool = False
    internal_reset: bool = False      # backend-internal reset/teleport detected (must never happen)


class RobotBackend(Protocol):
    """Interface implemented by the simulation backends."""

    dt: float

    def reset_to(self, x: float, y: float, yaw: float) -> RobotState:  # pragma: no cover - protocol
        ...

    def step(self, vx: float, vy: float, yaw_rate: float) -> RobotState:  # pragma: no cover - protocol
        ...

    def render(self) -> Optional[np.ndarray]:  # pragma: no cover - protocol
        ...

    def close(self) -> None:  # pragma: no cover - protocol
        ...

    def describe(self) -> dict:  # pragma: no cover - protocol
        ...


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class GoalNavEnv(gym.Env):
    """Gymnasium environment of the navigation MDP (backend independent)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        task: NavigationTask,
        backend: RobotBackend,
        *,
        vx_max: float = 0.7,
        vy_max: float = 0.7,
        obs_mode: str = "paper",
        yaw_hold_gain: float = 0.0,
        gamma: float = 0.99,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        if obs_mode not in ("paper", "extended"):
            raise ValueError("obs_mode must be 'paper' or 'extended'")
        self.task = task
        self.backend = backend
        self.obs_mode = obs_mode
        self.yaw_hold_gain = float(yaw_hold_gain)
        self.gamma = float(gamma)
        self.render_mode = render_mode
        self.action_space = spaces.Box(
            low=np.array([-vx_max, -vy_max], dtype=np.float32),
            high=np.array([vx_max, vy_max], dtype=np.float32),
            dtype=np.float32,
        )
        obs_dim = 4 if obs_mode == "paper" else 8
        # Loose bounds, only used by space checks (observations are never clipped).
        self.observation_space = spaces.Box(low=-20.0, high=20.0, shape=(obs_dim,), dtype=np.float32)
        self._state: Optional[RobotState] = None
        self._stats: Optional[EpisodeStats] = None
        self._step_count = 0
        self._episode_index = -1

    # ------------------------------------------------------------------------------------------
    def _obs(self, st: RobotState) -> np.ndarray:
        g = self.task.cfg.goal
        base = [st.xy[0], st.xy[1], g[0], g[1]]
        if self.obs_mode == "extended":
            base += [math.cos(st.yaw), math.sin(st.yaw), st.vel_body[0], st.vel_body[1]]
        return np.asarray(base, dtype=np.float32)

    @property
    def state(self) -> Optional[RobotState]:
        return self._state

    @property
    def episode_index(self) -> int:
        return self._episode_index

    # ------------------------------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = options or {}
        if options.get("start_xy") is not None:
            start = np.asarray(options["start_xy"], dtype=np.float64)
        else:
            start = self.task.sample_start(self.np_random)
        st = self.backend.reset_to(float(start[0]), float(start[1]), self.task.cfg.spawn_yaw)
        self._state = st
        self._step_count = 0
        if not options.get("evaluation", False):     # evaluation episodes are not training episodes
            self._episode_index += 1
        self._stats = EpisodeStats(start_xy=start.copy(), dt=self.task.cfg.dt, gamma=self.gamma)
        info = {"start_xy": start.copy(), "yaw": st.yaw, "episode_index": self._episode_index}
        return self._obs(st), info

    def step(self, action: np.ndarray):
        assert self._state is not None and self._stats is not None, "call reset() first"
        u_req = np.asarray(action, dtype=np.float64).reshape(-1)[:2]
        u = np.clip(u_req, self.action_space.low, self.action_space.high)
        action_clipped = bool(np.any(np.abs(u - u_req) > 1e-6))
        yaw_rate = 0.0
        if self.yaw_hold_gain > 0.0:
            yaw_rate = self.yaw_hold_gain * wrap_angle(self.task.cfg.spawn_yaw - self._state.yaw)
        xy_prev = self._state.xy.copy()
        st = self.backend.step(float(u[0]), float(u[1]), float(yaw_rate))
        self._step_count += 1
        outcome = self.task.classify(st.xy, height=st.height, contact=st.contact, step_count=self._step_count)
        res = self.task.reward(outcome, u)
        clearance = self.task.clearance(st.xy)
        self._stats.update(xy_prev, st.xy, res.reward, u, clearance, yaw=st.yaw)
        if st.internal_reset:
            self._stats.internal_resets += 1
        self._state = st

        info: dict[str, Any] = {
            "outcome": int(outcome),
            "executed_action": u.astype(np.float32),
            "action_clipped": action_clipped,
            "clearance": clearance,
            "dist_to_goal": self.task.distance_to_goal(st.xy),
            "yaw": st.yaw,
            "height": st.height,
            "vel_body": np.asarray(st.vel_body, dtype=np.float32),
            "internal_reset": bool(st.internal_reset),
            "reward_terms": res.terms,
        }
        if res.terminated or res.truncated:
            metrics = self._stats.summary(self.task, outcome, st.xy)
            info["episode_metrics"] = metrics
            info["is_success"] = bool(outcome == Outcome.GOAL)   # read by SB3 (rollout/success_rate)
        return self._obs(st), res.reward, res.terminated, res.truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.backend.render()
        return None

    def close(self):
        self.backend.close()

    def describe(self) -> dict:
        return {
            "obs_mode": self.obs_mode,
            "action_low": self.action_space.low.tolist(),
            "action_high": self.action_space.high.tolist(),
            "yaw_hold_gain": self.yaw_hold_gain,
            "backend": self.backend.describe(),
        }
