"""The sparse-reward goal-reaching navigation task of Sec. III-A / VI-A (backend independent).

This module contains everything that defines the MDP *except* the robot dynamics:

* the arena, the obstacles and the goal (Sec. VI-A),
* the goal set ``G`` and the failure set ``F`` (Sec. III-A),
* the distribution of initial states ``rho_0`` (rejection sampling of the spawn position),
* the reward of Eq. (1),
* per-episode metrics (success, crash, path ratio, duration, mean speed, ...).

Both simulation backends (:mod:`p2pac.envs.isaac_go2` and :mod:`p2pac.envs.kinematic`) use this
module, so the task logic that runs on the cluster is exactly the logic exercised by the unit tests.

Conventions
-----------
* Positions are planar world coordinates ``(x, y)`` in metres.
* The policy action ``u = [v_x, v_y]`` is a *body-frame* velocity command in m/s (Sec. VI-A).
* ``dt`` is the high-level control period (0.02 s, i.e. 50 Hz).
"""

from __future__ import annotations

import dataclasses
import enum
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from p2pac.layout import DEFAULT_OBSTACLES  # default layout of Sec. VI-A


class Outcome(enum.IntEnum):
    """Episode outcome codes (stored as integers in every log file)."""

    RUNNING = 0
    GOAL = 1                 # s_{t+1} in G
    COLLISION_OBSTACLE = 2   # in F: geometric obstacle collision
    OUT_OF_BOUNDS = 3        # in F: wall margin violated
    FALL = 4                 # in F: trunk height below threshold
    BASE_CONTACT = 5         # in F: physical contact of the trunk (simulator contact sensor)
    TIMEOUT = 6              # truncation after T_max steps (NOT a termination)

    @property
    def is_failure(self) -> bool:
        return self in (Outcome.COLLISION_OBSTACLE, Outcome.OUT_OF_BOUNDS, Outcome.FALL, Outcome.BASE_CONTACT)

    @property
    def is_terminal(self) -> bool:
        """``chi_{t+1}`` of Sec. III-A: one at termination (goal or failure), zero at truncation."""
        return self == Outcome.GOAL or self.is_failure


@dataclass
class TaskConfig:
    """Geometry, initial-state distribution and reward of the navigation MDP.

    Every field is exposed on the command line by :mod:`p2pac.config`.
    """

    # --- goal set G -------------------------------------------------------------------------
    goal_x: float = 0.0
    goal_y: float = 2.8
    goal_tolerance: float = 0.3                     # episode terminates in G when ||p - p_goal|| < tol

    # --- arena and failure set F ----------------------------------------------------------
    x_min: float = -2.0
    x_max: float = 2.0
    y_min: float = -2.0
    y_max: float = 3.5
    wall_margin: float = 0.35                       # F contains states closer than this to a wall
    obstacles: tuple[tuple[float, float, float], ...] = DEFAULT_OBSTACLES
    robot_radius: float = 0.30                      # planar footprint radius of the trunk
    leg_reach_margin: float = 0.02                  # extra clearance for the legs
    fall_height: float = 0.10                       # F contains trunk heights below this (fall)

    # --- episode -------------------------------------------------------------------------------
    max_episode_steps: int = 8000                   # T_max (truncation)
    dt: float = 0.02                                # control period (s)

    # --- initial-state distribution rho_0 -----------------------------------------------------
    spawn_y_max: float = 0.0                        # spawn in the lower part of the arena: y <= spawn_y_max
    spawn_clearance: float = 0.5                    # extra clearance from obstacles at spawn (beyond the collision radius)
    spawn_yaw_deg: float = 90.0                     # robot faces +y (body x-axis = world +y)
    spawn_max_tries: int = 100_000

    # --- reward, Eq. (1) ---------------------------------------------------------------------
    step_cost: float = 1.0                          # w_s
    action_cost: float = 0.02                       # w_u (on ||u||^2 of the executed action)
    goal_reward: float = 100.0                      # r_G
    failure_cost: float = 200.0                     # r_F

    def __post_init__(self) -> None:
        self.obstacles = tuple(tuple(float(v) for v in o) for o in self.obstacles)
        if not (self.x_min < self.x_max and self.y_min < self.y_max):
            raise ValueError("invalid arena bounds")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")

    # Convenience accessors ----------------------------------------------------------------------
    @property
    def goal(self) -> np.ndarray:
        return np.array([self.goal_x, self.goal_y], dtype=np.float64)

    @property
    def obstacle_array(self) -> np.ndarray:
        return np.asarray(self.obstacles, dtype=np.float64).reshape(-1, 3)

    @property
    def collision_radius(self) -> np.ndarray:
        """Centre-to-centre distance below which the robot collides with each obstacle."""
        return self.obstacle_array[:, 2] + self.robot_radius + self.leg_reach_margin

    @property
    def spawn_yaw(self) -> float:
        return math.radians(self.spawn_yaw_deg)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["obstacles"] = [list(o) for o in self.obstacles]
        return d


@dataclass
class StepResult:
    """Reward/termination outcome of one transition."""

    reward: float
    outcome: Outcome
    terminated: bool
    truncated: bool
    terms: dict = field(default_factory=dict)


class NavigationTask:
    """Pure-numpy implementation of the task logic shared by all backends."""

    def __init__(self, cfg: TaskConfig):
        self.cfg = cfg
        self._obs = cfg.obstacle_array
        self._coll = cfg.collision_radius

    # ------------------------------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------------------------------
    def distance_to_goal(self, xy: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(xy, dtype=np.float64)[:2] - self.cfg.goal))

    def obstacle_clearance(self, xy: np.ndarray) -> float:
        """Signed distance to the nearest inflated obstacle (negative = collision)."""
        if self._obs.shape[0] == 0:
            return math.inf
        d = np.linalg.norm(self._obs[:, :2] - np.asarray(xy, dtype=np.float64)[None, :2], axis=1)
        return float(np.min(d - self._coll))

    def wall_clearance(self, xy: np.ndarray) -> float:
        """Signed distance to the nearest wall margin (negative = out of bounds)."""
        c = self.cfg
        x, y = float(xy[0]), float(xy[1])
        return min(
            x - (c.x_min + c.wall_margin),
            (c.x_max - c.wall_margin) - x,
            y - (c.y_min + c.wall_margin),
            (c.y_max - c.wall_margin) - y,
        )

    def clearance(self, xy: np.ndarray) -> float:
        return min(self.obstacle_clearance(xy), self.wall_clearance(xy))

    def in_obstacle(self, xy: np.ndarray) -> bool:
        if self._obs.shape[0] == 0:
            return False
        d = np.linalg.norm(self._obs[:, :2] - np.asarray(xy, dtype=np.float64)[None, :2], axis=1)
        return bool(np.any(d <= self._coll))

    def out_of_bounds(self, xy: np.ndarray) -> bool:
        c = self.cfg
        x, y = float(xy[0]), float(xy[1])
        return (
            x < c.x_min + c.wall_margin
            or x > c.x_max - c.wall_margin
            or y < c.y_min + c.wall_margin
            or y > c.y_max - c.wall_margin
        )

    # ------------------------------------------------------------------------------------------
    # Initial-state distribution rho_0
    # ------------------------------------------------------------------------------------------
    def is_valid_start(self, xy: np.ndarray) -> bool:
        c = self.cfg
        x, y = float(xy[0]), float(xy[1])
        if not (c.x_min + c.wall_margin <= x <= c.x_max - c.wall_margin):
            return False
        if not (c.y_min + c.wall_margin <= y <= c.spawn_y_max):
            return False
        if self._obs.shape[0] == 0:
            return True
        d = np.linalg.norm(self._obs[:, :2] - np.array([[x, y]]), axis=1)
        return bool(np.all(d > self._coll + c.spawn_clearance))

    def sample_start(self, rng: np.random.Generator) -> np.ndarray:
        """Rejection sampling of the spawn position: draw x, then y, uniformly; reject starts too close to an obstacle."""
        c = self.cfg
        for _ in range(c.spawn_max_tries):
            x = rng.uniform(c.x_min + c.wall_margin, c.x_max - c.wall_margin)
            y = rng.uniform(c.y_min + c.wall_margin, c.spawn_y_max)
            if self.is_valid_start((x, y)):
                return np.array([x, y], dtype=np.float64)
        raise RuntimeError("could not sample a valid start position; check spawn_clearance/spawn_y_max")

    def sample_starts(self, n: int, seed: int) -> np.ndarray:
        """A fixed, reproducible set of ``n`` start positions (used by the evaluation protocol)."""
        rng = np.random.default_rng(seed)
        return np.stack([self.sample_start(rng) for _ in range(n)], axis=0)

    # ------------------------------------------------------------------------------------------
    # Termination and reward
    # ------------------------------------------------------------------------------------------
    def classify(self, xy: np.ndarray, *, height: float, contact: bool, step_count: int) -> Outcome:
        """Classify the next state ``s_{t+1}``.

        Failure takes precedence over the goal. Geometrically the goal region and the failure set are
        disjoint for the default layout, so the precedence only matters for physical events (contact, fall).
        """
        if contact:
            return Outcome.BASE_CONTACT
        if height < self.cfg.fall_height:
            return Outcome.FALL
        if self.in_obstacle(xy):
            return Outcome.COLLISION_OBSTACLE
        if self.out_of_bounds(xy):
            return Outcome.OUT_OF_BOUNDS
        if self.distance_to_goal(xy) < self.cfg.goal_tolerance:
            return Outcome.GOAL
        if step_count >= self.cfg.max_episode_steps:
            return Outcome.TIMEOUT
        return Outcome.RUNNING

    def reward(self, outcome: Outcome, action: np.ndarray) -> StepResult:
        """Reward of Eq. (1): ``r = r0(s, u) + r_G 1[s' in G] - r_F 1[s' in F]``.

        ``r0(s, u) = -w_s - w_u ||u||^2`` with ``u`` the *executed* body-frame command. ``r0`` is paid at every
        step, including the terminal one.
        """
        c = self.cfg
        u = np.asarray(action, dtype=np.float64)
        r0 = -c.step_cost - c.action_cost * float(u @ u)
        terms = {"r0": r0, "r_goal": 0.0, "r_fail": 0.0}
        if outcome == Outcome.GOAL:
            terms["r_goal"] = c.goal_reward
        elif outcome.is_failure:
            terms["r_fail"] = -c.failure_cost
        reward = r0 + terms["r_goal"] + terms["r_fail"]
        return StepResult(float(reward), outcome, outcome.is_terminal, outcome == Outcome.TIMEOUT, terms)


@dataclass
class EpisodeStats:
    """Accumulates per-episode metrics (Sec. VI-C)."""

    start_xy: np.ndarray
    dt: float
    gamma: float = 0.99
    steps: int = 0
    ret: float = 0.0
    disc_ret: float = 0.0
    path_length: float = 0.0
    min_clearance: float = math.inf
    planner_steps: int = 0
    max_abs_action: float = 0.0
    sum_speed_cmd: float = 0.0
    yaw_start: float = math.nan
    max_abs_yaw_drift: float = 0.0
    internal_resets: int = 0

    def update(self, xy_prev: np.ndarray, xy_next: np.ndarray, reward: float, action: np.ndarray,
               clearance: float, executed_by_planner: bool = False, yaw: float = math.nan) -> None:
        self.disc_ret += (self.gamma**self.steps) * reward
        self.ret += reward
        self.steps += 1
        self.path_length += float(np.linalg.norm(np.asarray(xy_next)[:2] - np.asarray(xy_prev)[:2]))
        self.min_clearance = min(self.min_clearance, clearance)
        self.planner_steps += int(executed_by_planner)
        a = np.asarray(action, dtype=np.float64)
        self.max_abs_action = max(self.max_abs_action, float(np.max(np.abs(a))))
        self.sum_speed_cmd += float(np.linalg.norm(a))
        if not math.isnan(yaw):
            if math.isnan(self.yaw_start):
                self.yaw_start = yaw
            drift = abs((yaw - self.yaw_start + math.pi) % (2 * math.pi) - math.pi)
            self.max_abs_yaw_drift = max(self.max_abs_yaw_drift, drift)

    def summary(self, task: NavigationTask, outcome: Outcome, final_xy: np.ndarray) -> dict:
        """Metrics of Sec. VI-C. Efficiency metrics are reported for every episode; tables use the
        successful ones only (path ratio / duration / speed are only meaningful when the goal is reached)."""
        direct = float(np.linalg.norm(task.cfg.goal - np.asarray(self.start_xy)[:2]))
        duration = self.steps * self.dt
        return {
            "outcome": int(outcome),
            "outcome_name": outcome.name.lower(),
            "success": int(outcome == Outcome.GOAL),
            "crash": int(outcome.is_failure),
            "timeout": int(outcome == Outcome.TIMEOUT),
            "steps": self.steps,
            "return": self.ret,
            "discounted_return": self.disc_ret,
            "path_length": self.path_length,
            "path_ratio": self.path_length / direct if direct > 0 else math.nan,
            "duration_s": duration,
            "mean_speed": self.path_length / duration if duration > 0 else math.nan,
            "mean_command_speed": self.sum_speed_cmd / max(self.steps, 1),
            "min_clearance": self.min_clearance,
            "final_distance": task.distance_to_goal(final_xy),
            "start_x": float(self.start_xy[0]),
            "start_y": float(self.start_xy[1]),
            "planner_steps": self.planner_steps,
            "max_abs_action": self.max_abs_action,
            "max_abs_yaw_drift": self.max_abs_yaw_drift,
            "internal_resets": self.internal_resets,
        }


def parse_obstacles(values: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    """Parse a flat list ``x1 y1 r1 x2 y2 r2 ...`` into obstacle tuples."""
    vals = [float(v) for v in values]
    if len(vals) % 3 != 0:
        raise ValueError("obstacles must be given as triplets x y r")
    return tuple(tuple(vals[i:i + 3]) for i in range(0, len(vals), 3))
