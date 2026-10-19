"""Evaluation protocol (Sec. VI-C).

* The evaluated controller acts **alone** -- for learners: the deterministic (mean) action of the policy; the
  planner is never queried, so planner-controlled episodes can never enter a reported success rate.
* Every evaluation uses the same fixed set of start positions, sampled once from ``rho_0`` with ``eval_seed``
  (``eval/starts.json``) -- identical for all methods and seeds (common random numbers).
* Evaluation episodes are never stored in the replay buffer and do not count as environment steps.
* Every episode is logged (outcome, return, path ratio, duration, speed, clearance, ...), as well as its
  trajectory, so that any statistic can be recomputed later without re-running.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Optional, Protocol

import numpy as np

from p2pac.planners.base import PlannerActionMap

EPISODE_FIELDS = [
    "round", "t", "eval_episode", "start_x", "start_y", "outcome", "outcome_name", "success", "crash", "timeout",
    "steps", "return", "discounted_return", "path_length", "path_ratio", "duration_s", "mean_speed",
    "mean_command_speed", "min_clearance", "final_distance", "max_abs_action", "max_abs_yaw_drift",
    "internal_resets", "planner_ms_mean", "planner_iterations_mean", "planner_max_constraint", "planner_outside_domain_frac",
]
SUMMARY_FIELDS = [
    "round", "t", "t_scheduled", "n_episodes", "success_rate", "crash_rate", "timeout_rate", "return_mean",
    "discounted_return_mean", "steps_mean", "path_ratio_mean_success", "duration_s_mean_success",
    "mean_speed_mean_success", "min_clearance_mean", "internal_resets", "wall_time_s", "policy_snapshot",
]


class Controller(Protocol):
    def reset(self) -> None: ...  # pragma: no cover
    def act(self, obs: np.ndarray) -> np.ndarray: ...  # pragma: no cover


class PolicyController:
    """Deterministic action of an SB3 model (or any object with an SB3-like ``predict``)."""

    def __init__(self, model: Any, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float64)

    def episode_diagnostics(self) -> dict:
        return {}


class PlannerController:
    """The planner acting alone (reference row of Table II; Assumption 2 estimates)."""

    def __init__(self, planner: Any, action_map: PlannerActionMap):
        self.planner = planner
        self.action_map = action_map
        self._ms: list[float] = []
        self._iters: list[int] = []
        self._cmax: list[float] = []
        self._outside: list[int] = []

    def reset(self) -> None:
        self.planner.reset()
        self._ms, self._iters, self._cmax, self._outside = [], [], [], []

    def act(self, obs: np.ndarray) -> np.ndarray:
        out = self.planner.plan(np.asarray(obs[:2], dtype=np.float64))
        self._ms.append(1e3 * out.wall_time_s)
        self._iters.append(out.iterations)
        self._cmax.append(out.max_constraint)
        self._outside.append(int(out.outside_domain))
        u, _ = self.action_map(out)
        return u

    def episode_diagnostics(self) -> dict:
        if not self._ms:
            return {}
        inside = [c for c, o in zip(self._cmax, self._outside) if not o]
        return {
            "planner_ms_mean": float(np.mean(self._ms)),
            "planner_iterations_mean": float(np.mean(self._iters)),
            "planner_max_constraint": float(np.max(inside)) if inside else math.nan,
            "planner_outside_domain_frac": float(np.mean(self._outside)),
        }


def run_episodes(env, controller: Controller, starts: np.ndarray, *, round_index: int, t: int,
                 record_trajectories: bool = True) -> tuple[list[dict], Optional[dict]]:
    """Run one evaluation round (one episode per start position)."""
    rows: list[dict] = []
    xy_all, act_all, offsets = [], [], [0]
    for i, start in enumerate(starts):
        obs, _ = env.reset(options={"start_xy": np.asarray(start, dtype=np.float64), "evaluation": True})
        controller.reset()
        xy = [np.asarray(env.state.xy, dtype=np.float32)]
        acts = []
        while True:
            a = controller.act(obs)
            obs, _, terminated, truncated, info = env.step(a)
            xy.append(np.asarray(env.state.xy, dtype=np.float32))
            acts.append(np.asarray(info["executed_action"], dtype=np.float32))
            if terminated or truncated:
                break
        row = dict(info["episode_metrics"])
        row.update({"round": round_index, "t": t, "eval_episode": i})
        diag = getattr(controller, "episode_diagnostics", lambda: {})()
        row.update(diag)
        rows.append(row)
        if record_trajectories:
            xy_all.append(np.stack(xy))
            act_all.append(np.stack(acts))
            offsets.append(offsets[-1] + len(xy))
    traj = None
    if record_trajectories:
        traj = {
            "xy": np.concatenate(xy_all),
            "actions": np.concatenate([np.vstack([a, np.full((1, 2), np.nan, np.float32)]) for a in act_all]),
            "offsets": np.asarray(offsets, dtype=np.int64),
            "outcome": np.asarray([r["outcome"] for r in rows], dtype=np.int8),
            "starts": np.asarray(starts, dtype=np.float32),
        }
    return rows, traj


def summarize(rows: list[dict], *, round_index: int, t: int, t_scheduled: Optional[int], wall: float,
              snapshot: str = "") -> dict:
    def mean(key, only_success=False):
        vals = [r[key] for r in rows if (not only_success or r["success"]) and r.get(key) is not None
                and not (isinstance(r[key], float) and math.isnan(r[key]))]
        return float(np.mean(vals)) if vals else math.nan

    n = len(rows)
    return {
        "round": round_index, "t": t, "t_scheduled": t_scheduled if t_scheduled is not None else t, "n_episodes": n,
        "success_rate": float(np.mean([r["success"] for r in rows])) if n else math.nan,
        "crash_rate": float(np.mean([r["crash"] for r in rows])) if n else math.nan,
        "timeout_rate": float(np.mean([r["timeout"] for r in rows])) if n else math.nan,
        "return_mean": mean("return"), "discounted_return_mean": mean("discounted_return"),
        "steps_mean": mean("steps"),
        "path_ratio_mean_success": mean("path_ratio", True), "duration_s_mean_success": mean("duration_s", True),
        "mean_speed_mean_success": mean("mean_speed", True), "min_clearance_mean": mean("min_clearance"),
        "internal_resets": int(sum(r.get("internal_resets", 0) for r in rows)), "wall_time_s": wall,
        "policy_snapshot": snapshot,
    }


def save_trajectories(directory: str, name: str, traj: Optional[dict]) -> Optional[str]:
    if traj is None:
        return None
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.npz")
    np.savez_compressed(path, **traj)
    return path


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
