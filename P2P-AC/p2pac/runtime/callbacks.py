"""The single Stable-Baselines3 callback that implements logging, evaluation, phase events and checkpoints.

Timing (Stable-Baselines3 2.7.1):

* ``_on_step`` is called after ``env.step`` and before the transition is stored. It logs the step and, at
  episode ends, the episode. Nothing heavy happens here.
* ``_on_rollout_start`` is called at the beginning of every rollout, i.e. after the previous gradient
  update (off-policy learners collect one step per rollout). All heavy operations happen here, where model,
  replay buffer and step counter are mutually consistent:

  1. flush learner statistics (every ``learner_log_every`` updates);
  2. phase events: at ``t = T_p`` the planner buffer ``D_P`` is exported and checkpointed (milestone), at
     ``t = T_r`` the removal of the planner is verified and logged (Proposition 3);
  3. periodic evaluation (every ``eval_every`` steps): inline mode evaluates at rollout/episode boundaries;
     snapshot mode only saves the frozen policy, leaving the training environment untouched;
  4. periodic checkpoint, and checkpoint + stop when a preemption signal was received.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from p2pac.runtime.checkpoint import CheckpointManager
from p2pac.runtime.evaluation import (EPISODE_FIELDS, SUMMARY_FIELDS, PolicyController, Timer, run_episodes,
                                      save_trajectories, summarize)
from p2pac.runtime.records import CSVLog, JSONLLog, StepLog
from p2pac.runtime.rundir import RunDir

TRAIN_EPISODE_FIELDS = [
    "episode", "t_start", "t_end", "phase_start", "phase_end", "kappa_end", "planner_executed_steps",
    "outcome", "outcome_name", "success", "crash", "timeout", "steps", "return", "discounted_return",
    "path_length", "path_ratio", "duration_s", "mean_speed", "mean_command_speed", "min_clearance",
    "final_distance", "start_x", "start_y", "max_abs_action", "max_abs_yaw_drift", "internal_resets", "aborted",
]


class RunCallback(BaseCallback):
    def __init__(
        self,
        run: RunDir,
        cfg,
        starts: np.ndarray,
        checkpoints: CheckpointManager,
        get_trainer_state: Callable[[], dict],
        state: Optional[dict] = None,
        is_on_policy: bool = False,
    ):
        super().__init__(verbose=0)
        self.run = run
        self.cfg = cfg
        self.starts = starts
        self.ckpt = checkpoints
        self.get_trainer_state = get_trainer_state
        self.is_on_policy = is_on_policy
        self.step_log = StepLog(run.path("logs", "steps"), cfg.step_log_chunk, enabled=cfg.log_steps == "all")
        self.episode_log = CSVLog(run.path("logs", "episodes.csv"), TRAIN_EPISODE_FIELDS)
        self.learner_log = JSONLLog(run.path("logs", "learner.jsonl"))
        self.eval_episode_log = CSVLog(run.path("eval", "episodes.csv"), EPISODE_FIELDS)
        self.eval_summary_log = CSVLog(run.path("eval", "summary.csv"), SUMMARY_FIELDS)
        s = state or {}
        self.state: dict[str, Any] = {
            "next_eval": s.get("next_eval", 0 if cfg.eval_at_start else cfg.eval_every),
            "eval_round": s.get("eval_round", 0),
            "next_ckpt": s.get("next_ckpt", cfg.checkpoint_every),
            "tp_done": s.get("tp_done", False),
            "tr_done": s.get("tr_done", False),
            "episodes_done": s.get("episodes_done", 0),
            "last_learner_flush": s.get("last_learner_flush", 0),
            "last_eval_t": s.get("last_eval_t"),
        }
        self._stop_requested = False
        self._stop_now = False
        self.preempted = False
        self._at_boundary = True
        self._ep_id: list[int] = []
        self._ep_start: list[int] = []
        self._ep_phase0: list[int] = []
        self._ep_step: list[int] = []
        self._ep_planner_steps: list[int] = []

    # ------------------------------------------------------------------------------------------
    def request_stop(self) -> None:
        """Called from the signal handler (SLURM preemption / time limit)."""
        self._stop_requested = True

    def scheduler_state(self) -> dict:
        return dict(self.state)

    @property
    def p2p(self):
        return getattr(self.model, "p2p", None)

    def _phase(self, t: int) -> int:
        return self.p2p.phase(t) if self.p2p is not None else 2

    def _kappa(self, t: int) -> float:
        return self.p2p.kappa(t) if self.p2p is not None else 0.0

    def _new_episode(self, i: int, t: int, reuse_empty: bool = False) -> None:
        """Start a new training episode in env ``i`` at step ``t`` (an empty episode keeps its id)."""
        if not (reuse_empty and self._ep_step[i] == 0):
            self._ep_id[i] = self._next_episode_id()
        self._ep_start[i] = t
        self._ep_phase0[i] = self._phase(t)
        self._ep_step[i] = 0
        self._ep_planner_steps[i] = 0

    def _log_aborted_episodes(self, t: int) -> None:
        """Record the unfinished training episode(s) at step ``t`` with ``aborted = 1``."""
        for i in range(self.training_env.num_envs):
            if self._ep_step[i] == 0:
                continue
            self.episode_log.append({"episode": self._ep_id[i], "t_start": self._ep_start[i], "t_end": t,
                                     "phase_start": self._ep_phase0[i], "phase_end": self._phase(t),
                                     "steps": self._ep_step[i], "aborted": 1})
            self.state["episodes_done"] += 1     # counts every logged episode, so episode ids stay unique on resume

    def _next_episode_id(self) -> int:
        self.state.setdefault("next_episode_id", self.state["episodes_done"])
        eid = self.state["next_episode_id"]
        self.state["next_episode_id"] = eid + 1
        return eid

    # ------------------------------------------------------------------------------------------
    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._ep_id = [0] * n
        self._ep_start = [0] * n
        self._ep_phase0 = [0] * n
        self._ep_step = [0] * n
        self._ep_planner_steps = [0] * n
        self.state["next_episode_id"] = self.state["episodes_done"]
        t = int(self.model.num_timesteps)
        for i in range(n):
            self._new_episode(i, t)
        self._at_boundary = True

    def _on_step(self) -> bool:
        n = self.training_env.num_envs
        t_after = int(self.model.num_timesteps)
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        rewards = self.locals["rewards"]
        rec = getattr(self.model, "step_record", None) or {}
        obs = self.model._last_obs
        for i in range(n):
            t = t_after - n + i
            info = infos[i]
            done = bool(dones[i])
            truncated = done and bool(info.get("TimeLimit.truncated", False))
            terminated = done and not truncated
            ex_pl = int(rec.get("executed_by_planner", np.zeros(n))[i]) if rec else 0
            self._ep_planner_steps[i] += ex_pl
            if self.step_log.enabled:
                pa = rec.get("planner_action") if rec else None
                beh = rec.get("behavior_action") if rec else None
                if beh is None:
                    beh = self.locals.get("actions")
                self.step_log.append({
                    "t": t, "episode": self._ep_id[i], "ep_step": self._ep_step[i], "phase": self._phase(t),
                    "kappa": self._kappa(t), "executed_by_planner": ex_pl,
                    "iota": float(not np.all(np.isnan(pa[i]))) if pa is not None else 0.0,
                    "obs": np.asarray(obs[i], dtype=np.float32),
                    "executed_action": np.asarray(info.get("executed_action", np.full(2, np.nan)), dtype=np.float32),
                    "behavior_action": np.asarray(beh[i], dtype=np.float32).reshape(-1)[:2] if beh is not None else np.full(2, np.nan, np.float32),
                    "planner_action": np.asarray(pa[i], dtype=np.float32) if pa is not None else np.full(2, np.nan, np.float32),
                    "reward": float(rewards[i]), "terminated": terminated, "truncated": truncated,
                    "outcome": int(info.get("outcome", 0)), "clearance": float(info.get("clearance", np.nan)),
                    "dist_to_goal": float(info.get("dist_to_goal", np.nan)), "yaw": float(info.get("yaw", np.nan)),
                    "vel_body": np.asarray(info.get("vel_body", np.full(2, np.nan)), dtype=np.float32),
                    "action_clipped": bool(info.get("action_clipped", False)),
                    "internal_reset": bool(info.get("internal_reset", False)),
                    "planner_iterations": int(rec["planner_iterations"][i]) if rec else 0,
                    "planner_ms": float(rec["planner_ms"][i]) if rec else np.nan,
                    "planner_max_constraint": float(rec["planner_max_constraint"][i]) if rec else np.nan,
                    "planner_outside_domain": int(rec["planner_outside_domain"][i]) if rec else 0,
                    "planner_action_clipped": int(rec["planner_action_clipped"][i]) if rec else 0,
                })
            self._ep_step[i] += 1
            if done:
                m = dict(info.get("episode_metrics", {}))
                m.update({
                    "episode": self._ep_id[i], "t_start": self._ep_start[i], "t_end": t + 1,
                    "phase_start": self._ep_phase0[i], "phase_end": self._phase(t), "kappa_end": self._kappa(t),
                    "planner_executed_steps": self._ep_planner_steps[i], "aborted": 0,
                })
                self.episode_log.append(m)
                self.state["episodes_done"] += 1
                self._new_episode(i, t + 1)
        self._at_boundary = bool(np.any(dones))
        return not self._stop_now

    # ------------------------------------------------------------------------------------------
    def _on_rollout_start(self) -> None:
        t = int(self.model.num_timesteps)
        self._flush_learner(t)
        self._phase_events(t)
        if t >= self.state["next_eval"] and (self.cfg.eval_mode == "snapshot" or self._at_boundary or self.is_on_policy):
            if self.cfg.eval_mode == "snapshot":
                self.save_evaluation_snapshot(t, scheduled=self.state["next_eval"])
            else:
                self.evaluate(t, scheduled=self.state["next_eval"])
            while self.state["next_eval"] <= t:
                self.state["next_eval"] += self.cfg.eval_every
        if self._stop_requested:
            if not self._at_boundary:
                # the episode in progress cannot be continued after the restart: record it as aborted
                self._log_aborted_episodes(t)
            self.save_checkpoint(t, tag="preemption")
            self.run.event("preempted", step=t)
            self.preempted = True
            self._stop_now = True
            return
        if t >= self.state["next_ckpt"]:
            self.save_checkpoint(t)
            while self.state["next_ckpt"] <= t:
                self.state["next_ckpt"] += self.cfg.checkpoint_every

    def _on_training_end(self) -> None:
        self._flush_learner(int(self.model.num_timesteps), force=True)
        self.step_log.flush()

    # ------------------------------------------------------------------------------------------
    def _flush_learner(self, t: int, force: bool = False) -> None:
        stats = getattr(self.model, "learner_stats", None)
        n_up = int(getattr(self.model, "_n_updates", 0))
        if stats is None or (not force and n_up - self.state["last_learner_flush"] < self.cfg.learner_log_every):
            return
        row = stats.pop()
        if not row or row.get("n_updates_in_window", 0) == 0:
            return
        row.update({"timesteps": t, "n_updates": n_up})
        rb = getattr(self.model, "replay_buffer", None)
        if rb is not None:
            if hasattr(rb, "planner"):
                row.update({"n_planner_buffer": rb.planner.size(), "n_online_buffer": rb.online.size(),
                            "n_goal_planner_buffer": rb.n_goal_transitions_planner()})
            else:
                row.update({"n_buffer": rb.size()})
        for k in ("n_planner_queries", "n_planner_queries_after_removal", "n_planner_executed_steps"):
            if hasattr(self.model, k):
                row[k] = int(getattr(self.model, k))
        self.learner_log.append(row)
        self.state["last_learner_flush"] = n_up

    def _phase_events(self, t: int) -> None:
        p = self.p2p
        if p is None or not p.uses_planner:
            return
        if not self.state["tp_done"] and t >= p.t_p:
            self.state["tp_done"] = True
            rb = getattr(self.model, "replay_buffer", None)
            data: dict[str, Any] = {"planner_executed_steps": int(getattr(self.model, "n_planner_executed_steps", 0))}
            if rb is not None and hasattr(rb, "planner"):
                data.update({"n_planner_buffer": rb.planner.size(), "n_goal_planner_buffer": rb.n_goal_transitions_planner()})
                if self.cfg.save_planner_buffer:
                    path = self.run.path("artifacts", "planner_buffer.npz")
                    rb.planner.export_npz(path)
                    data["planner_buffer_file"] = os.path.relpath(path, self.run.root)
            self.run.event("planner_phase_end", step=t, **data)
            self.save_checkpoint(t, milestone=True, tag="T_p")
        if not self.state["tr_done"] and t >= p.t_r:
            self.state["tr_done"] = True
            after = int(getattr(self.model, "n_planner_queries_after_removal", 0))
            self.run.event("planner_removed", step=t, kappa=p.kappa(t), queries_after_removal=after,
                           total_planner_queries=int(getattr(self.model, "n_planner_queries", 0)))
            if after != 0:
                self.run.event("warning", step=t, message="planner queried after T_r (Proposition 3 violated)")
            if p.t_r != p.t_p:
                self.save_checkpoint(t, milestone=True, tag="T_r")

    # ------------------------------------------------------------------------------------------
    def save_evaluation_snapshot(self, t: int, scheduled: Optional[int] = None) -> str:
        """Save only policy weights; never step/reset or replace the training environment."""
        snapshot = os.path.join("policies", f"policy_step_{t:010d}.zip")
        self.model.save(self.run.path(snapshot))
        self.run.event("policy_snapshot", step=t, scheduled=scheduled, path=snapshot)
        self.state["last_eval_t"] = t
        return snapshot

    def evaluate(self, t: int, scheduled: Optional[int] = None) -> dict:
        """Evaluation round on the training environment (the policy acts alone, fixed starts)."""
        env = self.training_env.envs[0].unwrapped
        r = self.state["eval_round"]
        snapshot = ""
        if self.cfg.save_policy_snapshots:
            snapshot = os.path.join("policies", f"policy_step_{t:010d}.zip")
            self.model.save(self.run.path(snapshot))
        with Timer() as tm:
            rows, traj = run_episodes(env, PolicyController(self.model), self.starts, round_index=r, t=t,
                                      record_trajectories=self.cfg.save_eval_trajectories)
        for row in rows:
            self.eval_episode_log.append(row)
        summ = summarize(rows, round_index=r, t=t, t_scheduled=scheduled, wall=tm.elapsed, snapshot=snapshot)
        self.eval_summary_log.append(summ)
        save_trajectories(self.run.path("eval", "trajectories"), f"round_{r:04d}_step_{t:010d}", traj)
        self.run.event("evaluation", step=t, round=r, success_rate=summ["success_rate"], crash_rate=summ["crash_rate"])
        self.state["eval_round"] = r + 1
        self.state["last_eval_t"] = t
        # Restart training on a fresh episode (the environment was used for evaluation).
        if not self._at_boundary:
            self._log_aborted_episodes(t)
            self.run.event("training_episode_aborted_for_evaluation", step=t)
        obs = self.training_env.reset()
        self.model._last_obs = obs
        if hasattr(self.model, "_last_episode_starts"):
            self.model._last_episode_starts = np.ones((self.training_env.num_envs,), dtype=bool)
        if hasattr(self.model, "reset_planners"):
            self.model.reset_planners()
        for i in range(self.training_env.num_envs):
            self._new_episode(i, t, reuse_empty=True)
        self._at_boundary = True
        return summ

    def save_checkpoint(self, t: int, milestone: bool = False, tag: str = "") -> str:
        self.step_log.flush()
        path = self.ckpt.save(self.model, t, self.get_trainer_state(), milestone=milestone, tag=tag)
        self.run.event("checkpoint", step=t, path=os.path.relpath(path, self.run.root), milestone=milestone, tag=tag)
        self.run.update_status(last_checkpoint_step=t)
        return path

    def close(self) -> None:
        self.step_log.flush()
        for log in (self.episode_log, self.learner_log, self.eval_episode_log, self.eval_summary_log):
            log.close()


def mean_or_nan(values) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else math.nan
