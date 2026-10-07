"""Environment construction for every backend (Isaac Lab Go2, kinematic test model)."""

from __future__ import annotations

from typing import Optional

from p2pac.envs.base import GoalNavEnv, RobotBackend, RobotState
from p2pac.task import NavigationTask


def make_backend(cfg, task: NavigationTask, seed: int, render_mode: Optional[str] = None) -> RobotBackend:
    if cfg.backend == "kinematic":
        from p2pac.envs.kinematic import KinematicBackend

        return KinematicBackend(dt=task.cfg.dt, tau=cfg.kin_tau, vel_noise_std=cfg.kin_vel_noise,
                                yaw_noise_std=cfg.kin_yaw_noise, seed=seed)
    if cfg.backend == "isaac":
        from p2pac.envs.isaac_go2 import DEFAULT_CHECKPOINT, IsaacGo2Backend

        ckpt = cfg.locomotion_checkpoint or DEFAULT_CHECKPOINT
        sim_device = (cfg.app_launcher or {}).get("device") or "cuda:0"
        return IsaacGo2Backend(
            task.cfg, seed=seed, device=sim_device, render_mode=render_mode, locomotion_checkpoint=ckpt,
            episode_length_s=cfg.isaac_episode_length_s,
            obs_noise=cfg.locomotion_obs_noise, spawn_height=cfg.spawn_height, contact_threshold=cfg.contact_threshold,
        )
    raise ValueError(f"unknown backend {cfg.backend!r}")


def make_env(cfg, task: NavigationTask, seed: int, render_mode: Optional[str] = None) -> GoalNavEnv:
    backend = make_backend(cfg, task, seed, render_mode)
    return GoalNavEnv(task, backend, vx_max=cfg.vx_max, vy_max=cfg.vy_max, obs_mode=cfg.obs_mode,
                      yaw_hold_gain=cfg.yaw_hold_gain, gamma=cfg.gamma, render_mode=render_mode)


__all__ = ["GoalNavEnv", "RobotBackend", "RobotState", "make_backend", "make_env"]
