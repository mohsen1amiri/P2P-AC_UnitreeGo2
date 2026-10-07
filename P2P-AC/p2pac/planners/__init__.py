"""Planners: the anytime-feasible REAP planner of Sec. IV."""

from __future__ import annotations

from p2pac.planners.base import Planner, PlannerActionMap, PlannerOutput, world_to_body
from p2pac.planners.reap import ReapConfig, ReapPlanner

PLANNERS = ("reap", "none")


def make_planner(name: str, reap_cfg: ReapConfig, task_cfg) -> Planner | None:
    """Instantiate a planner for the given task.

    ``"reap"`` -- the anytime-feasible planner of Sec. IV (:class:`~p2pac.planners.reap.ReapPlanner`).
    ``"none"`` -- no planner (pure actor-critic baselines).
    """
    if name == "none":
        return None
    if name == "reap":
        return ReapPlanner(
            reap_cfg,
            goal=task_cfg.goal,
            obstacles=task_cfg.obstacle_array,
            robot_radius=task_cfg.robot_radius,
            leg_reach_margin=task_cfg.leg_reach_margin,
            arena=(task_cfg.x_min, task_cfg.x_max, task_cfg.y_min, task_cfg.y_max),
            wall_margin=task_cfg.wall_margin,
        )
    raise ValueError(f"unknown planner {name!r}; choose from {PLANNERS}")


__all__ = ["PLANNERS", "Planner", "PlannerActionMap", "PlannerOutput", "ReapConfig", "ReapPlanner", "make_planner", "world_to_body"]
