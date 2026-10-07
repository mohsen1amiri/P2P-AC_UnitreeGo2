"""Planner interface and the planner-to-action map.

A planner implements the map ``pi_P^{(t_cpu)}`` of Definition 1: it receives the planner state
``z_t = h_z(s_t)`` (here the planar position, i.e. the first two entries of the MDP state) and returns
a *world-frame* planar velocity. The world-to-body map of Sec. VI-A then produces the planner action
``u_t^dagger`` in the action space of the MDP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class PlannerOutput:
    """Result of one planner query.

    Attributes
    ----------
    u_world:
        First input ``Proj_p(u_hat)`` of the returned input sequence, world frame (m/s).
    iterations:
        Number of primal-dual iterations performed for this query (the computation budget used).
    wall_time_s:
        Measured wall-clock time of the query.
    max_constraint:
        ``max_i (eta_i^T u + g_i)`` at the returned sequence for the unrelaxed constraints.
        Non-positive means the returned sequence lies in ``U(z_t, d)`` (feasibility, Definition 1(i)).
    outside_domain:
        True if the initial condition was infeasible, i.e. ``z_t`` is not in ``S_P`` (Assumption 1
        violated) and the planner had to relax constraints to obtain a strictly feasible start.
    info:
        Planner-specific diagnostics (all scalars, logged to the step log).
    """

    u_world: np.ndarray
    iterations: int = 0
    wall_time_s: float = 0.0
    max_constraint: float = math.nan
    outside_domain: bool = False
    info: dict = field(default_factory=dict)


class Planner(Protocol):
    """Protocol implemented by every planner."""

    name: str

    def plan(self, z: np.ndarray) -> PlannerOutput:  # pragma: no cover - protocol
        ...

    def reset(self) -> None:  # pragma: no cover - protocol
        """Called at the beginning of every episode (Sec. IV-A: fixed feasible initial condition)."""
        ...


def world_to_body(u_world: np.ndarray, heading: float) -> np.ndarray:
    """Rotate a world-frame planar velocity into the body frame of a robot with yaw ``heading``.

    With the fixed heading ``pi/2`` of Sec. VI-A this is exactly ``u^dagger = [u_y, -u_x]``.
    """
    c, s = math.cos(heading), math.sin(heading)
    ux, uy = float(u_world[0]), float(u_world[1])
    return np.array([c * ux + s * uy, -s * ux + c * uy], dtype=np.float64)


class PlannerActionMap:
    """Maps planner outputs to MDP actions (body-frame commands within the action box).

    The paper assumes the robot keeps the spawn heading (``heading_rad = pi/2``), so the mapping uses this fixed
    heading: ``u_body = [u_y, -u_x]`` (Sec. VI-A). Heading drift of the real robot is part of
    the model mismatch ``w_t`` and is logged by the environment (``yaw`` / ``max_abs_yaw_drift``).
    """

    def __init__(self, heading_rad: float, action_low: np.ndarray, action_high: np.ndarray):
        self.heading = float(heading_rad)
        self.low = np.asarray(action_low, dtype=np.float64)
        self.high = np.asarray(action_high, dtype=np.float64)

    def __call__(self, out: PlannerOutput) -> tuple[np.ndarray, bool]:
        """Return ``(u_dagger, clipped)``; ``clipped`` flags a planner action outside the action box."""
        u = world_to_body(out.u_world, self.heading)
        u_clip = np.clip(u, self.low, self.high)
        return u_clip, bool(np.any(np.abs(u_clip - u) > 1e-9))
