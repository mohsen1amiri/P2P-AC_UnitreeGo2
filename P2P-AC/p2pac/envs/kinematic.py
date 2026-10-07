"""Kinematic test backend (pure numpy).

**Not a paper experiment.** This backend exists so that the complete pipeline -- task logic, planner, P2P-AC,
logging, evaluation, checkpoint/resume and aggregation -- can be exercised end to end on a laptop/CI without
Isaac Sim. Every result in the paper must come from the Isaac Lab backend.

Model (body-frame velocity command ``u = [v_x, v_y]``, yaw-rate command ``w``)::

    v_b   <- v_b + (dt / tau) (u - v_b) + N(0, sigma_v^2 I)      (first-order velocity tracking)
    yaw   <- yaw + dt * w + N(0, sigma_yaw^2)                     (heading drift)
    p     <- p + dt * R(yaw) v_b
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from p2pac.envs.base import RobotState


class KinematicBackend:
    """Planar point robot with first-order velocity tracking and noise."""

    def __init__(
        self,
        dt: float = 0.02,
        tau: float = 0.1,
        vel_noise_std: float = 0.01,
        yaw_noise_std: float = 0.0,
        height: float = 0.3,
        seed: int = 0,
    ):
        self.dt = float(dt)
        self.tau = float(tau)
        self.vel_noise_std = float(vel_noise_std)
        self.yaw_noise_std = float(yaw_noise_std)
        self.height = float(height)
        self._rng = np.random.default_rng(seed)
        self._xy = np.zeros(2)
        self._yaw = 0.0
        self._vb = np.zeros(2)

    # RNG state is part of the checkpoint so that resumed kinematic runs are bitwise reproducible.
    def get_rng_state(self) -> dict:
        return self._rng.bit_generator.state

    def set_rng_state(self, state: dict) -> None:
        self._rng.bit_generator.state = state

    def _state(self) -> RobotState:
        return RobotState(xy=self._xy.copy(), height=self.height, yaw=self._yaw, vel_body=self._vb.copy())

    def reset_to(self, x: float, y: float, yaw: float) -> RobotState:
        self._xy = np.array([x, y], dtype=np.float64)
        self._yaw = float(yaw)
        self._vb = np.zeros(2)
        return self._state()

    def step(self, vx: float, vy: float, yaw_rate: float) -> RobotState:
        u = np.array([vx, vy], dtype=np.float64)
        self._vb = self._vb + (self.dt / self.tau) * (u - self._vb)
        if self.vel_noise_std > 0:
            self._vb = self._vb + self._rng.normal(0.0, self.vel_noise_std, 2)
        self._yaw = self._yaw + self.dt * yaw_rate
        if self.yaw_noise_std > 0:
            self._yaw += float(self._rng.normal(0.0, self.yaw_noise_std))
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        v_w = np.array([c * self._vb[0] - s * self._vb[1], s * self._vb[0] + c * self._vb[1]])
        self._xy = self._xy + self.dt * v_w
        return self._state()

    def render(self) -> Optional[np.ndarray]:
        return None

    def close(self) -> None:
        pass

    def describe(self) -> dict:
        return {
            "backend": "kinematic",
            "dt": self.dt,
            "tau": self.tau,
            "vel_noise_std": self.vel_noise_std,
            "yaw_noise_std": self.yaw_noise_std,
            "note": "validation/test backend -- not used for any paper result",
        }
