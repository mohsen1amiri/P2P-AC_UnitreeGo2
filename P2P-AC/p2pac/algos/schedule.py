"""Phases and removal schedule of P2P-AC (Sec. V-A, V-D, Eq. (17)).

``t`` is the number of environment steps taken so far (``num_timesteps`` of Stable-Baselines3), i.e. the
index of the transition being collected (Algorithm 1, line 2).

* planner phase    ``0 <= t < T_p``   -- the planner selects the executed actions (Eq. (16));
* transition phase ``T_p <= t < T_r`` -- the policy acts, the imitation weight decays linearly to 0;
* AC phase         ``t >= T_r``       -- the planner is not queried; training is the underlying AC algorithm
  (Proposition 3).

with ``T_r = T_p + T_d`` and the imitation weight (Eq. (17))::

    kappa_t = kappa_0                              for t < T_p
    kappa_t = kappa_0 * (1 - (t - T_p) / T_d)      for T_p <= t < T_r
    kappa_t = 0                                    for t >= T_r
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

IMITATION_SPACES = ("logit", "action", "action_sample")


@dataclass
class P2PConfig:
    """Everything P2P-AC adds to the underlying actor-critic algorithm."""

    t_p: int = 100_000                 # planner horizon T_p
    t_d: int = 50_000                  # transition horizon T_d
    kappa0: float = 10.0               # initial imitation weight kappa_0
    execute_planner: bool = True       # planner selects the executed actions for t < T_p (Eq. (16))
    imitation_space: str = "logit"     # "logit" (Eq. (19)), "action" (deterministic action), "action_sample" (Accel. SAC)
    logit_clip_eps: float = 1e-4       # epsilon_c of Eq. (18)
    label_noise_std: float = 0.0       # Gaussian noise on the imitation target, normalised action units (Accel. SAC)

    def __post_init__(self) -> None:
        if self.t_p < 0 or self.t_d < 0:
            raise ValueError("t_p and t_d must be non-negative")
        if self.imitation_space not in IMITATION_SPACES:
            raise ValueError(f"imitation_space must be one of {IMITATION_SPACES}")
        if not (0.0 < self.logit_clip_eps < 1.0):
            raise ValueError("logit_clip_eps must be in (0, 1)")
        if self.kappa0 < 0:
            raise ValueError("the imitation weight kappa0 must be non-negative")

    @classmethod
    def disabled(cls) -> "P2PConfig":
        """No planner at all: the algorithm is exactly the underlying AC algorithm (baselines)."""
        return cls(t_p=0, t_d=0, kappa0=0.0, execute_planner=False)

    # ------------------------------------------------------------------------------------------
    @property
    def t_r(self) -> int:
        """Removal time T_r = T_p + T_d."""
        return self.t_p + self.t_d

    def kappa(self, t: int) -> float:
        """Imitation weight kappa_t of Eq. (17)."""
        if t < self.t_p:
            return self.kappa0
        if t < self.t_r:
            return self.kappa0 * (1.0 - (t - self.t_p) / self.t_d)
        return 0.0

    def phase(self, t: int) -> int:
        """0 = planner phase, 1 = transition phase, 2 = AC phase."""
        if t < self.t_p:
            return 0
        if t < self.t_r:
            return 1
        return 2

    def executes_planner(self, t: int) -> bool:
        return self.execute_planner and t < self.t_p

    def queries_planner(self, t: int) -> bool:
        """The planner is queried only while its action is executed or used as an imitation target."""
        return self.executes_planner(t) or self.kappa(t) > 0.0

    def routes_to_planner_buffer(self, t: int) -> bool:
        """Transitions of the planner phase (t < T_p) are stored in the protected buffer D_P."""
        return t < self.t_p

    @property
    def uses_planner(self) -> bool:
        return (self.execute_planner and self.t_p > 0) or self.kappa0 > 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["t_r"] = self.t_r
        return d
