"""P2P-AC: Planner-to-Policy Actor-Critic reinforcement learning via an anytime-feasible MPC planner.

Package layout (see README.md for the full mapping to the paper):

* :mod:`p2pac.task`       -- the navigation MDP (arena, goal set G, failure set F, reward Eq. (1)).
* :mod:`p2pac.envs`       -- simulation backends (NVIDIA Isaac Lab Go2, and a kinematic test backend).
* :mod:`p2pac.planners`   -- the anytime-feasible MPC planner (REAP, Sec. IV).
* :mod:`p2pac.algos`      -- P2P-AC (Sec. V) on top of SAC / TD3 / DDPG, and the baselines.
* :mod:`p2pac.runtime`    -- run directories, logging, checkpoints/resume, evaluation protocol.
* :mod:`p2pac.analysis`   -- aggregation of raw logs into the tables/figures of the paper.
* :mod:`p2pac.deploy`     -- hardware deployment on the Unitree Go2.
"""

from p2pac.version import __version__

__all__ = ["__version__"]
