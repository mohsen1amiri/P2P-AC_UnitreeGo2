#!/usr/bin/env python3
"""Closed-loop validation of the planners on the kinematic test model (no Isaac Sim).

Regenerates the numbers of docs/planner_validation.md:

    python scripts/validate_planner.py --episodes 30 --budgets 1 2 5 10 20 50 100 --out planner_validation.csv

For every budget (iterations of the primal-dual flow) it reports the outcome of
closed-loop episodes from rho_0 (first-order velocity lag tau = 0.1 s, velocity noise), the mean episode length
of successful episodes, the largest constraint value of in-domain planner actions (must be <= 0: anytime
feasibility), the fraction of queries outside S_P (Assumption 1) and the time per query.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p2pac.planners import ReapConfig, make_planner  # noqa: E402
from p2pac.task import NavigationTask, Outcome, TaskConfig  # noqa: E402


def run(planner_name: str, cfg: ReapConfig, episodes: int, t_max: int, seed: int, tau: float, noise: float) -> dict:
    tc = TaskConfig()
    task = NavigationTask(tc)
    pl = make_planner(planner_name, cfg, tc)
    rng = np.random.default_rng(seed)
    outcomes, lengths = [], []
    maxc, outside, nq, tq = -np.inf, 0, 0, 0.0
    for _ in range(episodes):
        pl.reset()
        z = task.sample_start(rng)
        v = np.zeros(2)
        o = Outcome.RUNNING
        for t in range(t_max):
            out = pl.plan(z)
            nq += 1
            tq += out.wall_time_s
            outside += int(out.outside_domain)
            if not out.outside_domain:
                maxc = max(maxc, out.max_constraint)
            v = v + (tc.dt / tau) * (out.u_world - v)
            z = z + tc.dt * v + rng.normal(0, noise, 2)
            o = task.classify(z, height=0.3, contact=False, step_count=t + 1)
            if o != Outcome.RUNNING:
                break
        outcomes.append(o.name.lower())
        lengths.append(t + 1)
    c = Counter(outcomes)
    succ = [l for o, l in zip(outcomes, lengths) if o == "goal"]
    return {"goal_rate": c.get("goal", 0) / episodes, "collision_rate": c.get("collision_obstacle", 0) / episodes,
            "other": {k: v for k, v in c.items() if k not in ("goal", "collision_obstacle")},
            "success_len_mean": float(np.mean(succ)) if succ else np.nan, "max_constraint_in_SP": float(maxc),
            "outside_SP_frac": outside / nq, "ms_per_query": 1e3 * tq / nq}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100])
    ap.add_argument("--t-max", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--noise", type=float, default=0.002)
    ap.add_argument("--out", default="docs/planner_validation.csv")
    args = ap.parse_args()
    rows = []
    for k in args.budgets:
        r = run("reap", ReapConfig(iterations=k), args.episodes, args.t_max, args.seed, args.tau, args.noise)
        rows.append({"planner": "reap", "iterations": k, **r})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
