"""Suboptimality gap of the planner as a function of its computation budget (Corollary 1).

For states ``z`` visited by the system (taken from the step logs of a run, or sampled from ``rho_0``), compute

* ``Delta(k) = || pi_P^{(k)}(z) - u*(z) ||``  -- gap to the exact MPC minimiser (Definition 1),
* ``|| pi_P^{(k)}(z) - Proj_p(u^dagger) ||``   -- gap to the converged flow (Eq. (21), exponential decay),

for a list of iteration budgets ``k``, and fit ``log gap = log c - eta * k`` to the median gap. Offline only.

Usage::

    python -m p2pac.analysis.planner_gap --run runs/main/p2p-sac/seed_0 --budgets 1 2 5 10 20 50 100 200 \\
        --n-states 200 --out paper_results/planner_gap.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from p2pac.config import DEFAULTS, reap_config, task_config
from p2pac.planners import make_planner
from p2pac.planners.reference import converged_barrier_solution, exact_mpc_solution
from p2pac.runtime.records import load_steps
from p2pac.task import NavigationTask


def _ns_from_run(run_dir: str) -> argparse.Namespace:
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    merged = dict(DEFAULTS)
    merged.update(cfg)
    return argparse.Namespace(**merged)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="run directory (planner config + visited states)")
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100, 200])
    ap.add_argument("--n-states", type=int, default=200)
    ap.add_argument("--converged-iterations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="paper_results/planner_gap.csv")
    args = ap.parse_args(argv)

    ns = _ns_from_run(args.run) if args.run else argparse.Namespace(**DEFAULTS)
    if ns.planner != "reap":
        ns.planner = "reap"
    task = NavigationTask(task_config(ns))
    planner = make_planner("reap", reap_config(ns), task.cfg)
    rng = np.random.default_rng(args.seed)
    if args.run:
        d = load_steps(args.run, columns=["obs", "iota"])
        zs = d["obs"][d["iota"] > 0][:, :2] if "obs" in d else np.zeros((0, 2))
        if len(zs) == 0:
            zs = d["obs"][:, :2]
        zs = zs[rng.choice(len(zs), size=min(args.n_states, len(zs)), replace=False)]
    else:
        zs = np.stack([task.sample_start(rng) for _ in range(args.n_states)])

    rows = []
    for i, z in enumerate(zs):
        ex = exact_mpc_solution(planner, z)
        u_dag = converged_barrier_solution(planner, z, args.converged_iterations)
        for k in args.budgets:
            planner.reset()
            t0 = time.perf_counter()
            out = planner.plan(z, iterations=k)
            ms = 1e3 * (time.perf_counter() - t0)
            rows.append({"state": i, "z_x": z[0], "z_y": z[1], "iterations": k, "ms": ms,
                         "gap_exact": float(np.linalg.norm(out.u_world - ex["u0"])),
                         "gap_converged": float(np.linalg.norm(out.u_world - u_dag)),
                         "max_constraint": out.max_constraint, "outside_domain": out.outside_domain,
                         "exact_solver_success": ex["success"]})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    summ = df.groupby("iterations").agg(
        gap_exact_median=("gap_exact", "median"), gap_exact_p90=("gap_exact", lambda s: s.quantile(0.9)),
        gap_converged_median=("gap_converged", "median"), gap_converged_p90=("gap_converged", lambda s: s.quantile(0.9)),
        ms_mean=("ms", "mean"), max_constraint=("max_constraint", "max")).reset_index()
    pos = summ[summ["gap_converged_median"] > 1e-9]
    if len(pos) >= 2:
        slope, intercept = np.polyfit(pos["iterations"], np.log(pos["gap_converged_median"]), 1)
        summ.attrs["fit"] = {"eta": -slope, "c": float(np.exp(intercept))}
        with open(args.out.replace(".csv", "_fit.json"), "w") as f:
            json.dump({"eta_per_iteration": float(-slope), "c": float(np.exp(intercept)),
                       "model": "median gap_converged = c * exp(-eta * iterations)"}, f, indent=2)
    summ.to_csv(args.out.replace(".csv", "_summary.csv"), index=False)
    print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
