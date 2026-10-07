"""Empirical checks of the assumptions and statements of Sec. V-VI, computed from the logs of P2P-AC runs.

For every run with a planner phase:

* **Assumption 1** -- fraction of planner queries whose state was outside ``S_P`` (``planner_outside_domain``);
* **Definition 1(i)** -- largest constraint value of planner actions for states in ``S_P`` (must be <= 0);
* **Assumption 2** -- ``q_hat`` and ``K_hat`` from the planner-phase episodes (the planner run of Sec. V-B);
* **Proposition 2** -- ``N_G`` (steps until the first transition into ``G``), compared with the bound
  ``K + (1 - q)/q * T_max``; ``n_G`` (goal transitions in ``D_P``) compared with ``q * floor(T_p / T_max)``;
* **Corollary 2** -- observed fraction of minibatches containing at least one goal transition from ``D_P``
  versus ``1 - (1 - n_G / T_p)^{floor(n_b / 2)}``;
* **Proposition 3** -- number of planner queries after ``T_r`` (must be 0) and ``kappa_t = 0`` for ``t >= T_r``.

Usage: ``python -m p2pac.analysis.theory --runs runs --out paper_results/theory_checks.csv``
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

from p2pac.analysis.load import Run, discover
from p2pac.runtime.records import load_steps

GOAL = 1


def check_run(run: Run) -> dict:
    c = run.config
    t_p, t_d = int(c.get("t_p", 0)), int(c.get("t_d", 0))
    t_r = t_p + t_d
    t_max = int(c["max_episode_steps"])
    n_b = int(c["batch_size"])
    out = {"run_dir": run.root, "experiment": run.experiment, "run": run.name, "seed": run.seed,
           "T_p": t_p, "T_r": t_r, "T_max": t_max, "batch_size": n_b, "completed": run.completed}
    steps = load_steps(run.root, columns=["t", "outcome", "executed_by_planner", "iota", "kappa", "planner_outside_domain",
                                          "planner_max_constraint", "planner_ms", "episode", "terminated", "truncated"])
    if not steps:
        return out
    t = steps["t"]
    q_mask = steps["iota"] > 0
    out["planner_queries"] = int(q_mask.sum())
    out["planner_queries_after_T_r"] = int((q_mask & (t >= t_r)).sum())
    out["kappa_max_after_T_r"] = float(np.max(steps["kappa"][t >= t_r])) if np.any(t >= t_r) else math.nan
    if q_mask.any():
        outside = steps["planner_outside_domain"][q_mask] > 0
        out["assumption1_violation_frac"] = float(outside.mean())
        inside_c = steps["planner_max_constraint"][q_mask][~outside]
        out["max_constraint_in_SP"] = float(np.nanmax(inside_c)) if inside_c.size else math.nan
        out["planner_ms_mean"] = float(np.nanmean(steps["planner_ms"][q_mask]))
        out["planner_ms_p95"] = float(np.nanpercentile(steps["planner_ms"][q_mask], 95))

    # planner run = transitions executed by the planner (t < T_p)
    ex = steps["executed_by_planner"] > 0
    if ex.any():
        goal_idx = np.nonzero(ex & (steps["outcome"] == GOAL))[0]
        out["N_G"] = int(t[goal_idx[0]] + 1) if goal_idx.size else math.nan
        out["n_G_observed"] = int(goal_idx.size)
        # completed planner episodes: those that ended (terminated or truncated) while the planner acted
        ends = np.nonzero(ex & ((steps["terminated"] > 0) | (steps["truncated"] > 0)))[0]
        eps = steps["episode"][ends]
        lengths, success = [], []
        for e, i_end in zip(eps, ends):
            idx = np.nonzero(steps["episode"] == e)[0]
            if np.all(ex[idx]):
                lengths.append(len(idx))
                success.append(steps["outcome"][i_end] == GOAL)
        if success:
            q = float(np.mean(success))
            succ_len = [l for l, s in zip(lengths, success) if s]
            k_hat = float(np.max(succ_len)) if succ_len else math.nan
            out.update({"planner_episodes": len(success), "q_hat": q, "K_hat_max": k_hat,
                        "K_hat_p95": float(np.percentile(succ_len, 95)) if succ_len else math.nan})
            if q > 0:
                out["prop2_bound_E_NG"] = k_hat + (1 - q) / q * t_max
                out["prop2iii_bound_E_nG"] = q * math.floor(t_p / t_max)
            n_g = out["n_G_observed"]
            out["cor2_predicted_hit_prob"] = 1 - (1 - n_g / t_p) ** (n_b // 2) if t_p > 0 else math.nan

    learner = run.learner()
    if not learner.empty and "batch_has_goal_from_planner_buffer" in learner:
        after = learner[learner["timesteps"] >= t_p]
        if len(after):
            out["cor2_observed_hit_prob"] = float(after["batch_has_goal_from_planner_buffer"].mean())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="paper_results/theory_checks.csv")
    ap.add_argument("--experiment", default=None)
    args = ap.parse_args(argv)
    rows = []
    for run in discover(args.runs):
        if args.experiment and run.experiment != args.experiment:
            continue
        if run.config.get("method") != "p2p":
            continue
        rows.append(check_run(run))
        print(f"[theory] {run.root}")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    if not df.empty:
        agg = df.groupby(["experiment", "run"]).agg(lambda s: s.mean() if np.issubdtype(s.dtype, np.number) else s.iloc[0])
        agg.to_csv(args.out.replace(".csv", "_by_run.csv"))
    print(f"[done] {len(df)} runs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
