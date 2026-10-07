"""Aggregate hardware trials (Sec. VI-H): success rate with a Wilson interval, efficiency metrics, and the data of
the hardware figure (trajectory and commanded vs. measured velocity of a chosen trial).

    python -m p2pac.deploy.hardware_analysis --trials hardware/p2p-sac --out paper_results/hardware
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from p2pac.analysis.stats import wilson_interval


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", required=True, help="directory written by p2pac.deploy.go2_deploy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--figure-trial", type=int, default=None, help="trial index exported for the hardware figure")
    args = ap.parse_args(argv)
    rows = []
    for p in sorted(glob.glob(os.path.join(args.trials, "trial_*", "summary.json"))):
        with open(p) as f:
            d = json.load(f)
        d["trial"] = os.path.basename(os.path.dirname(p))
        rows.append(d)
    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "hardware_trials.csv"), index=False)
    n, s = len(df), int(df["success"].sum()) if len(df) else 0
    lo, hi = wilson_interval(s, n)
    succ = df[df["success"] == 1] if len(df) else df
    summary = {"n_trials": n, "n_success": s, "success_rate": s / n if n else np.nan, "wilson_lo": lo, "wilson_hi": hi,
               "duration_s_mean": float(succ["duration_s"].mean()) if len(succ) else np.nan,
               "path_ratio_mean": float(succ["path_ratio"].mean()) if len(succ) else np.nan,
               "mean_speed_mean": float(succ["mean_speed"].mean()) if len(succ) else np.nan,
               "min_clearance_min": float(df["min_clearance"].min()) if len(df) else np.nan}
    with open(os.path.join(args.out, "hardware_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if args.figure_trial is not None:
        log = pd.read_csv(os.path.join(args.trials, f"trial_{args.figure_trial:03d}", "log.csv"))
        log[["wall_time", "x", "y", "obs_gx", "obs_gy", "cmd_vx", "cmd_vy", "meas_vx", "meas_vy"]].to_csv(
            os.path.join(args.out, "hardware_figure_data.csv"), index=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
