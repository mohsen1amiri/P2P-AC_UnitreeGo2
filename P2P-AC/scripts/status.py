#!/usr/bin/env python3
"""Show the state of every run under a runs folder, one line per run.

    python scripts/status.py                       # all runs under ./runs
    python scripts/status.py --runs runs --experiment main
    python scripts/status.py --only-unfinished     # what still has to run (or failed)

Columns: run directory, state (created / running / completed / failed / preempted), progress, the success rate of
the latest evaluation, and the time of the last update. Reads only status.json and eval/summary.csv; safe to run
while training is going on.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from datetime import datetime, timezone


def _last_eval(run: str):
    path = os.path.join(run, "eval", "summary.csv")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def _n(x: float) -> str:
    return f"{x / 1e6:.2f}M" if x >= 1e6 else (f"{x / 1e3:.0f}k" if x >= 1e3 else str(int(x)))


def _age(iso: str | None) -> str:
    if not iso:
        return "-"
    try:
        dt = datetime.fromisoformat(iso)
        mins = (datetime.now(timezone.utc) - dt).total_seconds() / 60
        return f"{mins:.0f} min ago" if mins < 120 else f"{mins / 60:.1f} h ago"
    except ValueError:
        return iso


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", help="runs folder (default: runs)")
    ap.add_argument("--experiment", default=None, help="only this experiment (first folder level)")
    ap.add_argument("--only-unfinished", action="store_true", help="hide completed runs")
    args = ap.parse_args()
    pattern = os.path.join(args.runs, args.experiment or "*", "*", "seed_*", "status.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"no runs found under {os.path.abspath(args.runs)}")
        return 0
    counts: dict[str, int] = {}
    print(f"{'run':60s} {'state':10s} {'progress':>17s} {'success':>8s}  updated")
    for p in paths:
        run = os.path.dirname(p)
        try:
            st = json.load(open(p))
        except (OSError, json.JSONDecodeError):
            st = {}
        state = st.get("state", "?")
        counts[state] = counts.get(state, 0) + 1
        if args.only_unfinished and state == "completed":
            continue
        step, total = st.get("step"), st.get("total_steps")
        prog = f"{_n(step)} / {_n(total)}" if step is not None and total else "-"
        ev = _last_eval(run)
        succ = f"{100 * float(ev['success_rate']):.0f}%" if ev else "-"
        print(f"{os.path.relpath(run, args.runs):60s} {state:10s} {prog:>17s} {succ:>8s}  {_age(st.get('updated_utc'))}")
    print("\n" + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
