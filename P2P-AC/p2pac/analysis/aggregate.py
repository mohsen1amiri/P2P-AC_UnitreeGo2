"""Turn raw run directories into every table, figure data file and number of the paper.

Usage::

    python -m p2pac.analysis.aggregate --spec configs/analysis/paper.yaml --runs runs --out paper_results

The spec (YAML) declares which runs feed which artefact. Outputs (all machine readable):

* ``curves/data_<name>.csv``  -- learning curves in the format read by the paper's pgfplots macro
  (``steps_M, suc_mean, suc_lo, suc_hi``) plus std, CI and per-seed columns;
* ``tables/<table>.csv`` and ``tables/<table>_rows.tex`` -- Table II / III (mean +- std over seeds);
* ``budget_sweep.csv``        -- Fig. 3 (``tcpu, n80_mean, n80_std`` in ms / 1e6 steps) plus q-hat and K-hat;
* ``numbers.tex``             -- ``\\res{<key>}`` macros for every number quoted in the text;
* ``provenance.json``         -- the exact run directories, seeds, config hashes and statuses used.

Nothing is silently dropped: missing/unfinished runs are listed in ``provenance.json`` and in the console.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from p2pac.analysis.load import Run, discover, select
from p2pac.analysis.stats import (interpolate_curve, n_to_threshold, summarize_n80, summarize_seeds,
                                  wilson_interval)


def _load_spec(path: str) -> dict:
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            from p2pac.yaml_compat import safe_load

            return safe_load(f)
        return json.load(f)


# ----------------------------------------------------------------------------------------------------
# per-run quantities
# ----------------------------------------------------------------------------------------------------
def run_n80(run: Run, threshold: float) -> Optional[float]:
    s = run.eval_summary()
    if s.empty:
        return None
    return n_to_threshold(
        s["t"].to_numpy(),
        s["success_rate"].to_numpy(),
        threshold,
        step_interval=50_000,
    )


def run_final_metrics(run: Run) -> dict:
    """Final evaluation metrics of one run (efficiency metrics over successful episodes only)."""
    ep = run.final_episodes()
    if ep.empty:
        return {}
    succ = ep[ep["success"] == 1]
    return {
        "success": float(ep["success"].mean()), "crash": float(ep["crash"].mean()), "timeout": float(ep["timeout"].mean()),
        "path_ratio": float(succ["path_ratio"].mean()) if len(succ) else math.nan,
        "duration_s": float(succ["duration_s"].mean()) if len(succ) else math.nan,
        "speed": float(succ["mean_speed"].mean()) if len(succ) else math.nan,
        "return": float(ep["return"].mean()), "n_episodes": int(len(ep)), "n_success": int(ep["success"].sum()),
        "success_steps_p95": float(succ["steps"].quantile(0.95)) if len(succ) else math.nan,
        "success_steps_max": float(succ["steps"].max()) if len(succ) else math.nan,
        "planner_ms_mean": float(ep["planner_ms_mean"].mean()) if "planner_ms_mean" in ep and ep["planner_ms_mean"].notna().any() else math.nan,
    }


# ----------------------------------------------------------------------------------------------------
# artefacts
# ----------------------------------------------------------------------------------------------------
def build_curve(runs: list[Run], total_steps: int, grid_step: int, band: str = "std") -> pd.DataFrame:
    grid = np.arange(0, total_steps + 1, grid_step, dtype=float)
    cols = {}
    for r in runs:
        s = r.eval_summary()
        if s.empty:
            continue
        cols[f"seed{r.seed}"] = 100.0 * interpolate_curve(s["t"].to_numpy(), s["success_rate"].to_numpy(), grid)
    df = pd.DataFrame({"steps": grid, "steps_M": grid / 1e6})
    if not cols:
        return df
    mat = np.vstack(list(cols.values()))
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)
    n = np.sum(~np.isnan(mat), axis=0)
    sem = std / np.sqrt(np.maximum(n, 1))
    ci = 1.959964 * sem
    df["suc_mean"] = mean
    df["suc_std"] = std
    df["suc_ci95"] = ci
    half = std if band == "std" else ci
    df["suc_lo"] = np.clip(mean - half, 0.0, 100.0)    # clipped to the valid range of a success rate
    df["suc_hi"] = np.clip(mean + half, 0.0, 100.0)
    df["n_seeds"] = n
    for k, v in cols.items():
        df[k] = v
    return df.dropna(subset=["suc_mean"])


def fmt_pm(mean: float, std: float, digits: int = 2) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "--"
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def build_table_rows(spec_rows: list[dict], runs: list[Run], threshold: float, total_steps: int) -> tuple[list[dict], dict]:
    rows, prov = [], {}
    for row in spec_rows:
        sel = select(runs, row["experiment"], row["run"])
        prov[row["label"]] = [{"root": r.root, "seed": r.seed, "config_hash": r.config.get("config_hash"),
                               "state": r.status.get("state")} for r in sel]
        out: dict[str, Any] = {"label": row["label"], "experiment": row["experiment"], "run": row["run"], "n_seeds": len(sel)}
        finals = [run_final_metrics(r) for r in sel]
        finals = [f for f in finals if f]
        if row.get("reference") or row.get("no_n80"):
            out["n80"] = {}
        else:
            budget = max([int(r.config.get("total_steps", total_steps)) for r in sel] or [total_steps])
            out["n80"] = summarize_n80([run_n80(r, threshold) for r in sel], budget)
        for key, scale in (("success", 100.0), ("crash", 100.0), ("path_ratio", 1.0), ("duration_s", 1.0), ("speed", 1.0)):
            vals = [f[key] * scale for f in finals if not math.isnan(f.get(key, math.nan))]
            out[key] = summarize_seeds(vals)
        if row.get("reference") and finals:
            n_s = sum(f["n_success"] for f in finals)
            n = sum(f["n_episodes"] for f in finals)
            out["success_wilson"] = wilson_interval(n_s, n)
        rows.append(out)
    return rows, prov


def table_to_latex(rows: list[dict], columns: list[str]) -> str:
    lines = []
    for r in rows:
        cells = [r["label"]]
        for c in columns:
            if c == "n80":
                n = r.get("n80") or {}
                if not n:
                    cells.append("")
                elif n["n_reached"] == 0:
                    cells.append(f"$>${n['mean_censored_as_budget'] / 1e6:g}")
                elif n["n_censored"] > 0:
                    cells.append(f"{fmt_pm(n['mean'] / 1e6, n['std'] / 1e6)} ({n['n_reached']}/{n['n_seeds']})")
                else:
                    cells.append(fmt_pm(n["mean"] / 1e6, n["std"] / 1e6))
            else:
                s = r.get(c) or {}
                digits = 1 if c in ("success", "crash") else (3 if c == "speed" else 2)
                cells.append(fmt_pm(s.get("mean", math.nan), s.get("std", math.nan), digits))
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def table_to_frame(rows: list[dict]) -> pd.DataFrame:
    flat = []
    for r in rows:
        d = {"label": r["label"], "experiment": r["experiment"], "run": r["run"], "n_seeds": r["n_seeds"]}
        for k in ("n80", "success", "crash", "path_ratio", "duration_s", "speed"):
            for kk, v in (r.get(k) or {}).items():
                d[f"{k}_{kk}"] = v
        if "success_wilson" in r:
            d["success_wilson_lo"], d["success_wilson_hi"] = r["success_wilson"]
        flat.append(d)
    return pd.DataFrame(flat)


def build_budget_sweep(spec: dict, runs: list[Run], threshold: float, total_steps: int) -> pd.DataFrame:
    rows = []
    for item in spec["points"]:
        it = item["iterations"]
        learn = select(runs, item["experiment"], item["run"])
        n80 = summarize_n80([run_n80(r, threshold) for r in learn], total_steps)
        row = {"iterations": it, "n_seeds": len(learn), "n80_mean": n80["mean"] / 1e6, "n80_std": n80["std"] / 1e6,
               "n80_n_reached": n80["n_reached"], "n80_censored": n80["n_censored"]}
        if item.get("planner_run"):
            pl = select(runs, item.get("planner_experiment", spec.get("planner_experiment")), item["planner_run"])
            finals = [run_final_metrics(r) for r in pl]
            finals = [f for f in finals if f]
            if finals:
                n_s = sum(f["n_success"] for f in finals)
                n = sum(f["n_episodes"] for f in finals)
                lo, hi = wilson_interval(n_s, n)
                row.update({"q_hat": n_s / n, "q_lo": lo, "q_hi": hi, "K_hat_p95": np.nanmean([f["success_steps_p95"] for f in finals]),
                            "K_hat_max": np.nanmax([f["success_steps_max"] for f in finals]),
                            "tcpu": np.nanmean([f["planner_ms_mean"] for f in finals])})
        rows.append(row)
    df = pd.DataFrame(rows)
    if "tcpu" not in df:
        df["tcpu"] = np.nan
    return df[["tcpu"] + [c for c in df.columns if c != "tcpu"]]


def latex_key(s: str) -> str:
    return s.replace("\\", "/")


def write_numbers(entries: dict[str, str], path: str) -> None:
    lines = ["% Generated by p2pac.analysis.aggregate -- do not edit by hand.",
             "% Usage in the manuscript: \\res{<key>}", r"\providecommand{\res}[1]{\csname res@#1\endcsname}"]
    for k in sorted(entries):
        lines.append(rf"\expandafter\def\csname res@{latex_key(k)}\endcsname{{{entries[k]}}}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--runs", default=None, help="runs root (overrides spec.runs_root)")
    ap.add_argument("--out", default="paper_results")
    args = ap.parse_args(argv)
    spec = _load_spec(args.spec)
    runs_root = args.runs or spec.get("runs_root", "runs")
    runs = discover(runs_root)
    os.makedirs(args.out, exist_ok=True)
    threshold = float(spec.get("n80_threshold", 0.8))
    total = int(spec.get("total_steps", 5_000_000))
    numbers: dict[str, str] = {}
    provenance: dict[str, Any] = {"runs_root": os.path.abspath(runs_root), "n_runs_found": len(runs),
                                  "incomplete_runs": [r.root for r in runs if not r.completed]}

    # ---- learning curves (Fig. 2) ------------------------------------------------------------
    cdir = os.path.join(args.out, "curves")
    os.makedirs(cdir, exist_ok=True)
    for c in spec.get("curves", []):
        sel = select(runs, c["experiment"], c["run"])
        df = build_curve(sel, total, int(c.get("grid_step", spec.get("curve_grid_step", 100_000))), spec.get("band", "std"))
        df.to_csv(os.path.join(cdir, c["file"]), index=False)
        provenance.setdefault("curves", {})[c["file"]] = [r.root for r in sel]
        print(f"[curve] {c['file']}: {len(sel)} seeds")

    # ---- tables (Table II, Table III) --------------------------------------------------------
    tdir = os.path.join(args.out, "tables")
    os.makedirs(tdir, exist_ok=True)
    for name, t in spec.get("tables", {}).items():
        rows, prov = build_table_rows(t["rows"], runs, threshold, total)
        table_to_frame(rows).to_csv(os.path.join(tdir, f"{name}.csv"), index=False)
        with open(os.path.join(tdir, f"{name}_rows.tex"), "w") as f:
            f.write(table_to_latex(rows, t["columns"]))
        provenance.setdefault("tables", {})[name] = prov
        for r in rows:
            key = f"{name}/{r['run']}"
            if r.get("n80"):
                numbers[f"{key}/n80"] = f"{r['n80']['mean'] / 1e6:.2f}" if r["n80"].get("n_reached") else "--"
                numbers[f"{key}/n80_reached"] = f"{r['n80'].get('n_reached', 0)}/{r['n80'].get('n_seeds', 0)}"
            for k in ("success", "crash", "path_ratio", "duration_s", "speed"):
                m = (r.get(k) or {}).get("mean", math.nan)
                if not math.isnan(m):
                    numbers[f"{key}/{k}"] = f"{m:.1f}" if k in ("success", "crash") else f"{m:.2f}"
        print(f"[table] {name}: {len(rows)} rows")

    # ---- budget sweep (Fig. 3) ----------------------------------------------------------------
    if spec.get("budget_sweep"):
        df = build_budget_sweep(spec["budget_sweep"], runs, threshold, total)
        df.to_csv(os.path.join(args.out, "budget_sweep.csv"), index=False)
        for _, r in df.iterrows():
            numbers[f"budget/{int(r['iterations'])}/n80"] = f"{r['n80_mean']:.2f}" if not math.isnan(r["n80_mean"]) else "--"
            if "q_hat" in r and not math.isnan(r.get("q_hat", math.nan)):
                numbers[f"budget/{int(r['iterations'])}/q_hat"] = f"{r['q_hat']:.3f}"
        print(f"[budget] {len(df)} budgets")

    write_numbers(numbers, os.path.join(args.out, "numbers.tex"))
    with open(os.path.join(args.out, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"[done] results written to {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
