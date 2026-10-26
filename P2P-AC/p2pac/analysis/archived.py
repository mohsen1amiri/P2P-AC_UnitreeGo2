"""Load verified historical planner references without creating synthetic runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def load_archived_planner(entry: dict) -> pd.DataFrame:
    """Validate an archived planner evaluation and return its episode records."""
    root = Path(entry["root"])
    eval_dir = root / entry["evaluation_dir"]
    episode_file = eval_dir / "episodes.csv"

    # Verify that the original episode data have not changed.
    checksum = hashlib.sha256(episode_file.read_bytes()).hexdigest()
    if checksum != entry["episodes_sha256"]:
        raise ValueError("Archived REAP episodes fail SHA-256 verification")

    meta = json.loads((eval_dir / "meta.json").read_text())
    cfg = json.loads((root / "original_config.json").read_text())
    planner = json.loads((root / "planner.json").read_text())

    args = meta.get("args", {})
    starts_seed = meta.get("starts_seed", args.get("starts_seed"))
    expected_n = int(entry["expected_episodes"])
    expected_successes = int(entry["expected_successes"])
    budget = float(entry["expected_budget_ms"])
    horizon = int(entry["expected_horizon"])

    if (
        args.get("what") != "planner"
        or int(args.get("episodes", -1)) != expected_n
        or starts_seed != entry["expected_starts_seed"]
        or len(meta.get("starts", [])) != expected_n
    ):
        raise ValueError("Archived REAP evaluation metadata do not match the protocol")

    if (
        cfg.get("backend") != "isaac"
        or cfg.get("planner") != "reap"
        or cfg.get("planner_horizon") != horizon
        or cfg.get("reap_time_budget_ms") != budget
        or planner.get("horizon") != horizon
        or planner.get("time_budget_ms") != budget
    ):
        raise ValueError("Archived REAP planner configuration does not match the protocol")

    episodes = pd.read_csv(episode_file)
    summary = pd.read_csv(eval_dir / "summary.csv")

    if (
        len(episodes) != expected_n
        or int(episodes["success"].sum()) != expected_successes
        or set(episodes["eval_episode"].astype(int)) != set(range(expected_n))
        or not (episodes["round"] == 0).all()
        or len(summary) != 1
        or int(summary.iloc[0]["n_episodes"]) != expected_n
        or summary.iloc[0]["policy_snapshot"] != "planner"
        or abs(float(summary.iloc[0]["success_rate"]) -
               expected_successes / expected_n) > 1e-12
    ):
        raise ValueError("Archived REAP episode and summary records are inconsistent")

    if (
        "planner_ms_mean" not in episodes
        or not (episodes["planner_ms_mean"] > 0).all()
    ):
        raise ValueError("Archived REAP planner timing measurements are missing")

    return episodes