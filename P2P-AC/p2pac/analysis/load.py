"""Discovery and loading of run directories."""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class Run:
    root: str
    config: dict
    status: dict = field(default_factory=dict)

    @property
    def experiment(self) -> str:
        return self.config.get("experiment", "")

    @property
    def name(self) -> str:
        m, a, v = self.config.get("method"), self.config.get("algo"), self.config.get("variant")
        n = m if a in (None, "", "None") else f"{m}-{a}"
        return f"{n}__{v}" if v else n

    @property
    def seed(self) -> int:
        return int(self.config.get("seed", -1))

    @property
    def completed(self) -> bool:
        return self.status.get("state") == "completed"

    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    def eval_summary(self) -> pd.DataFrame:
        p = self.path("eval", "summary.csv")
        return pd.read_csv(p) if os.path.isfile(p) else pd.DataFrame()

    def eval_episodes(self, name: str = "episodes") -> pd.DataFrame:
        p = self.path("eval", f"{name}.csv")
        return pd.read_csv(p) if os.path.isfile(p) else pd.DataFrame()

    def final(self) -> Optional[dict]:
        p = self.path("eval", "final.json")
        if not os.path.isfile(p):
            return None
        with open(p) as f:
            return json.load(f)

    def final_episodes(self) -> pd.DataFrame:
        return self.eval_episodes("final_episodes")

    def train_episodes(self) -> pd.DataFrame:
        p = self.path("logs", "episodes.csv")
        return pd.read_csv(p) if os.path.isfile(p) else pd.DataFrame()

    def learner(self) -> pd.DataFrame:
        p = self.path("logs", "learner.jsonl")
        return pd.read_json(p, lines=True) if os.path.isfile(p) and os.path.getsize(p) > 0 else pd.DataFrame()

    def events(self) -> pd.DataFrame:
        p = self.path("events.jsonl")
        return pd.read_json(p, lines=True) if os.path.isfile(p) and os.path.getsize(p) > 0 else pd.DataFrame()


def discover(runs_root: str) -> list[Run]:
    """All run directories below ``runs_root`` (a run directory contains config.json)."""
    out = []
    for cfg_path in sorted(glob.glob(os.path.join(runs_root, "**", "config.json"), recursive=True)):
        root = os.path.dirname(cfg_path)
        if os.sep + "attempts" + os.sep in root:
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        st_path = os.path.join(root, "status.json")
        status = json.load(open(st_path)) if os.path.isfile(st_path) else {}
        out.append(Run(root=root, config=cfg, status=status))
    return out


def select(runs: list[Run], experiment: str, name: str, completed_only: bool = True) -> list[Run]:
    sel = [r for r in runs if r.experiment == experiment and r.name == name and (r.completed or not completed_only)]
    # one run per seed: prefer the first attempt over "__rerun_" siblings unless only reruns exist
    by_seed: dict[int, Run] = {}
    for r in sorted(sel, key=lambda r: ("__rerun_" in r.root, r.root)):
        by_seed.setdefault(r.seed, r)
    return [by_seed[s] for s in sorted(by_seed)]
