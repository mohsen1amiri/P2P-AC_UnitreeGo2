"""Run directory layout, status file, event log and the resume/never-overwrite policy.

Layout of one run (``<output_root>/<experiment>/<method>-<algo>[__<variant>]/seed_<k>/``)::

    config.json            fully resolved configuration (+ config_hash)
    manifest.json          provenance: git commit/diff, package versions, host, GPU, SLURM ids, command line
    status.json            state machine: created -> running -> completed | failed | preempted
    events.jsonl           phase changes, evaluations, checkpoints, resumes, warnings (one JSON per line)
    logs/steps/*.npz       every environment step (chunked, see p2pac.runtime.records)
    logs/episodes.csv      every training episode
    logs/learner.jsonl     learner statistics every ``learner_log_every`` gradient updates
    logs/sb3/              raw Stable-Baselines3 logger output (progress.csv, monitor.csv, tensorboard)
    eval/summary.csv       one row per evaluation round
    eval/episodes.csv      one row per evaluation episode
    eval/starts.json       the fixed set of evaluation start positions
    eval/trajectories/     xy trajectories of every evaluation episode (npz per round)
    eval/final.json        final evaluation (``eval_final_episodes`` episodes)
    policies/              policy snapshot at every evaluation (re-evaluation without retraining)
    checkpoints/           full training state for resume (model, optimisers, replay buffer, RNG states)
    artifacts/             planner buffer D_P at T_p, git diff, planner description, ...
    attempts/              previous attempts that crashed before their first checkpoint (never deleted)
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import tempfile
from typing import Any, Optional

STATUS_CREATED = "created"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_PREEMPTED = "preempted"

EXIT_PREEMPTED = 3  # exit code used to ask the sbatch wrapper to requeue the job


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(o)


def read_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


class RunDir:
    """Paths and bookkeeping of one run."""

    SUBDIRS = ("logs", "logs/steps", "logs/sb3", "eval", "eval/trajectories", "policies", "checkpoints", "artifacts")

    def __init__(self, root: str):
        self.root = os.path.abspath(root)

    # paths ----------------------------------------------------------------------------------------
    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    @property
    def config_path(self) -> str:
        return self.path("config.json")

    @property
    def manifest_path(self) -> str:
        return self.path("manifest.json")

    @property
    def status_path(self) -> str:
        return self.path("status.json")

    @property
    def events_path(self) -> str:
        return self.path("events.jsonl")

    # state ----------------------------------------------------------------------------------------
    def exists(self) -> bool:
        return os.path.isfile(self.config_path)

    def status(self) -> dict:
        return read_json(self.status_path) or {}

    def is_completed(self) -> bool:
        return self.status().get("state") == STATUS_COMPLETED

    def latest_checkpoint(self) -> Optional[str]:
        ptr = self.path("checkpoints", "LATEST")
        if not os.path.isfile(ptr):
            return None
        with open(ptr) as f:
            name = f.read().strip()
        p = self.path("checkpoints", name)
        return p if os.path.isdir(p) else None

    def create(self) -> None:
        for sub in self.SUBDIRS:
            os.makedirs(self.path(sub), exist_ok=True)

    def archive_attempt(self) -> str:
        """Move the content of an attempt without checkpoint into ``attempts/`` (never deletes anything)."""
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = self.path("attempts", f"attempt_{stamp}")
        os.makedirs(dest, exist_ok=True)
        for name in os.listdir(self.root):
            if name == "attempts":
                continue
            shutil.move(self.path(name), os.path.join(dest, name))
        return dest

    def update_status(self, **kw: Any) -> None:
        st = self.status()
        st.update(kw)
        st["updated_utc"] = utcnow()
        write_json_atomic(self.status_path, st)

    def event(self, kind: str, step: Optional[int] = None, **data: Any) -> None:
        rec = {"time_utc": utcnow(), "event": kind, "step": step}
        rec.update(data)
        with open(self.events_path, "a") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")


def new_attempt_dir(root: str) -> str:
    """Directory for ``--force-new``: a sibling of the finished run, never overwriting it."""
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{root.rstrip(os.sep)}__rerun_{stamp}"
