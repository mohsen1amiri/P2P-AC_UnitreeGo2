"""Machine-readable log writers and readers.

* :class:`StepLog`   -- every environment step, buffered in memory and written as compressed ``.npz`` chunks
  ``logs/steps/steps_<first>_<last>.npz`` (columns are 1-D/2-D numpy arrays, see ``STEP_COLUMNS``).
* :class:`CSVLog`    -- append-only CSV with a fixed header (episodes, evaluation rounds/episodes).
* :class:`JSONLLog`  -- append-only JSON lines for records with variable keys (learner statistics, events).

Resume safety: logs are flushed when a checkpoint is written. When a run resumes from a checkpoint at step
``t_c``, every record written *after* that checkpoint (``step >= t_c`` for step logs, ``t > t_c`` otherwise) is
moved to an ``*.orphaned-<stamp>`` sibling (never deleted), so the logs always describe one consistent
trajectory of the training process.
"""

from __future__ import annotations

import csv
import datetime as _dt
import glob
import json
import os
from typing import Any, Iterable, Optional

import numpy as np

STEP_COLUMNS = (
    "t", "episode", "ep_step", "phase", "kappa", "executed_by_planner", "iota", "obs", "executed_action",
    "behavior_action", "planner_action", "reward", "terminated", "truncated", "outcome", "clearance",
    "dist_to_goal", "yaw", "vel_body", "action_clipped", "internal_reset", "planner_iterations", "planner_ms",
    "planner_max_constraint", "planner_outside_domain", "planner_action_clipped",
)


def _stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


class StepLog:
    """Chunked per-step log (columnar)."""

    def __init__(self, directory: str, chunk_size: int, enabled: bool = True):
        self.dir = directory
        self.chunk = int(chunk_size)
        self.enabled = enabled
        os.makedirs(directory, exist_ok=True)
        self._buf: dict[str, list] = {k: [] for k in STEP_COLUMNS}

    def append(self, row: dict[str, Any]) -> None:
        if not self.enabled:
            return
        for k in STEP_COLUMNS:
            self._buf[k].append(row.get(k, np.nan))
        if len(self._buf["t"]) >= self.chunk:
            self.flush()

    def flush(self) -> Optional[str]:
        if not self.enabled or not self._buf["t"]:
            return None
        t0, t1 = int(self._buf["t"][0]), int(self._buf["t"][-1])
        path = os.path.join(self.dir, f"steps_{t0:010d}_{t1:010d}.npz")
        arrays = {}
        for k, v in self._buf.items():
            arr = np.asarray(v)
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            arrays[k] = arr
        np.savez_compressed(path, **arrays)
        self._buf = {k: [] for k in STEP_COLUMNS}
        return path

    def truncate_after(self, t_ckpt: int) -> None:
        """Move chunks with steps ``>= t_ckpt`` aside (split a straddling chunk)."""
        orphan_dir = os.path.join(self.dir, f"orphaned-{_stamp()}")
        for path in sorted(glob.glob(os.path.join(self.dir, "steps_*.npz"))):
            name = os.path.basename(path)
            t0, t1 = (int(x) for x in name[len("steps_"):-len(".npz")].split("_"))
            if t1 < t_ckpt:
                continue
            os.makedirs(orphan_dir, exist_ok=True)
            os.replace(path, os.path.join(orphan_dir, name))
            if t0 < t_ckpt:  # keep the part before the checkpoint
                d = np.load(os.path.join(orphan_dir, name))
                keep = d["t"] < t_ckpt
                kept = {k: d[k][keep] for k in d.files}
                np.savez_compressed(os.path.join(self.dir, f"steps_{t0:010d}_{int(kept['t'][-1]):010d}.npz"), **kept)


def load_steps(run_root: str, columns: Optional[Iterable[str]] = None) -> dict[str, np.ndarray]:
    """Concatenate all step chunks of a run (sorted by step)."""
    paths = sorted(glob.glob(os.path.join(run_root, "logs", "steps", "steps_*.npz")))
    out: dict[str, list] = {}
    for p in paths:
        d = np.load(p)
        for k in (columns or d.files):
            out.setdefault(k, []).append(d[k])
    return {k: np.concatenate(v) for k, v in out.items()}


class CSVLog:
    """Append-only CSV with a fixed header declared at construction."""

    def __init__(self, path: str, fieldnames: list[str]):
        self.path = path
        self.fieldnames = list(fieldnames)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new = not os.path.isfile(path)
        if not new:
            with open(path, newline="") as f:
                header = next(csv.reader(f), None)
            if header is not None and header != self.fieldnames:
                # Never rewrite a file with a different schema: start a sibling file instead.
                base, ext = os.path.splitext(path)
                self.path = f"{base}.schema-{_stamp()}{ext}"
                new = True
        self._f = open(self.path, "a", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.fieldnames, extrasaction="ignore")
        if new:
            self._w.writeheader()
            self._f.flush()

    def append(self, row: dict[str, Any]) -> None:
        self._w.writerow({k: _scalar(row.get(k, "")) for k in self.fieldnames})
        self._f.flush()

    def close(self) -> None:
        self._f.close()


class JSONLLog:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = open(path, "a")

    def append(self, row: dict[str, Any]) -> None:
        self._f.write(json.dumps({k: _scalar(v) for k, v in row.items()}) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def _scalar(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, float) and (v != v):  # NaN -> empty in CSV/JSON
        return None
    return v


def truncate_table(path: str, key: str, t_ckpt: int, strict_greater: bool = True) -> int:
    """Move rows of a CSV/JSONL file with ``row[key] > t_ckpt`` (or ``>=``) to an orphan file. Returns #moved."""
    if not os.path.isfile(path):
        return 0
    keep, drop = [], []
    if path.endswith(".jsonl"):
        with open(path) as f:
            lines = [ln for ln in f if ln.strip()]
        for ln in lines:
            v = json.loads(ln).get(key)
            bad = v is not None and (v > t_ckpt if strict_greater else v >= t_ckpt)
            (drop if bad else keep).append(ln)
        if drop:
            with open(f"{path}.orphaned-{_stamp()}", "w") as f:
                f.writelines(drop)
            with open(path, "w") as f:
                f.writelines(keep)
        return len(drop)
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return 0
    header, body = rows[0], rows[1:]
    idx = header.index(key)
    for r in body:
        try:
            v = float(r[idx])
        except (ValueError, IndexError):
            keep.append(r)
            continue
        bad = v > t_ckpt if strict_greater else v >= t_ckpt
        (drop if bad else keep).append(r)
    if drop:
        with open(f"{path}.orphaned-{_stamp()}", "w", newline="") as f:
            csv.writer(f).writerows([header] + drop)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows([header] + keep)
    return len(drop)
