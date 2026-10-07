"""Checkpoints for exact resumption of training.

A checkpoint ``checkpoints/step_<t>/`` contains

* ``model.zip``          -- SB3 model: policy, target networks, optimiser states, counters (``num_timesteps``,
  ``_n_updates``, ...) and the P2P-AC configuration;
* ``replay_buffer.pkl``  -- the complete replay buffer (both ``D_P`` and ``D_O``) for off-policy learners;
* ``trainer_state.pkl``  -- global RNG states, environment/back-end RNG states and the scheduler state of the
  training callback (next evaluation, next checkpoint, phase flags, counters);
* ``meta.json``          -- step, wall-clock time, milestone flag.

It is written at the beginning of a rollout (after the previous gradient update), so that model, buffer and
step counter are mutually consistent. Writing is atomic (``.tmp`` directory renamed at the end) and
``checkpoints/LATEST`` points to the newest complete checkpoint. Old non-milestone checkpoints are pruned
(``keep_checkpoints``); milestones (``T_p``, ``T_r``, final) are kept.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
from typing import Any, Optional

from p2pac.runtime.rundir import RunDir, utcnow


class CheckpointManager:
    def __init__(self, run: RunDir, keep: int = 2):
        self.run = run
        self.keep = max(1, int(keep))
        self.dir = run.path("checkpoints")
        os.makedirs(self.dir, exist_ok=True)

    def save(self, model, t: int, trainer_state: dict, *, milestone: bool = False, tag: str = "") -> str:
        name = f"step_{t:010d}"
        final = os.path.join(self.dir, name)
        tmp = final + ".tmp"
        if os.path.isdir(final):              # a milestone saved earlier at the same step stays a milestone
            old = _read_meta(final)
            milestone = milestone or bool(old.get("milestone", False))
            tag = tag or old.get("tag", "")
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
        model.save(os.path.join(tmp, "model.zip"))
        rb = getattr(model, "replay_buffer", None)
        if rb is not None:
            with open(os.path.join(tmp, "replay_buffer.pkl"), "wb") as f:
                pickle.dump(rb, f, protocol=4)
        with open(os.path.join(tmp, "trainer_state.pkl"), "wb") as f:
            pickle.dump(trainer_state, f, protocol=4)
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump({"t": t, "milestone": milestone, "tag": tag, "time_utc": utcnow()}, f, indent=2)
        if os.path.isdir(final):              # same step saved twice (e.g. milestone + periodic): keep the newest
            shutil.rmtree(final)
        os.replace(tmp, final)
        with open(os.path.join(self.dir, "LATEST.tmp"), "w") as f:
            f.write(name)
        os.replace(os.path.join(self.dir, "LATEST.tmp"), os.path.join(self.dir, "LATEST"))
        self._prune()
        return final

    def _prune(self) -> None:
        entries = []
        for name in sorted(os.listdir(self.dir)):
            p = os.path.join(self.dir, name)
            if not (name.startswith("step_") and os.path.isdir(p)) or name.endswith(".tmp"):
                continue
            meta = _read_meta(p)
            entries.append((name, bool(meta.get("milestone", False))))
        regular = [n for n, m in entries if not m]
        for name in regular[:-self.keep]:
            shutil.rmtree(os.path.join(self.dir, name), ignore_errors=True)


def _read_meta(path: str) -> dict:
    try:
        with open(os.path.join(path, "meta.json")) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def load_trainer_state(ckpt_dir: str) -> dict:
    with open(os.path.join(ckpt_dir, "trainer_state.pkl"), "rb") as f:
        return pickle.load(f)


def load_replay_buffer(ckpt_dir: str, device) -> Optional[Any]:
    p = os.path.join(ckpt_dir, "replay_buffer.pkl")
    if not os.path.isfile(p):
        return None
    with open(p, "rb") as f:
        rb = pickle.load(f)
    rb.device = device
    for sub in ("planner", "online"):
        if hasattr(rb, sub):
            getattr(rb, sub).device = device
    return rb


def checkpoint_step(ckpt_dir: str) -> int:
    return int(_read_meta(ckpt_dir).get("t", int(os.path.basename(ckpt_dir).split("_")[-1])))
