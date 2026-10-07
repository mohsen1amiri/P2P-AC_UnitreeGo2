"""Seeding and capture/restore of every random-number-generator state used during training.

Isaac Sim physics on the GPU is not bitwise deterministic, so runs on the Isaac backend are reproducible
in distribution (same seeds, same configuration) rather than bitwise; the kinematic backend is bitwise
reproducible, including across a checkpoint/resume (tests/test_train_kinematic.py).
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_rng_states() -> dict[str, Any]:
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: dict[str, Any]) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if "torch_cuda" in states and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(states["torch_cuda"])
        except RuntimeError:  # different number of visible GPUs after a requeue
            pass
