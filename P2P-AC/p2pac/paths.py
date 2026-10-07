"""Locations of the files this repository ships (all relative to the repository root, so they also work inside
the Isaac Sim container, where the repository is mounted at ``/p2pac``)."""

from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Isaac Lab v2.3.0 (commit 3c6e67bb5c7ada942a6d1884ab69338f57596f77), vendored unmodified.
ISAACLAB_ROOT = os.environ.get("ISAACLAB_ROOT") or os.path.join(REPO_ROOT, "third_party", "IsaacLab")

# Frozen locomotion policy: Isaac Lab's pretrained RSL-RL checkpoint for Isaac-Velocity-Flat-Unitree-Go2-v0
# (md5 50f9fe1f7ae965ab3e7dc975156f3adb).
DEFAULT_LOCOMOTION_CHECKPOINT = os.path.join(
    REPO_ROOT, "assets", "locomotion_policy", "Isaac-Velocity-Flat-Unitree-Go2-v0", "checkpoint.pt")
