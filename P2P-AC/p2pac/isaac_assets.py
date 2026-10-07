"""Where Isaac Sim loads the robot model and textures from.

The Go2 task loads a few NVIDIA assets: the robot (``Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd`` and its meshes),
the flat ground (``Isaac/Environments/Grid/default_environment.usd`` and its textures), the sky texture and, when
videos are recorded, an arrow marker. Isaac Lab builds their paths from the Isaac Sim setting
``/persistent/isaac/asset_root/cloud``, which by default points to NVIDIA's asset server.

The repository contains a verified local copy of exactly these files (``assets/isaac_sim/``, list and checksums in
``assets/isaac_sim_manifest.json``, see ``scripts/setup/isaac_assets.py``). The environment variable
``P2PAC_ISAAC_ASSETS`` selects the source:

* ``local`` (default) -- use the local copy; stop with an explanation if it is missing or incomplete. Runs then
  need no internet access and always use the same files.
* ``nvidia`` -- download from NVIDIA's asset server, as Isaac Sim does by default.

The source used is recorded in every run's ``manifest.json`` (``simulator.assets``).
"""

from __future__ import annotations

import json
import os
from typing import Any

from p2pac.paths import REPO_ROOT

MIRROR = os.path.join(REPO_ROOT, "assets", "isaac_sim")
MANIFEST = os.path.join(REPO_ROOT, "assets", "isaac_sim_manifest.json")
SETTINGS = ("/persistent/isaac/asset_root/default", "/persistent/isaac/asset_root/cloud",
            "/persistent/isaac/asset_root/nvidia")


def mode() -> str:
    m = os.environ.get("P2PAC_ISAAC_ASSETS", "local").strip().lower()
    if m not in ("local", "nvidia"):
        raise SystemExit(f"P2PAC_ISAAC_ASSETS must be 'local' or 'nvidia', not {m!r}")
    return m


def missing_files() -> list[str]:
    """Files of the manifest that are absent or have the wrong size in the local copy (cheap check)."""
    with open(MANIFEST) as f:
        files = json.load(f)["files"]
    out = []
    for e in files:
        p = os.path.join(MIRROR, *e["path"].split("/"))
        if not os.path.isfile(p) or os.path.getsize(p) != e["bytes"]:
            out.append(e["path"])
    return out


def kit_args() -> list[str]:
    """Kit command-line settings that make Isaac Sim (and Isaac Lab) use the selected asset source."""
    if mode() == "nvidia":
        return []
    missing = missing_files()
    if missing:
        raise SystemExit(
            "The local copy of the Isaac Sim assets is incomplete (missing: " + ", ".join(missing[:3])
            + (" ..." if len(missing) > 3 else "") + ").\n"
            "  Restore it with:  python3 scripts/setup/isaac_assets.py download\n"
            "  or use NVIDIA's asset server for this run:  export P2PAC_ISAAC_ASSETS=nvidia")
    if any(c.isspace() for c in MIRROR):
        raise SystemExit(f"The repository path contains a space, which Isaac Sim's command line cannot handle: {MIRROR}")
    return [f"--{s}={MIRROR}" for s in SETTINGS]


def add_to_launcher_args(launcher_args: Any) -> None:
    """Append the asset settings to the ``kit_args`` of an AppLauncher argument namespace (in place)."""
    extra = kit_args()
    if extra:
        current = getattr(launcher_args, "kit_args", "") or ""
        launcher_args.kit_args = " ".join([current, *extra]).strip()


def describe() -> dict:
    """Recorded in the run manifest."""
    m = os.environ.get("P2PAC_ISAAC_ASSETS", "local").strip().lower()
    info: dict[str, Any] = {"mode": m}
    if m == "local":
        info["root"] = MIRROR
        try:
            with open(MANIFEST, "rb") as f:
                import hashlib

                info["manifest_sha256"] = hashlib.sha256(f.read()).hexdigest()
            info["missing_files"] = missing_files()
        except OSError:
            info["manifest_sha256"] = None
    else:
        info["root"] = "NVIDIA asset server (Isaac Sim default)"
    return info
