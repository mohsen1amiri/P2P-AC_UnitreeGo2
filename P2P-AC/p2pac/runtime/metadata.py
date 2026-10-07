"""Provenance metadata recorded in ``manifest.json`` for every run."""

from __future__ import annotations

import importlib
import os
import platform
import socket
import subprocess
import sys
from typing import Optional

from p2pac.runtime.rundir import utcnow
from p2pac.version import __version__

PACKAGES = ("numpy", "scipy", "torch", "gymnasium", "stable_baselines3", "pandas", "yaml",
            "isaaclab", "isaaclab_tasks", "isaaclab_rl", "isaacsim", "rsl_rl", "tensordict")
SLURM_VARS = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST",
              "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT", "SLURM_RESTART_COUNT", "SLURM_CPUS_PER_TASK")
ENV_VARS = ("CUDA_VISIBLE_DEVICES", "ISAACSIM_SIF", "ISAACLAB_ROOT", "P2PAC_ROOT", "OMP_NUM_THREADS")


def _git(args: list[str], cwd: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _host_git_state(name: str) -> Optional[str]:
    """Git state recorded on the host by scripts/container.sh (the container image may have no git)."""
    d = os.environ.get("P2PAC_GIT_STATE_DIR")
    if not d:
        return None
    try:
        with open(os.path.join(d, name)) as f:
            return f.read().strip("\n")
    except OSError:
        return None


def git_info(repo_dir: str) -> dict:
    commit = _git(["rev-parse", "HEAD"], repo_dir)
    if commit is None and _host_git_state("commit") is not None:
        status = _host_git_state("status")
        return {"commit": _host_git_state("commit") or None, "branch": _host_git_state("branch") or None,
                "dirty": bool(status) if status is not None else None,
                "describe": _host_git_state("describe") or None, "recorded_by": "host (scripts/container.sh)"}
    status = _git(["status", "--porcelain"], repo_dir)
    return {
        "commit": commit,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir),
        "dirty": bool(status) if status is not None else None,
        "describe": _git(["describe", "--always", "--dirty", "--tags"], repo_dir),
        "recorded_by": "git" if commit is not None else "unavailable",
    }


def git_diff(repo_dir: str) -> Optional[str]:
    diff = _git(["diff", "HEAD"], repo_dir)
    if diff is None:
        diff = _host_git_state("diff.patch")
    return diff


def package_versions() -> dict:
    out = {}
    for name in PACKAGES:
        mod = sys.modules.get(name)
        if mod is None:
            try:
                if name in ("isaaclab", "isaaclab_tasks", "isaaclab_rl", "isaacsim"):
                    continue  # importing these launches/needs Isaac Sim; record only if already imported
                mod = importlib.import_module(name)
            except Exception:
                continue
        out[name] = getattr(mod, "__version__", "unknown")
    try:
        from importlib.metadata import version

        for dist in ("rsl-rl-lib", "isaaclab", "isaacsim"):
            try:
                out[f"dist:{dist}"] = version(dist)
            except Exception:
                pass
    except ImportError:  # pragma: no cover
        pass
    return out


def hardware_info() -> dict:
    info = {"hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(), "processor": platform.processor(), "machine": platform.machine()}
    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["cudnn"] = torch.backends.cudnn.version()
    except Exception:  # pragma: no cover
        pass
    return info


def simulator_info() -> dict:
    """Versions of the simulator stack (vendored Isaac Lab, container image) -- cheap, no hashing of the image."""
    from p2pac.paths import ISAACLAB_ROOT

    info: dict = {"isaaclab_root": ISAACLAB_ROOT}
    try:
        with open(os.path.join(ISAACLAB_ROOT, "VERSION")) as f:
            info["isaaclab_version"] = f.read().strip()
    except OSError:
        info["isaaclab_version"] = None
    sif = os.environ.get("ISAACSIM_SIF")
    if sif:
        try:
            st = os.stat(sif)
            info["isaacsim_image"] = {"path": sif, "bytes": st.st_size, "mtime": st.st_mtime}
        except OSError:
            info["isaacsim_image"] = {"path": sif}
    try:
        from p2pac.isaac_assets import describe

        info["assets"] = describe()
    except Exception as e:  # never fail a run because of provenance bookkeeping
        info["assets"] = {"error": repr(e)}
    try:
        with open("/isaac-sim/VERSION") as f:
            info["isaacsim_version"] = f.read().strip()
    except OSError:
        pass
    return info


def collect_manifest(repo_dir: str, extra: Optional[dict] = None) -> dict:
    m = {
        "p2pac_version": __version__,
        "created_utc": utcnow(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "git": git_info(repo_dir),
        "packages": package_versions(),
        "hardware": hardware_info(),
        "simulator": simulator_info(),
        "slurm": {k: os.environ[k] for k in SLURM_VARS if k in os.environ},
        "env": {k: os.environ[k] for k in ENV_VARS if k in os.environ},
    }
    if extra:
        m.update(extra)
    return m


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
