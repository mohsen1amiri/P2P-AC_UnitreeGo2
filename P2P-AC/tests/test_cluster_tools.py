"""Cluster tooling: plain-Python sweep generation (login nodes), SLURM headers for each cluster profile."""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))


def test_configuration_does_not_need_numpy():
    code = "import sys; sys.path.insert(0, %r); import p2pac.config; assert 'numpy' not in sys.modules" % REPO
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def _profile_env(path):
    env = {}
    for line in open(path):
        line = line.strip()
        if line.startswith("export P2PAC_SLURM_"):
            k, v = line[len("export "):].split("=", 1)
            env[k] = v.split("#")[0].strip().strip('"')
    return env


def _header(monkeypatch, profile):
    import make_sweep

    for var in make_sweep.CLUSTER_KEYS.values():
        monkeypatch.delenv(var, raising=False)
    for k, v in _profile_env(os.path.join(REPO, "scripts", "clusters", profile)).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(make_sweep, "_cluster_env", lambda: {})
    spec = make_sweep.load_spec(os.path.join(REPO, "configs", "experiments", "main.yaml"))

    class Args:
        account = partition = runs_per_job = None

    sl = make_sweep.slurm_settings(spec, Args())
    return make_sweep.resource_lines(sl), sl


def test_arrhenius_profile(monkeypatch):
    lines, sl = _header(monkeypatch, "arrhenius.env")
    assert "#SBATCH --partition=gpu\n" in lines and "#SBATCH --gpus=1\n" in lines
    assert "--cpus-per-task=" in lines and "--mem=" in lines and "--gpus-per-node" not in lines
    assert sl["account"].endswith("-gpu") and sl["time"] == "3-00:00:00"


def test_alvis_profile(monkeypatch):
    lines, sl = _header(monkeypatch, "alvis.env")
    assert "#SBATCH --gpus-per-node=A40:1\n" in lines and "--gpus=" not in lines
    assert sl["time"] == "7-00:00:00"


def test_requirements_markers_for_both_architectures():
    import pytest

    try:
        from packaging.markers import Marker
    except ImportError:
        Marker = pytest.importorskip("pip._vendor.packaging.markers").Marker

    lines = [ln.split("#")[0].strip() for ln in open(os.path.join(REPO, "requirements", "isaac-overlay.txt"))]
    lines = [ln for ln in lines if ln]
    for arch, n in (("x86_64", 56), ("aarch64", 55)):
        keep = [ln for ln in lines if ";" not in ln or Marker(ln.split(";", 1)[1].strip()).evaluate({"platform_machine": arch})]
        assert len(keep) == n, (arch, len(keep))
