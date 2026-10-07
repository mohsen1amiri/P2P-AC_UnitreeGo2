"""Self-containment checks: shipped simulator assets, locomotion policy, pinned packages, provenance without git."""

import hashlib
import json
import os
import subprocess
import sys

import pytest

from p2pac import isaac_assets
from p2pac.paths import DEFAULT_LOCOMOTION_CHECKPOINT, ISAACLAB_ROOT, REPO_ROOT


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_shipped_isaac_assets_match_the_manifest():
    with open(isaac_assets.MANIFEST) as f:
        man = json.load(f)
    assert len(man["files"]) == 12
    for e in man["files"]:
        p = os.path.join(isaac_assets.MIRROR, *e["path"].split("/"))
        assert os.path.isfile(p), e["path"]
        assert os.path.getsize(p) == e["bytes"] and _sha256(p) == e["sha256"], e["path"]
    assert isaac_assets.missing_files() == []


def test_asset_source_selection(monkeypatch):
    monkeypatch.setenv("P2PAC_ISAAC_ASSETS", "nvidia")
    assert isaac_assets.kit_args() == []
    monkeypatch.setenv("P2PAC_ISAAC_ASSETS", "local")
    if any(c.isspace() for c in isaac_assets.MIRROR):
        with pytest.raises(SystemExit):          # Isaac Sim's command line cannot take paths with spaces
            isaac_assets.kit_args()
    else:
        args = isaac_assets.kit_args()
        assert args == [f"--{s}={isaac_assets.MIRROR}" for s in isaac_assets.SETTINGS]
    monkeypatch.setenv("P2PAC_ISAAC_ASSETS", "elsewhere")
    with pytest.raises(SystemExit):
        isaac_assets.mode()


def test_locomotion_policy_and_isaaclab_are_shipped():
    with open(DEFAULT_LOCOMOTION_CHECKPOINT, "rb") as f:
        assert hashlib.md5(f.read()).hexdigest() == "50f9fe1f7ae965ab3e7dc975156f3adb"
    with open(os.path.join(ISAACLAB_ROOT, "VERSION")) as f:
        assert f.read().strip() == "2.3.0"
    for pkg in ("isaaclab", "isaaclab_tasks", "isaaclab_rl", "isaaclab_assets"):
        assert os.path.isdir(os.path.join(ISAACLAB_ROOT, "source", pkg, pkg)), pkg
    assert os.path.isdir(os.path.join(ISAACLAB_ROOT, "apps"))


def test_overlay_requirements_are_exact_pins():
    with open(os.path.join(REPO_ROOT, "requirements", "isaac-overlay.txt")) as f:
        pins = [ln.split("#")[0].strip() for ln in f if ln.split("#")[0].strip()]
    assert len(pins) == 56 and all("==" in p for p in pins)


def test_git_state_recorded_on_the_host_is_used_without_git(tmp_path):
    state = tmp_path / "git_state"
    state.mkdir()
    (state / "commit").write_text("0123456789abcdef\n")
    (state / "branch").write_text("main\n")
    (state / "describe").write_text("0123456-dirty\n")
    (state / "status").write_text(" M README.md\n")
    (state / "diff.patch").write_text("--- a\n+++ b\n")
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from p2pac.runtime.metadata import git_info, git_diff\n"
            "i = git_info(%r); assert i['commit'] == '0123456789abcdef' and i['dirty'] and i['branch'] == 'main', i\n"
            "assert git_diff(%r).startswith('--- a')\n") % (REPO_ROOT, str(tmp_path), str(tmp_path))
    env = dict(os.environ, P2PAC_GIT_STATE_DIR=str(state), PATH="/nonexistent")   # no git available
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=str(tmp_path))
    assert res.returncode == 0, res.stderr
