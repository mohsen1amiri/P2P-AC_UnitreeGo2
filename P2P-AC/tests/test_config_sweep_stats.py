import os
import sys

import numpy as np
import pytest

from p2pac.analysis.stats import n_to_threshold, summarize_n80, wilson_interval
from p2pac.config import build_parser, config_hash, default_run_dir, resolve

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(*argv):
    return resolve(build_parser().parse_args(list(argv)))


def test_presets_and_resolution_order(tmp_path):
    c = _cfg("--method", "p2p", "--algo", "sac")
    assert c.buffer == "dual" and c.planner == "reap" and c.buffer_size == 2_000_000 and c.planner_buffer_capacity == 1_000_000
    b = _cfg("--method", "base", "--algo", "td3")
    assert b.planner == "none" and b.buffer == "single" and b.kappa0 == 0
    a = _cfg("--method", "accel", "--algo", "sac")
    assert a.execute_planner is False and a.imitation_space == "action_sample" and a.t_d == 50_000
    f = tmp_path / "c.yaml"
    f.write_text("kappa0: 3.0\nt_p: 5000\nplanner_buffer_capacity: 5000\n")
    c2 = _cfg("--method", "p2p", "--algo", "sac", "--config", str(f), "--kappa0", "4")
    assert c2.t_p == 5000 and c2.kappa0 == 4.0          # CLI overrides the file, file overrides the preset


def test_validation_rejects_inconsistent_settings():
    with pytest.raises(SystemExit):
        _cfg("--method", "p2p", "--algo", "sac", "--planner-buffer-capacity", "10", "--t-p", "100")
    with pytest.raises(SystemExit):
        _cfg("--method", "accel", "--algo", "td3")
    with pytest.raises(SystemExit):
        _cfg("--method", "p2p", "--algo", "ppo")


def test_run_dir_and_hash_are_deterministic():
    c = _cfg("--method", "p2p", "--algo", "sac", "--seed", "3", "--experiment", "main", "--variant", "x")
    assert default_run_dir(c).endswith(os.path.join("main", "p2p-sac__x", "seed_3"))
    d = _cfg("--method", "p2p", "--algo", "sac", "--seed", "4", "--experiment", "main", "--variant", "x")
    assert config_hash(c) == config_hash(d)             # the seed does not change the experiment hash
    e = _cfg("--method", "p2p", "--algo", "sac", "--seed", "3", "--experiment", "main", "--variant", "x", "--kappa0", "1")
    assert config_hash(c) != config_hash(e)


def test_sweep_expansion():
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from make_sweep import expand, load_spec

    spec = load_spec(os.path.join(REPO, "configs", "experiments", "main.yaml"))
    jobs = expand(spec)
    assert len(jobs) == 8 * 5
    assert all("--headless" in j["command"] for j in jobs)
    names = {j["name"] for j in jobs}
    assert names == {"base-ddpg", "p2p-ddpg", "base-td3", "p2p-td3", "base-sac", "p2p-sac", "accel-sac", "base-ppo"}
    for f in ("ablation.yaml", "budget.yaml", "planner_reference.yaml", "bc.yaml", "smoke_kinematic.yaml"):
        assert len(expand(load_spec(os.path.join(REPO, "configs", "experiments", f)))) > 0


def test_n80_and_censoring():
    t = np.array([0, 100, 200, 300])
    assert n_to_threshold(t, np.array([0.0, 0.5, 0.8, 1.0])) == 200
    assert n_to_threshold(t, np.array([0.0, 0.5, 0.7, 0.75])) is None
    s = summarize_n80([200.0, None, 300.0], budget=1000)
    assert s["n_reached"] == 2 and s["n_censored"] == 1 and s["mean"] == 250 and s["mean_censored_as_budget"] == 500


def test_n80_requires_two_consecutive_evaluations():
    t = np.array([0, 50_000, 100_000, 150_000, 200_000])

    # An isolated 80% crossing does not count.
    # The first confirmed crossing is at 150k.
    success = np.array([0.0, 0.80, 0.70, 0.85, 0.80])
    assert n_to_threshold(t, success, step_interval=50_000) == 150_000

    # No two consecutive evaluations reach 80%.
    success = np.array([0.0, 0.85, 0.70, 0.75, 0.80])
    assert n_to_threshold(t, success, step_interval=50_000) is None

    # Missing an evaluation does not count as consecutive.
    t_gap = np.array([0, 50_000, 150_000])
    success_gap = np.array([0.1, 0.9, 0.9])
    assert n_to_threshold(t_gap, success_gap, step_interval=50_000) is None

def test_wilson_interval():
    lo, hi = wilson_interval(20, 20)
    assert hi == pytest.approx(1.0) and 0.8 < lo < 0.9
    lo, hi = wilson_interval(0, 0)
    assert np.isnan(lo)
