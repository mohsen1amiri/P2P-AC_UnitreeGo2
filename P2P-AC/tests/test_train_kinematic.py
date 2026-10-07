"""End-to-end tests of the training entry point on the kinematic backend (no Isaac Sim needed).

* phase semantics of P2P-AC are visible in the step log (Eq. (16), Eq. (17), planner removal);
* preemption (SIGUSR1) writes a checkpoint and exits with code 3; the identical command resumes the run;
* after resuming, every log is a single consistent trajectory (no duplicated or missing steps);
* a finished run is never overwritten (re-running the command is a no-op).
"""

import os
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

from p2pac.runtime.records import load_steps

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _args(out, seed=0, total=3000, extra=()):
    return [
        sys.executable, "-m", "p2pac.train", "--method", "p2p", "--algo", "sac", "--backend", "kinematic",
        "--seed", str(seed), "--experiment", "test", "--output-root", str(out), "--total-steps", str(total),
        "--t-p", "1000", "--t-d", "500", "--eval-every", "1000", "--eval-episodes", "3", "--eval-final-episodes", "3",
        "--max-episode-steps", "1500", "--checkpoint-every", "700", "--learner-log-every", "200",
        "--planner-buffer-capacity", "2000", "--buffer-size", "10000", "--batch-size", "64", "--net-arch", "32", "32",
        *extra,
    ]


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_dir(out, seed=0):
    return os.path.join(out, "test", "p2p-sac", f"seed_{seed}")


def test_phases_in_step_log(tmp_path):
    subprocess.run(_args(tmp_path), check=True, env=_env(), cwd=REPO, capture_output=True)
    d = load_steps(_run_dir(tmp_path))
    t = d["t"]
    assert np.all(np.diff(t) == 1) and t[0] == 0 and t[-1] == 2999
    assert d["executed_by_planner"][t < 1000].all() and not d["executed_by_planner"][t >= 1000].any()
    assert d["iota"][t < 1500].all() and not d["iota"][t >= 1500].any()        # planner removed at T_r
    np.testing.assert_allclose(d["executed_action"][t < 1000], d["planner_action"][t < 1000], atol=1e-6)
    np.testing.assert_allclose(d["kappa"][[0, 999, 1250, 1500, 2999]], [10, 10, 5, 0, 0], atol=1e-6)
    ev = pd.read_json(os.path.join(_run_dir(tmp_path), "events.jsonl"), lines=True)
    removed = ev[ev.event == "planner_removed"].iloc[0]
    assert removed.queries_after_removal == 0
    # A finished run is skipped, never overwritten.
    before = os.path.getmtime(os.path.join(_run_dir(tmp_path), "status.json"))
    subprocess.run(_args(tmp_path), check=True, env=_env(), cwd=REPO, capture_output=True)
    assert os.path.getmtime(os.path.join(_run_dir(tmp_path), "status.json")) == before


def test_preemption_and_resume(tmp_path):
    proc = subprocess.Popen(_args(tmp_path, seed=1, total=6000), env=_env(), cwd=REPO,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    run = _run_dir(tmp_path, seed=1)
    deadline = time.time() + 120
    while time.time() < deadline:   # wait until a first checkpoint exists, then preempt
        if os.path.isfile(os.path.join(run, "checkpoints", "LATEST")):
            break
        time.sleep(0.2)
    time.sleep(1.0)
    proc.send_signal(signal.SIGUSR1)
    out, _ = proc.communicate(timeout=300)
    assert proc.returncode == 3, out.decode()[-2000:]
    assert pd.read_json(os.path.join(run, "status.json"), typ="series")["state"] == "preempted"

    res = subprocess.run(_args(tmp_path, seed=1, total=6000), env=_env(), cwd=REPO, capture_output=True)
    assert res.returncode == 0, res.stdout.decode()[-2000:] + res.stderr.decode()[-2000:]
    d = load_steps(run)
    t = d["t"]
    assert t[0] == 0 and t[-1] == 5999
    assert np.all(np.diff(t) == 1), "step log must be one consistent trajectory after resume"
    ep = pd.read_csv(os.path.join(run, "logs", "episodes.csv"))
    assert ep.t_end.is_monotonic_increasing
    # the episode log covers every step exactly once (the episode cut by the preemption is logged as aborted)
    assert ep.t_start.iloc[0] == 0 and (ep.t_start.iloc[1:].to_numpy() == ep.t_end.iloc[:-1].to_numpy()).all()
    assert ep.episode.tolist() == list(range(len(ep))), "episode ids must stay unique and consecutive"
    ev = pd.read_json(os.path.join(run, "events.jsonl"), lines=True)
    assert (ev.event == "resumed").sum() == 1 and (ev.event == "preempted").sum() == 1
    status = pd.read_json(os.path.join(run, "status.json"), typ="series")
    assert status["state"] == "completed" and int(status["step"]) == 6000
