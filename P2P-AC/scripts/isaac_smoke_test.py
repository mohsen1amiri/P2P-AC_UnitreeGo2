"""Isaac Lab acceptance test -- run it once on the cluster before launching any experiment.

    p2pac_run python scripts/isaac_smoke_test.py --headless --out isaac_smoke.json

Checks (each prints PASS/FAIL and is stored in the JSON report):

0. ``locomotion_checkpoint`` -- the shipped locomotion policy is the expected file (md5 50f9fe1f...);
1. ``no_internal_reset``    -- the robot stands for 1200 control steps (24 s, longer than Isaac's default 20 s
                                episode) with zero command: no teleport and no internal reset;
2. ``command_slice``       -- the velocity-command slice of the locomotion observation is the 'velocity_commands'
                                term (indices 9:12 for the Go2 flat task);
3. ``velocity_tracking``   -- a constant body command 0.4 m/s along x moves the robot along world +y
                                (heading 90 deg) at roughly the commanded speed;
4. ``contact_detection``   -- driving into the entry obstacle ends in a failure outcome;
5. ``planner_episode``     -- the planner reaches the goal from a fixed start, every in-domain action feasible.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default="isaac_smoke.json")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
from p2pac.isaac_assets import add_to_launcher_args  # noqa: E402

add_to_launcher_args(args)        # robot model and textures from assets/isaac_sim (P2PAC_ISAAC_ASSETS)
app = AppLauncher(args).app

import numpy as np  # noqa: E402

from p2pac.config import DEFAULTS, reap_config, task_config  # noqa: E402
from p2pac.envs import make_env  # noqa: E402
from p2pac.planners import PlannerActionMap, make_planner  # noqa: E402
from p2pac.task import NavigationTask, Outcome  # noqa: E402

cfg = argparse.Namespace(**dict(DEFAULTS))
cfg.backend = "isaac"
cfg.app_launcher = {"device": getattr(args, "device", "cuda:0")}
task = NavigationTask(task_config(cfg))
env = make_env(cfg, task, seed=args.seed)
report = {"backend": env.describe()}


def check(name, ok, **info):
    report[name] = {"pass": bool(ok), **info}
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)


# 0 -- the locomotion policy is the expected checkpoint
import hashlib  # noqa: E402

ckpt = report["backend"]["backend"]["locomotion_checkpoint"]
md5 = hashlib.md5(open(ckpt, "rb").read()).hexdigest()
check("locomotion_checkpoint", md5 == "50f9fe1f7ae965ab3e7dc975156f3adb", path=ckpt, md5=md5)


# 1 -- no internal reset during a long episode (> 20 s)
env.task.cfg.max_episode_steps = 10_000
obs, _ = env.reset(options={"start_xy": (-1.0, -1.2)})
start = env.state.xy.copy()
resets, max_jump, prev = 0, 0.0, start.copy()
for _ in range(1200):
    obs, r, term, trunc, info = env.step(np.zeros(2))
    resets += int(info["internal_reset"])
    max_jump = max(max_jump, float(np.linalg.norm(env.state.xy - prev)))
    prev = env.state.xy.copy()
    if term:
        break
check("no_internal_reset", resets == 0 and not term and max_jump < 0.05, internal_resets=resets, max_step_jump=max_jump,
      drift=float(np.linalg.norm(env.state.xy - start)))

# 2 -- command slice
sl = report["backend"]["backend"]["velocity_command_slice"]
check("command_slice", sl == [9, 12], slice=sl, layout=report["backend"]["backend"]["policy_obs_layout"])

# 3 -- velocity tracking along the heading
obs, _ = env.reset(options={"start_xy": (-1.0, -1.5)})
p0 = env.state.xy.copy()
for _ in range(100):
    obs, r, term, trunc, info = env.step(np.array([0.4, 0.0]))
d = env.state.xy - p0
check("velocity_tracking", d[1] > 0.4 * 2.0 * 0.6 and abs(d[0]) < 0.3, displacement=d.tolist(), expected=[0.0, 0.8])

# 4 -- contact / collision detection
obs, _ = env.reset(options={"start_xy": (0.0, -1.0)})
outcome = None
for _ in range(500):
    obs, r, term, trunc, info = env.step(np.array([0.5, 0.0]))
    if term or trunc:
        outcome = Outcome(info["outcome"])
        break
check("contact_detection", outcome is not None and outcome.is_failure, outcome=str(outcome))

# 4b -- informational: with the geometric collision margin disabled, does the trunk contact sensor fire?
saved = (env.task.cfg.robot_radius, env.task.cfg.leg_reach_margin)
env.task.cfg.robot_radius, env.task.cfg.leg_reach_margin = 0.0, 0.0
env.task._coll = env.task.cfg.collision_radius
obs, _ = env.reset(options={"start_xy": (0.0, -1.0)})
outcome_b = None
for _ in range(600):
    obs, r, term, trunc, info = env.step(np.array([0.5, 0.0]))
    if term or trunc:
        outcome_b = Outcome(info["outcome"])
        break
report["contact_sensor_info"] = {"outcome_without_geometric_margin": str(outcome_b)}
print(f"[INFO] contact sensor test: {outcome_b}", flush=True)
env.task.cfg.robot_radius, env.task.cfg.leg_reach_margin = saved
env.task._coll = env.task.cfg.collision_radius

# 5 -- planner episode
env.task.cfg.max_episode_steps = 8000
planner = make_planner("reap", reap_config(cfg), task.cfg)
amap = PlannerActionMap(math.radians(90.0), [-0.7, -0.7], [0.7, 0.7])
obs, _ = env.reset(options={"start_xy": (-0.8, -1.4)})
planner.reset()
cmax, outcome, steps = -1e9, None, 0
while True:
    out = planner.plan(obs[:2])
    if not out.outside_domain:
        cmax = max(cmax, out.max_constraint)
    u, _ = amap(out)
    obs, r, term, trunc, info = env.step(u)
    steps += 1
    if term or trunc:
        outcome = Outcome(info["outcome"])
        break
check("planner_episode", outcome == Outcome.GOAL and cmax <= 1e-9, outcome=str(outcome), steps=steps, max_constraint_in_SP=cmax)

with open(args.out, "w") as f:
    json.dump(report, f, indent=2, default=str)
ok = all(v.get("pass", True) for v in report.values() if isinstance(v, dict) and "pass" in v)
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED", flush=True)
env.close()
app.close()
sys.exit(0 if ok else 1)
