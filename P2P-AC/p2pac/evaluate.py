"""Re-evaluate saved policies of a finished (or running) run without retraining.

Every evaluation round of a training run saves a policy snapshot (``policies/policy_step_<t>.zip``) and the run
saves ``final_model.zip``; BC runs save ``policies/bc_policy.pt``. This tool evaluates any of them on a new set
of start states, e.g. to use more evaluation episodes for the final table or to recompute N80 with a different
protocol. Results go to ``<run>/eval/reeval_<tag>/`` and never overwrite existing results.

    python -m p2pac.evaluate --run-dir runs/main/p2p-sac/seed_0 --what final --episodes 200 --starts-seed 7 --tag final200 --headless
    python -m p2pac.evaluate --run-dir runs/main/p2p-sac/seed_0 --what snapshots --episodes 50 --tag curve50 --headless
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys

from p2pac.config import DEFAULTS, task_config
from p2pac.runtime.rundir import RunDir, write_json_atomic

ISAAC_AVAILABLE = importlib.util.find_spec("isaaclab") is not None


def _cfg_from_run(run_dir: str) -> argparse.Namespace:
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    merged = dict(DEFAULTS)
    merged.update(cfg)
    return argparse.Namespace(**merged)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--what", choices=("final", "snapshots", "planner"), default="final")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--starts-seed", type=int, default=None, help="default: eval_seed + 2 of the run")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=None, help="environment seed (default: run seed)")
    if ISAAC_AVAILABLE:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args(argv)
    cfg = _cfg_from_run(args.run_dir)
    run = RunDir(args.run_dir)
    out_dir = run.path("eval", f"reeval_{args.tag}")
    if os.path.exists(out_dir):
        raise SystemExit(f"{out_dir} exists -- choose a new --tag (results are never overwritten)")

    app = None
    if cfg.backend == "isaac":
        from isaaclab.app import AppLauncher

        from p2pac.isaac_assets import add_to_launcher_args

        own = {"run_dir", "what", "episodes", "starts_seed", "tag", "seed"}
        launcher_args = argparse.Namespace(**{k: v for k, v in vars(args).items() if k not in own})
        add_to_launcher_args(launcher_args)
        app = AppLauncher(launcher_args).app
    try:
        from p2pac.envs import make_env
        from p2pac.runtime.evaluation import (EPISODE_FIELDS, SUMMARY_FIELDS, PlannerController, PolicyController,
                                              run_episodes, save_trajectories, summarize)
        from p2pac.runtime.records import CSVLog
        from p2pac.runtime.seeding import seed_everything
        from p2pac.task import NavigationTask

        task = NavigationTask(task_config(cfg))
        seed = cfg.seed if args.seed is None else args.seed
        seed_everything(seed)
        env = make_env(cfg, task, seed=seed)
        starts = task.sample_starts(args.episodes, args.starts_seed if args.starts_seed is not None else cfg.eval_seed + 2)
        os.makedirs(out_dir)
        write_json_atomic(os.path.join(out_dir, "meta.json"), {"args": {k: v for k, v in vars(args).items()}, "starts": starts.tolist()})

        targets = []
        if args.what == "planner":
            from p2pac.train import build_planner_parts

            factory, amap = build_planner_parts(cfg, task)
            targets.append((0, "planner", PlannerController(factory(), amap)))
        elif cfg.method == "bc":
            from p2pac.algos.bc import load_bc_policy

            targets.append((0, "policies/bc_policy.pt", PolicyController(load_bc_policy(run.path("policies", "bc_policy.pt")))))
        else:
            from p2pac.algos import model_class

            cls = model_class(cfg.algo)
            if args.what == "final":
                targets.append((int(json.load(open(run.status_path)).get("step", 0)), "final_model.zip",
                                PolicyController(cls.load(run.path("final_model.zip"), device="cpu"))))
            else:
                for p in sorted(glob.glob(run.path("policies", "policy_step_*.zip"))):
                    t = int(os.path.basename(p)[len("policy_step_"):-len(".zip")])
                    targets.append((t, os.path.relpath(p, run.root), PolicyController(cls.load(p, device="cpu"))))

        ep_log = CSVLog(os.path.join(out_dir, "episodes.csv"), EPISODE_FIELDS)
        sum_log = CSVLog(os.path.join(out_dir, "summary.csv"), SUMMARY_FIELDS)
        for i, (t, name, ctrl) in enumerate(targets):
            rows, traj = run_episodes(env, ctrl, starts, round_index=i, t=t)
            for r in rows:
                ep_log.append(r)
            s = summarize(rows, round_index=i, t=t, t_scheduled=t, wall=0.0, snapshot=name)
            sum_log.append(s)
            save_trajectories(os.path.join(out_dir, "trajectories"), f"round_{i:04d}_step_{t:010d}", traj)
            print(f"[evaluate] {name}: success {s['success_rate']:.3f} crash {s['crash_rate']:.3f}")
        ep_log.close()
        sum_log.close()
        run.event("reevaluation", tag=args.tag, n_targets=len(targets), episodes=args.episodes)
        env.close()
        return 0
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    sys.exit(main())
