
"""Re-evaluate saved policies of a finished (or running) run without retraining.

Every evaluation round of a training run saves a policy snapshot (``policies/policy_step_<t>.zip``) and the run
saves ``final_model.zip``; BC runs save ``policies/bc_policy.pt``. This tool evaluates any of them on a new set
of start states, e.g. to use more evaluation episodes for the final table or to recompute N80 with a different
protocol. Results go to ``<run>/eval/reeval_<tag>/``; completed rounds are preserved on restart.

    python -m p2pac.evaluate --run-dir runs/main/p2p-sac/seed_0 --what final --episodes 200 --starts-seed 7 --tag final200 --headless
    python -m p2pac.evaluate --run-dir runs/main/p2p-sac/seed_0 --what snapshots --episodes 50 --tag curve50 --headless
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import os
import shutil
import sys
import time

from p2pac.config import DEFAULTS, task_config
from p2pac.runtime.rundir import RunDir, write_json_atomic

ISAAC_AVAILABLE = importlib.util.find_spec("isaaclab") is not None


def _cfg_from_run(run_dir: str) -> argparse.Namespace:
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    merged = dict(DEFAULTS)
    merged.update(cfg)
    return argparse.Namespace(**merged)


def _completed_rounds(out_dir, targets, n_episodes, episode_fields, summary_fields):
    """Keep only complete rounds; preserve interrupted CSV tails as backup files."""
    ep_path = os.path.join(out_dir, "episodes.csv")
    sum_path = os.path.join(out_dir, "summary.csv")

    def read_csv(path, fields):
        if not os.path.isfile(path):
            return []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != fields:
                raise SystemExit(f"Unexpected CSV columns in {path}; use a different tag")
            return list(reader)

    episodes = read_csv(ep_path, episode_fields)
    summaries = read_csv(sum_path, summary_fields)

    # A different target set must never silently reuse or overwrite completed results.
    for i, row in enumerate(summaries):
        if (i >= len(targets) or row.get("round") != str(i)
                or row.get("t") != str(targets[i][0])
                or row.get("policy_snapshot") != targets[i][1]
                or row.get("n_episodes") != str(n_episodes)):
            raise SystemExit("Existing evaluation summary does not match the targets; use a new tag")

    completed = 0
    for i, (t, _, _) in enumerate(targets[:len(summaries)]):
        rows = episodes[i * n_episodes:(i + 1) * n_episodes]
        traj = os.path.join(out_dir, "trajectories", f"round_{i:04d}_step_{t:010d}.npz")
        if (len(rows) != n_episodes
                or any(r.get("round") != str(i) or r.get("t") != str(t)
                       or r.get("eval_episode") != str(j) for j, r in enumerate(rows))
                or not os.path.isfile(traj)):
            break
        completed += 1

    # Episodes are written before summaries, so an interrupted round can leave extra episode rows.
    for path, fields, rows, n_keep in (
        (ep_path, episode_fields, episodes, completed * n_episodes),
        (sum_path, summary_fields, summaries, completed),
    ):
        if len(rows) != n_keep:
            shutil.copy2(path, f"{path}.partial.{time.time_ns()}")
            tmp = path + ".resume.tmp"
            with open(tmp, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows[:n_keep])
            os.replace(tmp, path)

    print(f"[evaluate] {completed} previously completed rounds", flush=True)
    return completed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--what", choices=("final", "snapshots", "planner"), default="final")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--starts-seed", type=int, default=None,
                    help="default: eval_seed for snapshots, eval_seed + 1 for planner, eval_seed + 2 for final")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=None, help="environment seed (default: run seed)")
    if ISAAC_AVAILABLE:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args(argv)
    cfg = _cfg_from_run(args.run_dir)
    run = RunDir(args.run_dir)
    out_dir = run.path("eval", f"reeval_{args.tag}")

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
        if args.starts_seed is not None:
            starts_seed = args.starts_seed
        elif args.what == "snapshots":
            starts_seed = cfg.eval_seed
        elif args.what == "planner":
            starts_seed = cfg.eval_seed + 1
        else:
            starts_seed = cfg.eval_seed + 2

        starts = task.sample_starts(args.episodes, starts_seed)
        os.makedirs(out_dir, exist_ok=True)
        meta_path = os.path.join(out_dir, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                previous = json.load(f)
            if (previous.get("starts_seed") != starts_seed
                    or previous.get("starts") != starts.tolist()
                    or any(previous.get("args", {}).get(k) != getattr(args, k)
                           for k in ("what", "episodes", "seed"))):
                raise SystemExit("Existing evaluation tag has different settings; use a new tag")
        else:
            if os.listdir(out_dir):
                raise SystemExit("Evaluation directory exists without metadata; use a new tag")
            write_json_atomic(meta_path, {"args": vars(args), "starts_seed": starts_seed, "starts": starts.tolist()})

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
                                run.path("final_model.zip")))
            else:
                for p in sorted(glob.glob(run.path("policies", "policy_step_*.zip"))):
                    t = int(os.path.basename(p)[len("policy_step_"):-len(".zip")])
                    targets.append((t, os.path.relpath(p, run.root), p))

        resume_from = _completed_rounds(out_dir, targets, len(starts), EPISODE_FIELDS, SUMMARY_FIELDS)
        ep_log = CSVLog(os.path.join(out_dir, "episodes.csv"), EPISODE_FIELDS)
        sum_log = CSVLog(os.path.join(out_dir, "summary.csv"), SUMMARY_FIELDS)
        for i, (t, name, ctrl) in enumerate(targets):
            if i < resume_from:
                continue
            if isinstance(ctrl, str):
                ctrl = PolicyController(cls.load(ctrl, device="cpu"))
            rows, traj = run_episodes(env, ctrl, starts, round_index=i, t=t)
            for r in rows:
                ep_log.append(r)
            s = summarize(rows, round_index=i, t=t, t_scheduled=t, wall=0.0, snapshot=name)
            save_trajectories(os.path.join(out_dir, "trajectories"), f"round_{i:04d}_step_{t:010d}", traj)
            sum_log.append(s)  # completion marker: written only after episodes and trajectory
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
