"""Single entry point for every experiment of the paper.

Examples::

    # P2P-SAC (paper method), seed 0, Isaac Lab (inside the Isaac Sim container)
    python -m p2pac.train --method p2p --algo sac --seed 0 --experiment main --headless

    # SAC without planner, PPO, Accelerated SAC
    python -m p2pac.train --method base --algo sac --seed 0 --experiment main --headless
    python -m p2pac.train --method base --algo ppo --seed 0 --experiment main --headless
    python -m p2pac.train --method accel --algo sac --seed 0 --experiment main --headless

    # planner acting alone (Table II reference row, q and K of Assumption 2) at a given budget
    python -m p2pac.train --method planner --reap-iterations 20 --seed 0 --experiment planner_reference --headless

    # behaviour cloning on D_P of the P2P-SAC run with the same seed
    python -m p2pac.train --method bc --bc-data runs/main/p2p-sac/seed_0 --seed 0 --experiment main --headless

    # fast pipeline check without Isaac Sim (NOT a paper experiment)
    python -m p2pac.train --method p2p --algo sac --backend kinematic --total-steps 20000 --experiment smoke

Exit codes: 0 = finished (or already finished), 1 = error, 3 = preempted after writing a checkpoint
(the sbatch wrapper requeues the job and the next start resumes automatically).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import sys
import traceback
from typing import Optional

from p2pac.config import (RESUME_VOLATILE_KEYS, build_parser, config_dict, config_hash, default_run_dir,
                          diff_configs, reap_config, resolve, task_config)
from p2pac.runtime.rundir import (EXIT_PREEMPTED, STATUS_COMPLETED, STATUS_FAILED, STATUS_PREEMPTED, STATUS_RUNNING,
                                  RunDir, new_attempt_dir, read_json, utcnow, write_json_atomic)

ISAAC_AVAILABLE = importlib.util.find_spec("isaaclab") is not None


# ====================================================================================================
# run-directory policy
# ====================================================================================================
def prepare_run_dir(cfg: argparse.Namespace) -> tuple[Optional[RunDir], Optional[str]]:
    """Decide between fresh start, resume, skip (already finished) or a new attempt. Never overwrites."""
    root = default_run_dir(cfg)
    run = RunDir(root)
    if run.exists():
        if run.is_completed() and not cfg.force_new:
            print(f"[p2pac] run already completed, nothing to do: {root}")
            return None, None
        if run.is_completed() and cfg.force_new:
            run = RunDir(new_attempt_dir(root))
        elif cfg.resume == "never":
            raise SystemExit(f"run directory exists and --resume never was given: {root}")
        else:
            ckpt = run.latest_checkpoint()
            if ckpt is not None:
                old = read_json(run.config_path) or {}
                new = config_dict(cfg)
                diff = diff_configs(old, new, ignore=RESUME_VOLATILE_KEYS | {"config_hash"})
                if diff and not cfg.allow_config_change:
                    raise SystemExit(f"configuration differs from the existing run (use --allow-config-change):\n{json.dumps(diff, indent=2, default=str)}")
                if diff:
                    run.event("config_changed_on_resume", diff=diff)
                return run, ckpt
            if cfg.resume == "must":
                raise SystemExit(f"--resume must: no checkpoint in {root}")
            archived = run.archive_attempt()
            print(f"[p2pac] previous attempt without checkpoint archived to {archived}")
    elif cfg.resume == "must":
        raise SystemExit(f"--resume must: run directory does not exist: {root}")
    run.create()
    return run, None


def write_run_files(run: RunDir, cfg: argparse.Namespace, resumed: bool) -> None:
    from p2pac.runtime.metadata import collect_manifest, git_diff, repo_root

    cfgd = config_dict(cfg)
    cfgd["config_hash"] = config_hash(cfg)
    if not resumed:
        write_json_atomic(run.config_path, cfgd)
    with open(run.path("config_history.jsonl"), "a") as f:
        f.write(json.dumps({"time_utc": utcnow(), "resumed": resumed, "config": cfgd}, default=str) + "\n")
    manifest = collect_manifest(repo_root(), extra={"run_dir": run.root, "config_hash": cfgd["config_hash"]})
    name = "manifest.json" if not resumed else f"manifest_resume_{utcnow().replace(':', '')}.json"
    write_json_atomic(run.path(name), manifest)
    diff = git_diff(repo_root())
    if diff:
        with open(run.path("artifacts", f"git_diff_{utcnow().replace(':', '')}.patch"), "w") as f:
            f.write(diff)


def segment_dir(run: RunDir, sub: str) -> str:
    """A fresh sub-directory per (re)start for writers that would otherwise overwrite (SB3 csv, monitor, videos)."""
    base = run.path(sub)
    os.makedirs(base, exist_ok=True)
    k = len([d for d in os.listdir(base) if d.startswith("segment_")])
    path = os.path.join(base, f"segment_{k:02d}")
    os.makedirs(path)
    return path


def resolve_device(name: str) -> str:
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


# ====================================================================================================
# shared helpers
# ====================================================================================================
def build_planner_parts(cfg, task):
    """Planner factory (one instance per environment) and the planner-to-action map."""
    import math

    from p2pac.planners import PlannerActionMap, make_planner

    if cfg.planner == "none":
        return None, None
    rcfg = reap_config(cfg)
    heading = math.radians(cfg.planner_heading_deg if cfg.planner_heading_deg is not None else cfg.spawn_yaw_deg)
    action_map = PlannerActionMap(heading, [-cfg.vx_max, -cfg.vy_max], [cfg.vx_max, cfg.vy_max])
    return (lambda: make_planner(cfg.planner, rcfg, task.cfg)), action_map


def final_evaluation(run: RunDir, env, controller, task, cfg, t: int, name: str = "final") -> dict:
    """Evaluation with ``eval_final_episodes`` start states drawn with ``eval_seed + 1``."""
    import numpy as np

    from p2pac.runtime.evaluation import EPISODE_FIELDS, Timer, run_episodes, save_trajectories, summarize
    from p2pac.runtime.records import CSVLog

    starts = task.sample_starts(cfg.eval_final_episodes, cfg.eval_seed + 1)
    write_json_atomic(run.path("eval", f"{name}_starts.json"), {"seed": cfg.eval_seed + 1, "starts": starts.tolist()})
    with Timer() as tm:
        rows, traj = run_episodes(env, controller, starts, round_index=-1, t=t, record_trajectories=cfg.save_eval_trajectories)
    log = CSVLog(run.path("eval", f"{name}_episodes.csv"), EPISODE_FIELDS)
    for r in rows:
        log.append(r)
    log.close()
    summ = summarize(rows, round_index=-1, t=t, t_scheduled=None, wall=tm.elapsed)
    save_trajectories(run.path("eval", "trajectories"), f"{name}_step_{t:010d}", traj)
    succ_steps = [r["steps"] for r in rows if r["success"]]
    summ["success_steps_quantiles"] = {q: float(np.quantile(succ_steps, q)) for q in (0.5, 0.9, 0.95, 1.0)} if succ_steps else {}
    write_json_atomic(run.path("eval", f"{name}.json"), summ)
    run.event(f"{name}_evaluation", step=t, success_rate=summ["success_rate"], n_episodes=len(rows))
    return summ


# ====================================================================================================
# methods
# ====================================================================================================
def run_learning(cfg, run: RunDir, resume_ckpt: Optional[str]) -> int:
    import numpy as np
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from p2pac.algos import build_model, model_class
    from p2pac.envs import make_env
    from p2pac.runtime.callbacks import RunCallback
    from p2pac.runtime.checkpoint import CheckpointManager, checkpoint_step, load_replay_buffer, load_trainer_state
    from p2pac.runtime.evaluation import PolicyController
    from p2pac.runtime.records import StepLog, truncate_table
    from p2pac.runtime.seeding import get_rng_states, seed_everything, set_rng_states
    from p2pac.task import NavigationTask

    task = NavigationTask(task_config(cfg))
    seed_everything(cfg.seed)
    render_mode = "rgb_array" if cfg.video else None
    env = make_env(cfg, task, seed=cfg.seed, render_mode=render_mode)
    wrapped = env
    if cfg.video:
        from gymnasium.wrappers import RecordVideo

        wrapped = RecordVideo(env, video_folder=segment_dir(run, "videos"), disable_logger=True,
                              step_trigger=lambda s: s % cfg.video_every == 0, video_length=cfg.video_length)
    sb3_dir = segment_dir(run, os.path.join("logs", "sb3"))
    monitor = Monitor(wrapped, filename=os.path.join(sb3_dir, "monitor.csv"))
    vec = DummyVecEnv([lambda: monitor])
    seed_everything(cfg.seed)  # Isaac Lab re-seeds the global generators while building the scene
    write_json_atomic(run.path("artifacts", "environment.json"), {"task": task.cfg.to_dict(), "env": env.describe()})

    factory, action_map = build_planner_parts(cfg, task)
    if factory is not None:
        pl = factory()
        write_json_atomic(run.path("artifacts", "planner.json"),
                          pl.describe() if hasattr(pl, "describe") else {"planner": cfg.planner})

    starts_path = run.path("eval", "starts.json")
    starts = task.sample_starts(cfg.eval_episodes, cfg.eval_seed)
    if os.path.isfile(starts_path):
        old = np.asarray(read_json(starts_path)["starts"])
        if old.shape != starts.shape or not np.allclose(old, starts):
            raise RuntimeError("evaluation start states differ from the ones stored in the run directory")
    else:
        write_json_atomic(starts_path, {"seed": cfg.eval_seed, "starts": starts.tolist()})

    device = resolve_device(cfg.learner_device)
    trainer_state = None
    if resume_ckpt is not None:
        t_ckpt = checkpoint_step(resume_ckpt)
        model = model_class(cfg.algo).load(os.path.join(resume_ckpt, "model.zip"), env=vec, device=device)
        rb = load_replay_buffer(resume_ckpt, model.device)
        if rb is not None:
            model.replay_buffer = rb
        trainer_state = load_trainer_state(resume_ckpt)
        vec._reset_seeds()                      # SB3's load() re-seeded the env; continue its stream instead
        set_rng_states(trainer_state["rng"])
        env.np_random.bit_generator.state = trainer_state["env_np_random"]
        if trainer_state.get("backend_rng") is not None and hasattr(env.backend, "set_rng_state"):
            env.backend.set_rng_state(trainer_state["backend_rng"])
        # Move every record written after the checkpoint aside (never deleted).
        StepLog(run.path("logs", "steps"), cfg.step_log_chunk).truncate_after(t_ckpt)
        truncate_table(run.path("logs", "episodes.csv"), "t_end", t_ckpt)
        truncate_table(run.path("logs", "learner.jsonl"), "timesteps", t_ckpt)
        truncate_table(run.path("eval", "summary.csv"), "t", t_ckpt)
        truncate_table(run.path("eval", "episodes.csv"), "t", t_ckpt)
        run.event("resumed", step=t_ckpt, checkpoint=os.path.relpath(resume_ckpt, run.root))
    else:
        model = build_model(cfg, vec, device, seed=cfg.seed)
    if hasattr(model, "attach_planner"):
        model.attach_planner(factory if model.p2p.uses_planner else None, action_map)
    model.set_logger(configure(sb3_dir, ["csv", "tensorboard"]))

    callback_holder = {}

    def trainer_state_fn() -> dict:
        return {
            "rng": get_rng_states(),
            "env_np_random": env.np_random.bit_generator.state,
            "backend_rng": env.backend.get_rng_state() if hasattr(env.backend, "get_rng_state") else None,
            "callback": callback_holder["cb"].scheduler_state(),
            "num_timesteps": int(model.num_timesteps),
        }

    callback = RunCallback(run, cfg, starts, CheckpointManager(run, cfg.keep_checkpoints), trainer_state_fn,
                           state=(trainer_state or {}).get("callback"), is_on_policy=cfg.algo == "ppo")
    callback_holder["cb"] = callback

    def on_signal(signum, _frame):
        print(f"[p2pac] received signal {signum}: checkpointing and stopping", flush=True)
        callback.request_stop()

    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, on_signal)

    run.update_status(state=STATUS_RUNNING, started_utc=utcnow(), pid=os.getpid(), total_steps=cfg.total_steps,
                      host=os.uname().nodename, slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    remaining = cfg.total_steps - int(model.num_timesteps)
    try:
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=callback, log_interval=1,
                        reset_num_timesteps=resume_ckpt is None, progress_bar=False)
        t = int(model.num_timesteps)
        if callback.preempted:
            run.update_status(state=STATUS_PREEMPTED, step=t)
            return EXIT_PREEMPTED
        callback.save_checkpoint(t, milestone=True, tag="final")
        model.save(run.path("final_model.zip"))
        if cfg.eval_mode == "snapshot":
            # A separate Isaac Sim process evaluates the snapshots; do not touch the training environment.
            if callback.state.get("last_eval_t") != t:
                callback.save_evaluation_snapshot(t, scheduled=t)
            run.event("evaluation_pending", step=t)
            run.update_status(state=STATUS_COMPLETED, step=t, finished_utc=utcnow())
        else:
            # Legacy in-process periodic and final evaluations.
            if callback.state.get("last_eval_t") != t:
                callback.evaluate(t, scheduled=t)
            summ = final_evaluation(run, env, PolicyController(model), task, cfg, t)
            run.update_status(state=STATUS_COMPLETED, step=t, finished_utc=utcnow(), final_success_rate=summ["success_rate"])
        return 0
    finally:
        callback.close()
        vec.close()


def run_planner_reference(cfg, run: RunDir) -> int:
    """The planner acting alone on the final evaluation start set (and the periodic one)."""
    from p2pac.envs import make_env
    from p2pac.runtime.evaluation import PlannerController
    from p2pac.runtime.seeding import seed_everything
    from p2pac.task import NavigationTask

    task = NavigationTask(task_config(cfg))
    seed_everything(cfg.seed)
    env = make_env(cfg, task, seed=cfg.seed)
    seed_everything(cfg.seed)
    factory, action_map = build_planner_parts(cfg, task)
    planner = factory()
    write_json_atomic(run.path("artifacts", "planner.json"), planner.describe() if hasattr(planner, "describe") else {"planner": cfg.planner})
    write_json_atomic(run.path("artifacts", "environment.json"), {"task": task.cfg.to_dict(), "env": env.describe()})
    run.update_status(state=STATUS_RUNNING, started_utc=utcnow())
    summ = final_evaluation(run, env, PlannerController(planner, action_map), task, cfg, t=0)
    run.update_status(state=STATUS_COMPLETED, finished_utc=utcnow(), final_success_rate=summ["success_rate"])
    env.close()
    return 0


def run_behaviour_cloning(cfg, run: RunDir) -> int:
    """BC on the planner-phase data of a P2P-AC run, then the same final evaluation as the learners."""
    import numpy as np

    from p2pac.algos.bc import BCConfig, load_planner_dataset, train_bc
    from p2pac.envs import make_env
    from p2pac.runtime.evaluation import PolicyController
    from p2pac.runtime.metadata import package_versions  # noqa: F401  (manifest already written)
    from p2pac.runtime.seeding import seed_everything
    from p2pac.task import NavigationTask

    data = cfg.bc_data
    if os.path.isdir(data):
        data = os.path.join(data, "artifacts", "planner_buffer.npz")
    if not os.path.isfile(data):
        raise FileNotFoundError(f"BC data not found: {data} (run the P2P-AC experiment first)")
    obs, act = load_planner_dataset(data)
    run.event("bc_data", n_samples=int(obs.shape[0]), source=os.path.abspath(data))
    seed_everything(cfg.seed)
    bc_cfg = BCConfig(net_arch=tuple(cfg.net_arch), activation=cfg.activation, learning_rate=cfg.bc_lr,
                      batch_size=cfg.bc_batch_size, epochs=cfg.bc_epochs, val_fraction=cfg.bc_val_fraction,
                      loss=cfg.bc_loss, logit_clip_eps=cfg.logit_clip_eps, seed=cfg.seed)
    low, high = np.array([-cfg.vx_max, -cfg.vy_max]), np.array([cfg.vx_max, cfg.vy_max])
    policy = train_bc(obs, act, low, high, bc_cfg, run.path("policies"), log=lambda m: print(m, flush=True))
    task = NavigationTask(task_config(cfg))
    env = make_env(cfg, task, seed=cfg.seed)
    run.update_status(state=STATUS_RUNNING, started_utc=utcnow())
    summ = final_evaluation(run, env, PolicyController(policy), task, cfg, t=0)
    run.update_status(state=STATUS_COMPLETED, finished_utc=utcnow(), final_success_rate=summ["success_rate"])
    env.close()
    return 0


# ====================================================================================================
def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    if ISAAC_AVAILABLE:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    ns = parser.parse_args(argv)
    cfg = resolve(ns)
    if ns.print_config:
        d = config_dict(cfg)
        d["config_hash"] = config_hash(cfg)
        d["run_dir"] = default_run_dir(cfg)
        print(json.dumps(d, indent=2, default=str))
        return 0

    run, resume_ckpt = prepare_run_dir(cfg)
    if run is None:
        return 0
    app = None
    if cfg.backend == "isaac":
        if not ISAAC_AVAILABLE:
            raise SystemExit("backend=isaac requires Isaac Lab (run inside the Isaac Sim container)")
        from isaaclab.app import AppLauncher

        from p2pac.isaac_assets import add_to_launcher_args

        launcher_args = argparse.Namespace(**cfg.app_launcher)
        if cfg.video:
            launcher_args.enable_cameras = True
        add_to_launcher_args(launcher_args)      # robot model and textures from assets/isaac_sim (P2PAC_ISAAC_ASSETS)
        app = AppLauncher(launcher_args).app
    try:
        write_run_files(run, cfg, resumed=resume_ckpt is not None)
        if cfg.method == "planner":
            return run_planner_reference(cfg, run)
        if cfg.method == "bc":
            return run_behaviour_cloning(cfg, run)
        return run_learning(cfg, run, resume_ckpt)
    except BaseException as exc:  # record every failure in the run directory
        if not isinstance(exc, SystemExit) or exc.code not in (0, None):
            run.update_status(state=STATUS_FAILED, error=repr(exc), traceback=traceback.format_exc(), failed_utc=utcnow())
            run.event("failed", error=repr(exc))
        raise
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    sys.exit(main())
