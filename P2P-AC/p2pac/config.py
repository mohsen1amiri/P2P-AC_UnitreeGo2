"""Every option of every experiment, in one place.

Each option is declared once in :data:`OPTIONS` with its default value, a plain-language explanation and whether
the paper states its value. From this single table we build

* the command-line interface (``python -m p2pac.train --help``),
* the defaults used by every entry point (:data:`DEFAULTS`),
* the reference page ``docs/configuration.md`` (``python -m p2pac.config --markdown > docs/configuration.md``).

How the final value of an option is decided (later wins):

1. the default in :data:`OPTIONS`,
2. the method preset (:data:`PRESETS`, chosen with ``--method``),
3. configuration files given with ``--config`` (YAML or JSON; may be repeated),
4. flags typed on the command line.

The fully resolved configuration is saved in every run directory (``config.json``), so every number in the paper
can be traced back to the exact settings that produced it. ``python -m p2pac.train --print-config ...`` shows the
resolved configuration without running anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

from p2pac.layout import DEFAULT_OBSTACLES

METHODS = ("p2p", "base", "accel", "bc", "planner")
ALGOS = ("sac", "td3", "ddpg", "ppo")

METHOD_HELP = {
    "p2p": "P2P-AC, the method of the paper (with --algo sac, td3 or ddpg)",
    "base": "the RL algorithm alone, without planner (with --algo sac, td3, ddpg or ppo)",
    "accel": "Accelerated SAC baseline: the policy acts, the planner only provides imitation labels (--algo sac)",
    "bc": "behaviour cloning of the planner actions collected by a P2P-AC run (needs --bc-data)",
    "planner": "the planner acting alone (reference row of Table II, estimates of q and K)",
}


@dataclass(frozen=True)
class Option:
    """One configuration option.

    ``paper`` says where the value comes from: "Sec. VI-A" etc. if the paper states it, "" if the paper does not
    state it (these values must be reported in the paper or its supplement).
    """

    name: str
    default: Any
    group: str
    help: str
    type: Any = None          # float | int | str | "bool"; None = str
    nargs: Any = None
    choices: Optional[tuple] = None
    paper: str = ""
    symbol: str = ""


_O = Option
OPTIONS: list[Option] = [
    # ---------------------------------------------------------------------------------------------- run
    _O("experiment", "default", "run", "Experiment group; the first directory level under --output-root (e.g. main, ablation).", str),
    _O("variant", "", "run", "Optional tag appended to the run name, e.g. 'budget_5' or 'single_buffer'. "
       "Use it whenever you change an option, so the run gets its own directory.", str),
    _O("seed", 0, "run", "Random seed of the run (network initialisation, exploration, training start states, simulator).", int),
    _O("output_root", "runs", "run", "Root directory of all run directories.", str),
    _O("run_dir", None, "run", "Explicit run directory (overrides the automatic <output_root>/<experiment>/<name>/seed_<k>).", str),
    _O("resume", "auto", "run", "What to do if the run directory already exists. auto: continue an unfinished run from its last "
       "checkpoint and skip a finished one; never: refuse to use an existing directory; must: fail if there is no checkpoint.",
       str, choices=("auto", "never", "must")),
    _O("allow_config_change", False, "run", "Allow resuming a run with a different configuration (the change is recorded).", "bool"),
    _O("force_new", False, "run", "Start a fresh attempt even if a finished run exists. It is written next to the old one "
       "(<name>__rerun_<time>); nothing is overwritten.", "bool"),
    _O("total_steps", 5_000_000, "run", "Training budget in environment steps.", int, paper="Fig. 2 (x-axis)"),
    _O("learner_device", "auto", "run", "PyTorch device of the learner: auto, cpu or cuda:0. "
       "(The Isaac Sim device is set separately with Isaac Lab's --device flag.)", str),
    _O("notes", "", "run", "Free text stored in the run's manifest.", str),

    # ------------------------------------------------------------------------------------------- backend
    _O("backend", "isaac", "simulator", "Simulator. isaac: Unitree Go2 in NVIDIA Isaac Lab (all paper experiments). "
       "kinematic: a simple numpy model of the robot, only for testing the code on a laptop.", str,
       choices=("isaac", "kinematic"), paper="Sec. VI-A"),
    _O("num_envs", 1, "simulator", "Number of parallel environments (the Isaac backend supports 1).", int),
    _O("locomotion_checkpoint", None, "simulator", "Frozen locomotion policy (RSL-RL checkpoint). "
       "Default: assets/locomotion_policy/Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt.", str),
    _O("isaac_episode_length_s", None, "simulator", "Isaac Lab's internal episode length in seconds. Default: longer than "
       "T_max, so Isaac never ends an episode by itself (episodes are ended by the navigation task).", float),
    _O("locomotion_obs_noise", True, "simulator", "Keep the observation noise of the frozen locomotion policy "
       "(Isaac Lab's default for this task).", "bool"),
    _O("spawn_height", 0.5, "simulator", "Height (m) at which the robot is placed at the start of an episode.", float),
    _O("contact_threshold", 1.0, "simulator", "Contact force (N) on the trunk above which the episode ends as a crash.", float),
    _O("video", False, "simulator", "Record videos of training episodes (Isaac only; slows training down).", "bool"),
    _O("video_every", 500_000, "simulator", "Record a video every this many environment steps.", int),
    _O("video_length", 2000, "simulator", "Length of each video in environment steps.", int),
    _O("kin_tau", 0.1, "simulator", "Kinematic test model only: time constant (s) of the velocity response.", float),
    _O("kin_vel_noise", 0.01, "simulator", "Kinematic test model only: standard deviation (m/s) of the velocity noise.", float),
    _O("kin_yaw_noise", 0.0, "simulator", "Kinematic test model only: standard deviation (rad/s) of the heading noise.", float),

    # ---------------------------------------------------------------------------------------------- task
    _O("obs_mode", "paper", "task", "State given to the policy. paper: [x, y, x_goal, y_goal] (Eq. 18). extended: "
       "additionally cos/sin of the heading and the body velocity (not used in the paper).", str,
       choices=("paper", "extended"), paper="Eq. (18)"),
    _O("vx_max", 0.7, "task", "Action bound (m/s) of the forward velocity command v_x.", float, paper="Sec. VI-A"),
    _O("vy_max", 0.7, "task", "Action bound (m/s) of the lateral velocity command v_y.", float, paper="Sec. VI-A"),
    _O("yaw_hold_gain", 0.0, "task", "Gain of an optional heading controller (0 = off: yaw-rate command 0, as in the paper).", float),
    _O("goal", [0.0, 2.8], "task", "Goal position x y (m).", float, nargs=2, paper="Sec. VI-A"),
    _O("goal_tolerance", 0.3, "task", "The episode ends in the goal set G when the robot is closer than this (m) to the goal.",
       float, paper="Sec. VI-A"),
    _O("arena", [-2.0, 2.0, -2.0, 3.5], "task", "Arena bounds x_min x_max y_min y_max (m).", float, nargs=4, paper="Sec. VI-A"),
    _O("wall_margin", 0.35, "task", "The robot centre must stay this far (m) from the arena bounds; closer counts as a failure.", float),
    _O("obstacles", [v for o in DEFAULT_OBSTACLES for v in o], "task", "Cylindrical obstacles as a flat list "
       "x1 y1 r1 x2 y2 r2 ... (m).", float, nargs="+", paper="Sec. VI-A"),
    _O("robot_radius", 0.3, "task", "Planar radius (m) of the robot trunk used for collision checking.", float),
    _O("leg_reach_margin", 0.02, "task", "Extra collision margin (m) for the legs.", float),
    _O("fall_height", 0.1, "task", "The episode ends as a fall when the trunk is lower than this (m).", float, paper="Sec. VI-A"),
    _O("max_episode_steps", 8000, "task", "Maximum episode length (steps of 0.02 s); the episode is truncated after it.",
       int, paper="Sec. VI-A", symbol="T_max"),
    _O("spawn_y_max", 0.0, "task", "Start positions are drawn with y <= this value (the lower part of the arena).", float),
    _O("spawn_clearance", 0.5, "task", "Start positions keep at least this extra distance (m) from every obstacle.", float),
    _O("spawn_yaw_deg", 90.0, "task", "Initial heading of the robot (degrees; 90 = facing +y).", float),
    _O("step_cost", 1.0, "task", "Cost paid at every step.", float, paper="Sec. VI-A", symbol="w_s"),
    _O("action_cost", 0.02, "task", "Weight of the control-effort cost ||u||^2.", float, paper="Sec. VI-A", symbol="w_u"),
    _O("goal_reward", 100.0, "task", "Reward for reaching the goal.", float, paper="Sec. VI-A", symbol="r_G"),
    _O("failure_cost", 200.0, "task", "Cost of a failure (collision, leaving the arena, fall).", float, paper="Sec. VI-A", symbol="r_F"),

    # ------------------------------------------------------------------------------------------- planner
    _O("planner", "reap", "planner", "Planner used by p2p, accel and planner runs (none = no planner).", str,
       choices=("reap", "none"), paper="Sec. IV"),
    _O("planner_horizon", 15, "planner", "Prediction horizon (steps).", int, paper="Sec. VI-A", symbol="N"),
    _O("planner_dt", 0.02, "planner", "Sampling time (s) of the planner's model z+ = z + dt u.", float, paper="Sec. VI-A"),
    _O("planner_vmax", 0.5, "planner", "Bound (m/s) on each component of the planner's velocity.", float, paper="Sec. VI-A"),
    _O("planner_q", 1.0, "planner", "State weight of the MPC cost.", float),
    _O("planner_r", 0.5, "planner", "Input weight of the MPC cost.", float),
    _O("planner_terminal_cost", "dare", "planner", "Terminal weight: dare = solution of the discrete Riccati equation; "
       "q = same as the state weight.", str, choices=("dare", "q")),
    _O("reap_beta", 100.0, "planner", "Barrier parameter beta of the modified barrier function (Eq. 13).", float,
       paper="Sec. VI-A", symbol="beta"),
    _O("reap_omega", 200.0, "planner", "Parameter omega of the modified barrier function (Eq. 13); must be larger than beta.",
       float, symbol="omega"),
    _O("reap_step", 1.0, "planner", "Step size of the primal update (with backtracking, see docs/planner_validation.md).", float),
    _O("reap_dual_step", 0.05, "planner", "Step size of the dual (multiplier) update.", float),
    _O("reap_iterations", 20, "planner", "Computation budget: iterations of the primal-dual flow per planner query.", int,
       symbol="t_cpu"),
    _O("reap_time_budget_ms", None, "planner", "Optional extra wall-clock cap (ms) per query. Off by default because it makes "
       "runs machine dependent.", float),
    _O("reap_init", "cold", "planner", "Initial point of each query. cold: u = 0 (Assumption 1). warm: the shifted previous "
       "solution when it is feasible.", str, choices=("cold", "warm")),
    _O("reap_lambda_init", 0.01, "planner", "Initial value of every dual variable.", float),
    _O("reap_armijo", True, "planner", "Require a sufficient decrease of the barrier function in the primal step.", "bool"),
    _O("reap_dual_update", "gauss_seidel", "planner", "Dual update at the new primal iterate (gauss_seidel) or at the old one "
       "(jacobi = plain forward Euler).", str, choices=("gauss_seidel", "jacobi")),
    _O("planner_obstacle_margin", 0.05, "planner", "Extra distance (m) the planner keeps from obstacles beyond the collision radius.", float),
    _O("planner_wall_margin", 0.05, "planner", "Extra distance (m) the planner keeps from the walls beyond --wall-margin.", float),
    _O("planner_walls", True, "planner", "Include the arena walls in the planner's constraints.", "bool", paper="Sec. VI-A"),
    _O("planner_heading_deg", None, "planner", "Heading (degrees) used to convert the planner's world-frame velocity into the "
       "body-frame command. Default: --spawn-yaw-deg.", float),

    # ------------------------------------------------------------------------------------------- P2P-AC
    _O("t_p", 100_000, "P2P-AC", "Planner phase length: for t < T_p the planner chooses the executed actions.", int,
       paper="Sec. VI-B", symbol="T_p"),
    _O("t_d", 50_000, "P2P-AC", "Transition length: the imitation weight decays linearly to 0 between T_p and T_r = T_p + T_d.",
       int, symbol="T_d"),
    _O("kappa0", 10.0, "P2P-AC", "Initial imitation weight (constant until T_p).", float, paper="Sec. VI-B", symbol="kappa_0"),
    _O("execute_planner", True, "P2P-AC", "The planner chooses the executed actions for t < T_p "
       "(--no-execute-planner = ablation 'without planner phase').", "bool"),
    _O("imitation_space", "logit", "P2P-AC", "Imitation loss. logit: Eq. (19), in the pre-tanh space (paper). action: on the "
       "deterministic action (ablation). action_sample: on the sampled action (Accelerated SAC).", str,
       choices=("logit", "action", "action_sample"), paper="Sec. V-C"),
    _O("logit_clip_eps", 1e-4, "P2P-AC", "Clipping epsilon before artanh in the logit target (Eq. 18).", float, symbol="epsilon_c"),
    _O("label_noise_std", 0.0, "P2P-AC", "Gaussian noise (m/s) added to the imitation targets (used by Accelerated SAC).", float),
    _O("buffer", "dual", "P2P-AC", "Replay buffer. dual: protected planner buffer D_P plus FIFO buffer D_O, each minibatch half "
       "from each (paper). single: one FIFO buffer.", str, choices=("dual", "single"), paper="Sec. V-B"),
    _O("planner_buffer_capacity", 1_000_000, "P2P-AC", "Capacity of the protected planner buffer D_P (must be >= T_p).", int,
       paper="Sec. VI-B", symbol="C_P"),
    _O("planner_batch_fraction", 0.5, "P2P-AC", "Fraction of every minibatch drawn from D_P.", float, paper="Sec. V-B"),

    # ------------------------------------------------------------------------------------- actor-critic
    _O("learning_rate", 3e-4, "learner", "Adam learning rate of all networks.", float, paper="Sec. VI-B"),
    _O("buffer_size", 1_000_000, "learner", "Total replay capacity C (for the dual buffer: D_O holds C - C_P).", int,
       paper="Sec. VI-B (P2P-AC: 2e6)", symbol="C"),
    _O("learning_starts", 100, "learner", "Number of steps collected before the first gradient update.", int),
    _O("batch_size", 256, "learner", "Minibatch size.", int, symbol="n_b"),
    _O("tau", 0.005, "learner", "Polyak coefficient of the target networks.", float),
    _O("gamma", 0.99, "learner", "Discount factor.", float, symbol="gamma"),
    _O("train_freq", 1, "learner", "Environment steps between two training phases.", int),
    _O("gradient_steps", 1, "learner", "Gradient updates per training phase.", int),
    _O("net_arch", [256, 256], "learner", "Hidden layer sizes of the actor and the critic.", int, nargs="+", paper="Sec. VI-B"),
    _O("activation", "relu", "learner", "Activation function of the hidden layers.", str, choices=("relu", "tanh", "elu"),
       paper="Sec. VI-B"),
    _O("adam_eps", None, "learner", "Adam epsilon. Default: the Stable-Baselines3 default of each algorithm "
       "(1e-8 for SAC/TD3/DDPG, 1e-5 for PPO).", float),
    _O("ent_coef", "auto", "learner", "SAC entropy coefficient: 'auto' (learned), 'auto_0.1' (learned, initial value 0.1) or a number.", str),
    _O("target_entropy", None, "learner", "SAC target entropy for ent_coef=auto. Default: -dim(action) = -2.", float),
    _O("exploration_noise", 0.1, "learner", "TD3/DDPG: standard deviation of the Gaussian exploration noise, in normalised "
       "action units ([-1, 1]).", float),
    _O("td3_policy_delay", 2, "learner", "TD3: critic updates per actor update.", int),
    _O("td3_target_noise", 0.2, "learner", "TD3: standard deviation of the target policy smoothing noise.", float),
    _O("td3_target_noise_clip", 0.5, "learner", "TD3: clipping of the target policy smoothing noise.", float),
    _O("ppo_n_steps", 2048, "learner", "PPO: steps per rollout.", int),
    _O("ppo_batch_size", 64, "learner", "PPO: minibatch size.", int),
    _O("ppo_n_epochs", 10, "learner", "PPO: epochs per rollout.", int),
    _O("ppo_gae_lambda", 0.95, "learner", "PPO: GAE lambda.", float),
    _O("ppo_clip_range", 0.2, "learner", "PPO: clipping range.", float),
    _O("ppo_ent_coef", 0.0, "learner", "PPO: entropy bonus.", float),
    _O("ppo_vf_coef", 0.5, "learner", "PPO: value-loss weight.", float),
    _O("ppo_max_grad_norm", 0.5, "learner", "PPO: gradient-norm clipping.", float),

    # --------------------------------------------------------------------------------- behaviour cloning
    _O("bc_data", None, "behaviour cloning", "Run directory of a finished P2P-AC run (its artifacts/planner_buffer.npz is used) "
       "or the path of a planner_buffer.npz file.", str),
    _O("bc_epochs", 50, "behaviour cloning", "Training epochs.", int),
    _O("bc_batch_size", 256, "behaviour cloning", "Minibatch size.", int),
    _O("bc_lr", 3e-4, "behaviour cloning", "Adam learning rate.", float),
    _O("bc_val_fraction", 0.1, "behaviour cloning", "Fraction of the data held out to select the best epoch.", float),
    _O("bc_loss", "action", "behaviour cloning", "Regression target: action (MSE on actions) or logit (Eq. 19).", str,
       choices=("action", "logit")),

    # -------------------------------------------------------------------------------------- evaluation
    _O("eval_every", 100_000, "evaluation", "Evaluate the policy every this many environment steps.", int),
    _O("eval_episodes", 20, "evaluation", "Episodes per periodic evaluation (always the same start positions).", int),
    _O("eval_seed", 1_000_003, "evaluation", "Seed of the evaluation start positions (identical for all methods and seeds). "
       "The final evaluation uses eval_seed + 1.", int),
    _O("eval_at_start", True, "evaluation", "Also evaluate the untrained policy at t = 0.", "bool"),
    _O("eval_final_episodes", 100, "evaluation", "Episodes of the final evaluation of the final policy (Table II).", int),
    _O("save_eval_trajectories", True, "evaluation", "Save the xy trajectories of evaluation episodes (for figures).", "bool"),

    # ----------------------------------------------------------------------------- logging/checkpoints
    _O("log_steps", "all", "logging", "all: log every environment step to logs/steps/*.npz; none: do not.", str,
       choices=("all", "none")),
    _O("step_log_chunk", 50_000, "logging", "Environment steps per step-log file.", int),
    _O("learner_log_every", 1000, "logging", "Write learner statistics every this many gradient updates.", int),
    _O("checkpoint_every", 250_000, "logging", "Save a resumable checkpoint every this many environment steps.", int),
    _O("keep_checkpoints", 2, "logging", "Number of periodic checkpoints kept (checkpoints at T_p, T_r and the end are always kept).", int),
    _O("save_policy_snapshots", True, "logging", "Save the policy at every evaluation (re-evaluation without retraining).", "bool"),
    _O("save_planner_buffer", True, "logging", "Export D_P at t = T_p to artifacts/planner_buffer.npz (needed by BC).", "bool"),
]

DEFAULTS: dict[str, Any] = {o.name: o.default for o in OPTIONS}
OPTION_BY_NAME: dict[str, Option] = {o.name: o for o in OPTIONS}

# Values each --method sets on top of the defaults (and below --config files and command-line flags).
PRESETS: dict[str, dict[str, Any]] = {
    # P2P-AC as described in the paper (Sec. V, VI-B): C = 2e6, C_P = 1e6.
    "p2p": {"planner": "reap", "execute_planner": True, "buffer": "dual", "imitation_space": "logit",
            "buffer_size": 2_000_000, "planner_buffer_capacity": 1_000_000},
    # The RL algorithm alone.
    "base": {"planner": "none", "buffer": "single", "kappa0": 0.0, "execute_planner": False,
             "buffer_size": 1_000_000, "t_p": 0, "t_d": 0},
    # Accelerated SAC (Sec. VI-B): the policy acts, the planner provides labels, imitation on sampled actions with
    # label noise 0.01 m/s, one FIFO buffer, weight constant for 1e5 steps then linear decay over 5e4 steps.
    "accel": {"planner": "reap", "execute_planner": False, "buffer": "single", "imitation_space": "action_sample",
              "label_noise_std": 0.01, "t_p": 100_000, "t_d": 50_000, "kappa0": 10.0, "buffer_size": 1_000_000},
    # Behaviour cloning on D_P of a P2P-AC run (no RL, no planner).
    "bc": {"planner": "none"},
    # The planner acting alone.
    "planner": {"planner": "reap"},
}

# Keys that may differ between the first invocation and a resume without --allow-config-change.
RESUME_VOLATILE_KEYS = {
    "resume", "allow_config_change", "force_new", "learner_device", "notes", "run_dir", "output_root", "total_steps",
    "locomotion_checkpoint", "video", "config", "app_launcher",
}
# Keys excluded from the configuration hash (they do not change the experiment).
HASH_EXCLUDED_KEYS = RESUME_VOLATILE_KEYS | {"seed", "experiment", "variant"}

GROUP_TITLES = {
    "run": "Run management", "simulator": "Simulator", "task": "Task (Sec. VI-A)", "planner": "Planner (Sec. IV, VI-A)",
    "P2P-AC": "P2P-AC (Sec. V)", "learner": "RL algorithm", "behaviour cloning": "Behaviour cloning baseline",
    "evaluation": "Evaluation (Sec. VI-C)", "logging": "Logging and checkpoints",
}


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _help(o: Option) -> str:
    extra = []
    if o.symbol:
        extra.append(f"symbol {o.symbol}")
    extra.append(f"paper: {o.paper}" if o.paper else "not stated in the paper")
    return f"{o.help} [{'; '.join(extra)}]"


def build_parser(description: str = "P2P-AC experiment runner") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("what to run")
    g.add_argument("--method", choices=METHODS, default=None,
                   help="Training scheme: " + "; ".join(f"{k} = {v}" for k, v in METHOD_HELP.items()))
    g.add_argument("--algo", choices=ALGOS, default=None, help="RL algorithm (required for p2p, base and accel)")
    g.add_argument("--config", action="append", default=None,
                   help="YAML/JSON file with option values (repeatable; command-line flags override it)")
    g.add_argument("--print-config", dest="print_config", action="store_true",
                   help="Print the fully resolved configuration and exit")
    groups: dict[str, argparse._ArgumentGroup] = {}
    for o in OPTIONS:
        grp = groups.get(o.group) or p.add_argument_group(GROUP_TITLES[o.group])
        groups[o.group] = grp
        # default=None everywhere: "not given on the command line" (the real default comes from DEFAULTS/PRESETS)
        if o.type == "bool":
            grp.add_argument(_flag(o.name), dest=o.name, action=argparse.BooleanOptionalAction, default=None,
                             help=_help(o) + f" (default: {o.default})")
        else:
            kw: dict[str, Any] = {"dest": o.name, "default": None, "help": _help(o) + f" (default: {o.default})"}
            if o.type is not None:
                kw["type"] = o.type
            if o.nargs is not None:
                kw["nargs"] = o.nargs
            if o.choices is not None:
                kw["choices"] = o.choices
            grp.add_argument(_flag(o.name), **kw)
    return p


def load_config_file(path: str) -> dict:
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            from p2pac.yaml_compat import safe_load

            data = safe_load(f) or {}
        else:
            data = json.load(f)
    unknown = set(data) - set(DEFAULTS) - {"method", "algo"}
    if unknown:
        raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
    return data


def resolve(ns: argparse.Namespace) -> argparse.Namespace:
    """Apply defaults, preset, config files and CLI flags; validate; return a complete namespace."""
    own_keys = set(DEFAULTS) | {"method", "algo", "config", "print_config"}
    raw = vars(ns)
    # Options added by Isaac Lab's AppLauncher (headless, device, enable_cameras, ...) are kept separately.
    app_launcher = {k: v for k, v in raw.items() if k not in own_keys}
    cli = {k: v for k, v in raw.items() if v is not None and k in own_keys}
    files: dict[str, Any] = {}
    for path in cli.get("config") or []:
        files.update(load_config_file(path))
    method = cli.get("method", files.get("method"))
    algo = cli.get("algo", files.get("algo"))
    if method is None:
        raise SystemExit("--method is required (or set it in a --config file)")
    values = dict(DEFAULTS)
    values.update(PRESETS.get(method, {}))
    values.update(files)
    values.update(cli)
    values["method"] = method
    values["algo"] = algo if algo is not None else ("sac" if method in ("p2p", "base", "accel") else None)
    values.setdefault("config", cli.get("config"))
    values.pop("print_config", None)
    values["app_launcher"] = app_launcher
    out = argparse.Namespace(**values)
    validate(out)
    return out


def validate(c: argparse.Namespace) -> None:
    """Reject inconsistent settings (fail fast, before any simulator is launched)."""
    errors = []
    if c.method in ("p2p", "accel", "planner") and c.planner == "none":
        errors.append(f"method={c.method} requires a planner")
    if c.method in ("p2p", "base", "accel") and c.algo is None:
        errors.append("--algo is required")
    if c.algo == "ppo" and c.method != "base":
        errors.append("PPO is only supported as a base method (--method base --algo ppo)")
    if c.method == "accel" and c.algo != "sac":
        errors.append("Accelerated SAC requires --algo sac")
    if c.method in ("p2p", "accel") and c.buffer == "dual":
        if c.planner_buffer_capacity < c.t_p * c.num_envs:
            errors.append(f"C_P={c.planner_buffer_capacity} must be >= T_p*num_envs={c.t_p * c.num_envs}")
        if c.planner_buffer_capacity >= c.buffer_size:
            errors.append("buffer_size (C) must exceed planner_buffer_capacity (C_P)")
    if c.reap_omega <= c.reap_beta:
        errors.append("reap_omega must be larger than reap_beta")
    if c.backend == "isaac" and c.num_envs != 1:
        errors.append("the Isaac Go2 backend currently supports num_envs=1")
    if c.method == "bc" and not c.bc_data:
        errors.append("method=bc requires --bc-data")
    if c.eval_every <= 0 or c.eval_episodes <= 0:
        errors.append("eval_every and eval_episodes must be positive")
    if len(c.obstacles) % 3 != 0:
        errors.append("obstacles must be triplets x y r")
    if errors:
        raise SystemExit("invalid configuration:\n  - " + "\n  - ".join(errors))


def config_dict(c: argparse.Namespace) -> dict:
    d = dict(vars(c))
    return {k: d[k] for k in sorted(d)}


def config_hash(c: argparse.Namespace) -> str:
    d = {k: v for k, v in config_dict(c).items() if k not in HASH_EXCLUDED_KEYS}
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]


def run_name(c: argparse.Namespace) -> str:
    name = c.method if c.algo is None else f"{c.method}-{c.algo}"
    return f"{name}__{c.variant}" if c.variant else name


def default_run_dir(c: argparse.Namespace) -> str:
    if c.run_dir:
        return os.path.abspath(c.run_dir)
    return os.path.abspath(os.path.join(c.output_root, c.experiment, run_name(c), f"seed_{c.seed}"))


def diff_configs(old: dict, new: dict, ignore: Optional[set] = None) -> dict:
    ignore = ignore or set()
    keys = (set(old) | set(new)) - ignore
    return {k: (old.get(k), new.get(k)) for k in sorted(keys) if json.dumps(old.get(k), default=str) != json.dumps(new.get(k), default=str)}


# ----------------------------------------------------------------------------------------------------
# Builders used by the entry points
# ----------------------------------------------------------------------------------------------------
def task_config(c: argparse.Namespace):
    from p2pac.task import TaskConfig, parse_obstacles

    return TaskConfig(
        goal_x=c.goal[0], goal_y=c.goal[1], goal_tolerance=c.goal_tolerance,
        x_min=c.arena[0], x_max=c.arena[1], y_min=c.arena[2], y_max=c.arena[3], wall_margin=c.wall_margin,
        obstacles=parse_obstacles(c.obstacles), robot_radius=c.robot_radius, leg_reach_margin=c.leg_reach_margin,
        fall_height=c.fall_height, max_episode_steps=c.max_episode_steps, dt=0.02,
        spawn_y_max=c.spawn_y_max, spawn_clearance=c.spawn_clearance, spawn_yaw_deg=c.spawn_yaw_deg,
        step_cost=c.step_cost, action_cost=c.action_cost, goal_reward=c.goal_reward, failure_cost=c.failure_cost,
    )


def reap_config(c: argparse.Namespace):
    from p2pac.planners.reap import ReapConfig

    return ReapConfig(
        horizon=c.planner_horizon, dt=c.planner_dt, v_max=c.planner_vmax, q_weight=c.planner_q, r_weight=c.planner_r,
        terminal_cost=c.planner_terminal_cost, beta=c.reap_beta, omega=c.reap_omega, step_size=c.reap_step,
        dual_step_size=c.reap_dual_step, iterations=c.reap_iterations, time_budget_ms=c.reap_time_budget_ms,
        init=c.reap_init, obstacle_margin=c.planner_obstacle_margin, wall_margin_extra=c.planner_wall_margin,
        use_walls=c.planner_walls, lambda_init=c.reap_lambda_init, armijo=c.reap_armijo, dual_update=c.reap_dual_update,
    )


def parse_ent_coef(value):
    if isinstance(value, (int, float)):
        return float(value)
    v = str(value)
    if v.startswith("auto"):
        return v
    return float(v)


# ----------------------------------------------------------------------------------------------------
# Reference documentation
# ----------------------------------------------------------------------------------------------------
def options_markdown() -> str:
    """The option reference ``docs/configuration.md`` (generated, never edited by hand)."""
    def fmt(v: Any) -> str:
        if v is None:
            return "–"
        if isinstance(v, list):
            return " ".join(f"{x:g}" if isinstance(x, float) else str(x) for x in v)
        if isinstance(v, bool):
            return "on" if v else "off"
        if isinstance(v, float):
            return f"{v:g}"
        if isinstance(v, int):
            return f"{v:,}".replace(",", " ")
        return f"`{v}`" if v else "''"

    lines = [
        "# Configuration reference",
        "",
        "<!-- Generated by `python -m p2pac.config --markdown > docs/configuration.md`. Do not edit by hand. -->",
        "",
        "Every option can be set in three ways (later wins): the method preset (`--method`), a YAML file "
        "(`--config file.yaml`, keys written with underscores, e.g. `t_p: 100000`), or a command-line flag "
        "(dashes, e.g. `--t-p 100000`). Yes/no options are switched with `--flag` / `--no-flag`.",
        "",
        "The column **Paper** says where the paper states the value. An empty cell means the paper does not state it; "
        "such values should be reported in the paper or its supplement.",
        "",
        "## Methods and their presets",
        "",
        "| `--method` | What it runs | Values it sets |",
        "|---|---|---|",
    ]
    for m in METHODS:
        preset = ", ".join(f"`{k}={fmt(v).strip('`')}`" for k, v in PRESETS.get(m, {}).items()) or "–"
        lines.append(f"| `{m}` | {METHOD_HELP[m]} | {preset} |")
    lines.append("")
    for group, title in GROUP_TITLES.items():
        lines += [f"## {title}", "", "| Flag | Default | Paper | Meaning |", "|---|---|---|---|"]
        for o in OPTIONS:
            if o.group != group:
                continue
            flag = f"`{_flag(o.name)}`" + (" / `" + "--no-" + o.name.replace("_", "-") + "`" if o.type == "bool" else "")
            choices = f" Choices: {', '.join('`' + c + '`' for c in o.choices)}." if o.choices else ""
            sym = f" Symbol: {o.symbol}." if o.symbol else ""
            lines.append(f"| {flag} | {fmt(o.default)} | {o.paper} | {o.help}{choices}{sym} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    if "--markdown" in sys.argv:
        print(options_markdown())
    else:
        build_parser().print_help()
