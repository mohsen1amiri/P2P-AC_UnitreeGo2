# What a run saves

Every run writes everything it does into one folder, its **run directory**:

```
<output_root>/<experiment>/<run name>/seed_<k>/
```

- `<output_root>` is `--output-root` (default `runs`).
- `<experiment>` is `--experiment` (for example `main`, `ablation`).
- `<run name>` is `<method>-<algo>`, with `__<variant>` added when `--variant` is set (for example `p2p-sac`, `base-ppo`, `p2p-sac__single_buffer`, `planner__budget_20`).

Everything needed to produce the paper's numbers, tables and figures is in this folder. Nothing has to be recomputed by rerunning training. The analysis scripts read only these files.

**Contents**

1. [Overview](#1-overview)
2. [Status, configuration and provenance](#2-status-configuration-and-provenance)
3. [Training logs](#3-training-logs)
4. [Evaluation results](#4-evaluation-results)
5. [Saved policies, checkpoints and artifacts](#5-saved-policies-checkpoints-and-artifacts)
6. [Codes used in the logs](#6-codes-used-in-the-logs)
7. [Reading the logs in Python](#7-reading-the-logs-in-python)

---

## 1. Overview

| Path | Written | Content |
|---|---|---|
| `status.json` | continuously | Is the run running, finished, failed or preempted? Current step. |
| `config.json` | at the start | The complete configuration of the run (every option, after presets, files and flags). |
| `config_history.jsonl` | at every (re)start | The configuration of each start or resume. |
| `manifest.json` | at the start | Provenance: git commit, package versions, machine, GPU, SLURM job, command line. |
| `manifest_resume_<time>.json` | at every resume | The same for each resume. |
| `events.jsonl` | continuously | Timeline of important events (phase changes, evaluations, checkpoints, resumes). |
| `logs/steps/steps_<from>_<to>.npz` | continuously | **Every environment step** of training. |
| `logs/episodes.csv` | continuously | **Every training episode**. |
| `logs/learner.jsonl` | every 1000 updates | Losses, Q-values, entropy coefficient, buffer statistics. |
| `logs/sb3/segment_<k>/` | continuously | Stable-Baselines3's own logs (`progress.csv`, `monitor.csv`, TensorBoard). One segment per (re)start. |
| `eval/starts.json` | at the start | The fixed start positions of the periodic evaluations. |
| `eval/summary.csv` | every evaluation | One row per evaluation: success rate etc. **This is the learning curve.** |
| `eval/episodes.csv` | every evaluation | One row per evaluation episode. |
| `eval/trajectories/*.npz` | every evaluation | The xy path of every evaluation episode. |
| `eval/final.json`, `eval/final_episodes.csv`, `eval/final_starts.json` | at the end | Final evaluation of the final policy (**Table II**). |
| `eval/reeval_<tag>/` | when you run `p2pac.evaluate` | Extra evaluations of saved policies. |
| `policies/policy_step_<t>.zip` | every evaluation | The policy evaluated at step t. |
| `final_model.zip` | at the end | The final policy. |
| `checkpoints/step_<t>/` | every `--checkpoint-every` steps, at T_p, at T_r, at the end | Everything needed to resume training exactly. |
| `artifacts/` | at the start and at T_p | Environment and planner description, the planner buffer D_P, uncommitted code changes. |
| `attempts/attempt_<time>/` | only if a start crashed before its first checkpoint | The files of that attempt (kept, never deleted). |
| `videos/segment_<k>/` | only with `--video` | Videos of training episodes. |

---

## 2. Status, configuration and provenance

### `status.json`

| Key | Meaning |
|---|---|
| `state` | `created` → `running` → `completed`, or `failed` (an error; see `events.jsonl` and the SLURM log), or `preempted` (stopped at the time limit after writing a checkpoint; it continues when restarted). |
| `step` / `total_steps` | Environment steps done / training budget. |
| `last_checkpoint_step` | Step of the newest checkpoint (where a restart continues). |
| `final_success_rate` | Success rate of the final evaluation (set when the run is completed). |
| `started_utc`, `updated_utc`, `finished_utc` | Times (UTC). |
| `host`, `pid`, `slurm_job_id` | Where the run is executing. |

### `config.json`

All options of [configuration.md](configuration.md) with their final values, plus `method`, `algo`, `config_hash` and `app_launcher` (Isaac Lab's own options such as `headless` and `device`).

`config_hash` is a short fingerprint of everything that changes the experiment. Two runs with the same hash differ only in seed, experiment name or variant name.

### `manifest.json`

| Key | Content |
|---|---|
| `git` | Commit, branch, whether there were uncommitted changes (the changes are saved in `artifacts/git_diff_<time>.patch`). |
| `packages` | Versions of numpy, torch, stable-baselines3, gymnasium, Isaac Lab, rsl-rl, … |
| `simulator` | Isaac Lab version and path, Isaac Sim image path and size, Isaac Sim version. |
| `hardware` | Host name, platform, number of CPUs, GPU names, CUDA version. |
| `slurm`, `env` | SLURM job/array ids and relevant environment variables. |
| `argv`, `cwd` | The exact command line and working directory. |

### `events.jsonl`

One JSON object per line, each with `time_utc`, `event` and details:

| `event` | When | Details |
|---|---|---|
| `evaluation` | after every periodic evaluation | step, success rate, … |
| `planner_phase_end` | at t = T_p | number of planner steps, size of D_P, goal transitions in D_P, file of the exported D_P |
| `planner_removed` | at t = T_r | number of planner queries after T_r (must be 0) |
| `checkpoint` | after every checkpoint | step, whether it is a milestone (T_p, T_r, final) |
| `training_episode_aborted_for_evaluation` | when an evaluation cannot wait for the end of a training episode (PPO, whose evaluations run between rollouts, and the last evaluation at the end of training) | the unfinished training episode is logged with `aborted = 1` |
| `final_evaluation` | at the end | final success rate |
| `resumed` | after a restart | the step it continued from |
| `config_changed_on_resume` | resume with `--allow-config-change` | which options changed |
| `preempted` | the run received the stop signal | step of the checkpoint written (the unfinished episode is logged with `aborted = 1`) |
| `failed` | an error stopped the run | the error message |
| `warning` | something unexpected that did not stop the run | message |
| `bc_data` | behavior cloning runs | the planner data used |
| `reevaluation` | after `python -m p2pac.evaluate` | tag, number of policies evaluated |

---

## 3. Training logs

### `logs/steps/steps_<from>_<to>.npz` — every environment step

Each file holds `--step-log-chunk` steps (default 50 000). All arrays have one entry per step.

| Array | Shape | Meaning |
|---|---|---|
| `t` | (n,) | Global step index (0, 1, 2, …). |
| `episode`, `ep_step` | (n,) | Training-episode index and step within the episode. |
| `phase` | (n,) | 0 = planner phase (t < T_p), 1 = transition (T_p ≤ t < T_r), 2 = RL only (t ≥ T_r). |
| `kappa` | (n,) | Imitation weight κ_t (Eq. 17). |
| `executed_by_planner` | (n,) | 1 if the executed action was the planner's (t < T_p in P2P-AC). |
| `iota` | (n,) | ι_t: 1 if a planner action was available for this step (t < T_r). |
| `obs` | (n, 4) | State s_t = [x, y, x_goal, y_goal] (8 columns with `--obs-mode extended`). |
| `executed_action` | (n, 2) | Command actually sent to the robot [v_x, v_y] (m/s). |
| `behavior_action` | (n, 2) | Action proposed by the RL policy (before the planner possibly replaced it) (m/s). |
| `planner_action` | (n, 2) | Planner action u†_t in body frame (m/s); 0 when the planner was not queried. |
| `reward` | (n,) | Reward r_t (Eq. 1). |
| `terminated`, `truncated` | (n,) | Episode ended in G or F / was cut at T_max. |
| `outcome` | (n,) | Outcome code of s_{t+1} (see [section 6](#6-codes-used-in-the-logs)). |
| `clearance` | (n,) | Distance (m) to the nearest obstacle or wall margin (negative = collision). |
| `dist_to_goal` | (n,) | Distance (m) to the goal. |
| `yaw` | (n,) | Robot heading (rad). |
| `vel_body` | (n, 2) | Measured body-frame velocity (m/s). |
| `action_clipped` | (n,) | 1 if the action had to be clipped to the action box. |
| `internal_reset` | (n,) | 1 if the simulator reset or teleported the robot by itself. **Must always be 0.** |
| `planner_iterations` | (n,) | Iterations the planner used for this query. |
| `planner_ms` | (n,) | Wall-clock time of the planner query (ms). |
| `planner_max_constraint` | (n,) | Largest constraint value of the planner's solution; ≤ 0 means feasible (Definition 1). |
| `planner_outside_domain` | (n,) | 1 if the state was outside S_P (Assumption 1 did not hold for this query). |
| `planner_action_clipped` | (n,) | 1 if the planner action had to be clipped to the action box. |

Logging every step costs disk space (about 50–60 MB per million steps). Switch it off with `--log-steps none`; the theory checks then have less data.

### `logs/episodes.csv` — every training episode

| Column | Meaning |
|---|---|
| `episode` | Training-episode index. |
| `t_start`, `t_end` | Global step at the start and end of the episode. |
| `phase_start`, `phase_end`, `kappa_end` | Phase at start/end and κ at the end. |
| `planner_executed_steps` | Steps in which the planner chose the action. |
| `outcome`, `outcome_name` | How the episode ended (code and name). |
| `success`, `crash`, `timeout` | 1/0 flags: reached the goal / failed / hit T_max. |
| `steps` | Episode length (steps of 0.02 s). |
| `return`, `discounted_return` | Sum of rewards and discounted sum (with `--gamma`). |
| `path_length` | Length (m) of the path travelled. |
| `path_ratio` | Path length divided by the straight-line distance from the start to the goal. |
| `duration_s` | Episode duration (s). |
| `mean_speed` | Path length / duration (m/s). |
| `mean_command_speed` | Mean norm of the commanded velocity (m/s). |
| `min_clearance` | Smallest clearance during the episode (m). |
| `final_distance` | Distance to the goal at the end (m). |
| `start_x`, `start_y` | Start position. |
| `max_abs_action` | Largest command component (m/s). |
| `max_abs_yaw_drift` | Largest heading change from the start (rad). |
| `internal_resets` | Simulator-internal resets (must be 0). |
| `aborted` | 1 if the episode was cut before its end: to run an evaluation, or because the run was stopped (time limit). Together, the rows cover every training step exactly once. |

### `logs/learner.jsonl` — learner statistics

One JSON object every `--learner-log-every` gradient updates (averages over that window):

| Key | Meaning |
|---|---|
| `timesteps`, `n_updates`, `n_updates_in_window` | Environment steps, total updates, updates in this window. |
| `phase`, `kappa` | Phase and imitation weight. |
| `critic_loss`, `actor_loss_ac`, `actor_loss_total` | Critic loss; actor loss of the RL algorithm; actor loss including the imitation term. |
| `imitation_loss`, `imitation_loss_on_labelled`, `logit_abs_mean` | Imitation term E[ι ℓ], ℓ on rows with a planner label, mean ‖ξ_θ(s)‖ (only while κ > 0). |
| `ent_coef`, `ent_coef_loss`, `entropy` | SAC entropy coefficient and its loss. |
| `q_mean`, `q_mean_planner_rows`, `q_mean_online_rows` | Mean Q over the minibatch, over rows from D_P, over rows from D_O. |
| `batch_frac_planner_buffer` | Fraction of the minibatch from D_P (0.5 with the dual buffer). |
| `batch_frac_iota` | Fraction of minibatch rows with a planner label. |
| `batch_goal_rows`, `batch_goal_rows_from_planner_buffer`, `batch_has_goal_from_planner_buffer` | Goal transitions in the minibatch (for Corollary 2). |
| `n_planner_buffer`, `n_goal_planner_buffer`, `n_online_buffer` | Size of D_P, goal transitions in D_P, size of D_O. |
| `n_planner_queries`, `n_planner_queries_after_removal`, `n_planner_executed_steps` | Planner queries so far, queries after T_r (**must be 0**), steps executed by the planner. |

### `logs/sb3/segment_<k>/`

Stable-Baselines3's standard output, unchanged: `progress.csv` (its usual `train/*`, `rollout/*`, `time/*` columns plus `p2p/kappa`), `monitor.csv` (reward and length of each training episode) and TensorBoard event files. A new segment folder is started at every resume, so nothing is overwritten.

---

## 4. Evaluation results

The policy is evaluated **alone** (the planner never acts), with its deterministic or mean action, every `--eval-every` steps on the same `--eval-episodes` start positions. The start positions are drawn once from the training start distribution with `--eval-seed` and are identical for every method and seed. Evaluation episodes are never stored in the replay buffer and are not counted as training steps.

### `eval/summary.csv` — one row per evaluation (the learning curve)

| Column | Meaning |
|---|---|
| `round` | Evaluation number (0 = untrained policy at t = 0). |
| `t`, `t_scheduled` | Step at which the evaluation actually ran (the next episode end) / was scheduled. |
| `n_episodes` | Episodes evaluated. |
| `success_rate`, `crash_rate`, `timeout_rate` | Fractions of episodes reaching the goal / failing / hitting T_max. |
| `return_mean`, `discounted_return_mean`, `steps_mean` | Means over all episodes. |
| `path_ratio_mean_success`, `duration_s_mean_success`, `mean_speed_mean_success` | Means over **successful** episodes only. |
| `min_clearance_mean` | Mean of the per-episode minimum clearance. |
| `internal_resets` | Simulator-internal resets (must be 0). |
| `wall_time_s` | Time the evaluation took. |
| `policy_snapshot` | File of the evaluated policy (`policies/policy_step_<t>.zip`). |

### `eval/episodes.csv` and `eval/final_episodes.csv` — one row per evaluation episode

The same columns as `logs/episodes.csv`, plus `round`, `t`, `eval_episode` and, for planner runs, `planner_ms_mean`, `planner_max_constraint` and `planner_outside_domain_frac`.

### `eval/final.json` — final evaluation (Table II)

The **final** policy (never a "best" checkpoint) on `--eval-final-episodes` start positions drawn with `eval_seed + 1`, so they differ from the periodic ones. The keys are the columns of `summary.csv` plus `success_steps_quantiles`. The start positions are in `eval/final_starts.json`.

### `eval/trajectories/*.npz`

| Array | Meaning |
|---|---|
| `xy` | All positions of all episodes, concatenated (m). |
| `actions` | All commands, concatenated (m/s). |
| `offsets` | Where each episode starts in `xy`; episode i is `xy[offsets[i]:offsets[i+1]]`. |
| `outcome` | Outcome code of each episode. |
| `starts` | Start position of each episode. |

---

## 5. Saved policies, checkpoints and artifacts

| Path | Content | Used by |
|---|---|---|
| `policies/policy_step_<t>.zip`, `final_model.zip` | Stable-Baselines3 model files. Load with `P2PSAC.load(path)` (or `SAC.load`, `TD3.load`, …). | `p2pac.evaluate`, hardware deployment |
| `policies/bc_policy.pt`, `policies/bc_training.json` (BC runs) | The behavior-cloning network and its training and validation loss per epoch. | `p2pac.evaluate`, deployment |
| `checkpoints/step_<t>/model.zip` | Model with optimizer states. | resume |
| `checkpoints/step_<t>/replay_buffer.pkl` | Replay buffer(s), including D_P. | resume |
| `checkpoints/step_<t>/trainer_state.pkl` | Random-number-generator states, step counters, evaluation schedule, environment state. | resume |
| `checkpoints/step_<t>/meta.json` | Step, whether it is a milestone, tag (`T_p`, `T_r`, `final`). | resume |
| `checkpoints/LATEST` | Name of the newest checkpoint. | resume |
| `artifacts/environment.json` | The task (arena, obstacles, reward, …) and the simulator (Isaac settings, locomotion checkpoint and its SHA-256, command slice). | provenance |
| `artifacts/planner.json` | The planner's exact parameters and its tightened constraints. | provenance, `planner_gap` |
| `artifacts/planner_buffer.npz` | D_P exported at t = T_p: `observations`, `next_observations`, `actions`, `rewards`, `dones`, `timeouts`, `planner_actions`, `planner_mask`, `executed_by_planner`, `goal`, `step_index` (actions in normalized units [−1, 1]). | BC baseline, theory checks |
| `artifacts/git_diff_<time>.patch` | Uncommitted changes of the code at the start of the run (if any). | provenance |

Periodic checkpoints are deleted when newer ones exist, keeping the last `--keep-checkpoints`. The milestone checkpoints (T_p, T_r, final) are always kept.

---

## 6. Codes used in the logs

**Outcome codes** (`outcome` columns):

| Code | Name | Meaning |
|---|---|---|
| 0 | `running` | Episode continues. |
| 1 | `goal` | Reached the goal set G (closer than `--goal-tolerance`). |
| 2 | `collision_obstacle` | Closer to an obstacle than its radius + robot radius + leg margin. |
| 3 | `out_of_bounds` | Closer to a wall than `--wall-margin`. |
| 4 | `fall` | Trunk lower than `--fall-height`. |
| 5 | `base_contact` | The trunk touched something (contact sensor, Isaac only). |
| 6 | `timeout` | Reached T_max (truncation, not a failure). |

Codes 2–5 are failures (`crash = 1`).

**Phase codes** (`phase`): 0 = planner phase, 1 = transition, 2 = RL only. Runs without a planner are always in phase 2.

---

## 7. Reading the logs in Python

```python
import json, glob, numpy as np, pandas as pd

run = "runs/main/p2p-sac/seed_0"
curve = pd.read_csv(f"{run}/eval/summary.csv")            # learning curve: curve.t, curve.success_rate
final = json.load(open(f"{run}/eval/final.json"))         # Table II values of this seed
episodes = pd.read_csv(f"{run}/logs/episodes.csv")       # every training episode
learner = pd.read_json(f"{run}/logs/learner.jsonl", lines=True)

# every training step, as one table (from p2pac.runtime.records)
from p2pac.runtime.records import load_steps
steps = load_steps(run)                                   # dict of arrays (see section 3)
print(steps["t"].shape, steps["kappa"][steps["t"] >= 150_000].max())   # kappa after T_r is 0
```
