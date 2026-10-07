# The paper and the code

This page explains how the code implements each part of the paper and lists everything the manuscript should state
or correct so that paper and code agree. It has five parts:

1. [Where each part of the paper is implemented](#1-where-each-part-of-the-paper-is-implemented)
2. [Implementation details the paper does not describe](#2-implementation-details-the-paper-does-not-describe)
3. [Values the manuscript leaves open (`\pending`)](#3-values-the-manuscript-leaves-open-pending)
4. [Manuscript text to correct](#4-manuscript-text-to-correct)
5. [How the theory is checked in the logs](#5-how-the-theory-is-checked-in-the-logs)

---

## 1. Where each part of the paper is implemented

| Paper | What it says | Code |
|---|---|---|
| Sec. III-A, Eq. (1) | MDP, goal set G, failure set F, reward r = −w_s − w_u‖u‖² + r_G 1[G] − r_F 1[F] | `p2pac/task.py` (`NavigationTask.classify`, `NavigationTask.reward`) |
| Sec. III-B | Off-policy actor-critic (DDPG, TD3, SAC) | Stable-Baselines3 2.7.1, `p2pac/algos/p2p.py` |
| Sec. III-C, Eq. (7)–(12) | MPC problem, tightened polyhedral constraints | `p2pac/planners/constraints.py` |
| Sec. IV, Eq. (13)–(15) | REAP: modified barrier, primal-dual flow, stop at the budget | `p2pac/planners/reap.py` |
| Sec. V-A, Eq. (16) | Behavior substitution: the planner acts for t < T_p | `PlannerToPolicyMixin._sample_action` in `p2pac/algos/p2p.py` |
| Sec. V-B | Protected planner buffer D_P + FIFO buffer D_O, half of each minibatch from each | `PlannerReplayBuffer` in `p2pac/algos/buffers.py` |
| Sec. V-C, Eq. (18)–(19) | Imitation in logit space | `PlannerToPolicyMixin._imitation_loss` |
| Sec. V-D, Eq. (17) | κ_t: constant until T_p, linear decay to 0 at T_r = T_p + T_d | `P2PConfig.kappa` in `p2pac/algos/schedule.py` |
| Algorithm 1 | The full loop for P2P-DDPG, P2P-TD3, P2P-SAC | `P2PDDPG`, `P2PTD3`, `P2PSAC` in `p2pac/algos/p2p.py` |
| Sec. VI-A | Go2 in Isaac Lab, frozen locomotion policy, arena, obstacles | `p2pac/envs/isaac_go2.py`, `assets/locomotion_policy/` |
| Sec. VI-B | Baselines: the algorithms alone, Accelerated SAC, PPO, behavior cloning, the planner alone | presets in `p2pac/config.py`; `p2pac/algos/bc.py` |
| Sec. VI-C | Evaluation protocol and metrics | `p2pac/runtime/evaluation.py`, `p2pac/runtime/callbacks.py` |
| Sec. VI-D–G | Main results, budget, removal, ablations | `configs/experiments/*.yaml`, `p2pac/analysis/` |
| Sec. VI-H | Hardware deployment | `p2pac/deploy/` |

When the planner is switched off, P2P-SAC, P2P-TD3 and P2P-DDPG are **bit-for-bit identical** to Stable-Baselines3's SAC, TD3 and DDPG (`tests/test_algorithm_equivalence.py`). Training after T_r is therefore exactly the underlying algorithm, as Proposition 3 states.

---

## 2. Implementation details the paper does not describe

These are choices made in the code that a reader cannot infer from the manuscript. Each should appear in the paper or its supplement.

### 2.1 Planner (Sec. IV)

- **Discretization of the flow.** Eq. (14) is a continuous-time flow. The code integrates it with a primal gradient step, a dual step evaluated at the new primal point, and step halving. The halving continues until all constraints are strictly satisfied and the barrier function has decreased sufficiently. Without these safeguards a fixed-step Euler discretization oscillates instead of converging. Details and evidence: [planner_validation.md](planner_validation.md).
- **Computation budget.** t_cpu is implemented as a number of iterations per query (`--reap-iterations`, default 20). This makes runs deterministic and independent of the machine. The time each query takes is logged, and the x-axis of Fig. 3 (`tcpu` in `budget_sweep.csv`) is the measured mean time per query of the planner-only runs. State the CPU it was measured on.
- **Starting point.** Every query starts from u = 0 and λ = 0.01 (`--reap-init cold`). This satisfies Assumption 1 whenever the robot's state is in S_P.
- **States outside S_P.** The real robot can enter the safety margin of the planner, because its motion differs from the planner's model. Then u = 0 is not strictly feasible. The planner relaxes the violated constraints just enough to start, returns an action, and flags the query (`planner_outside_domain`). The frequency is reported by the theory checks. The start region is slightly larger than S_P (starts with −1.65 ≤ y < −1.60 m are outside it, about 3% of starts), so Assumption 1 does not hold for every start.
- **No terminal set Ω.** The paper's problem (11) contains (ẑ_N, d) ∈ Ω. The code uses the terminal weight from the Riccati equation (`--planner-terminal-cost dare`) but no terminal set. The constraints are re-linearized around the current state at every step, and the goal is usually much farther than N = 15 steps (0.15 m at 0.5 m/s). A terminal set around the goal would therefore make the problem infeasible almost everywhere. Propositions 1–2 and Corollary 1 do not need Ω.
- **Safety margins.** The planner keeps a distance of r + 0.30 (robot radius) + 0.02 (legs) + 0.05 (planner margin) = 0.60 m from obstacle centers. The task counts a collision below 0.55 m. From the walls it keeps 0.35 + 0.05 = 0.40 m; the task counts leaving the arena below 0.35 m. All values are saved in `artifacts/planner.json` of every run.
- **Speed bound.** The planner bounds each velocity component by 0.5 m/s (a box), so its speed can reach 0.71 m/s.
- **World to body frame.** The planner's world-frame velocity is mapped to the body-frame command assuming the initial heading of 90°: u_body = [u_y, −u_x] (Sec. VI-A). Heading drift of the robot is logged (`max_abs_yaw_drift`).

### 2.2 P2P-AC (Sec. V)

- **Behavior substitution from the first step.** Stable-Baselines3 takes uniformly random actions during its first `learning_starts` = 100 steps. In P2P-AC the planner action replaces those too, so Eq. (16) holds from t = 0.
- **What is stored.** Each transition stores the executed action (the planner's for t < T_p), the planner action u† and ι_t = 1 for t < T_r. All actions are stored in Stable-Baselines3's normalized units [−1, 1]. Transitions with t < T_p go to D_P, all later ones to D_O. The capacity of D_O is C − C_P.
- **Logit target.** ξ† = artanh(clip(u†/0.7, ±(1 − ε_c))) with ε_c = 10⁻⁴. ξ_θ(s) is the pre-tanh mean of the actor: SAC's Gaussian mean, or the output of TD3/DDPG's actor before its final tanh. ℓ is the mean over the two action components.
- **TD3.** The imitation term is added at TD3's delayed actor updates (every second critic update).
- **Logit-space argument in this task.** Planner actions are at most 0.5 m/s and the action bound is 0.7 m/s, so every normalized target is at most 0.71, where tanh′ ≥ 0.49. The vanishing-gradient argument for logit space is weak here; the ablation "imitation on actions" may show only a small difference.

### 2.3 Baselines (Sec. VI-B)

- **Accelerated SAC:** the policy acts and the planner only provides labels. The loss is the squared error between the policy's sampled action and the planner action plus Gaussian label noise (0.01 m/s). One FIFO buffer. Weight 10 for 10⁵ steps, then linear decay over 5×10⁴ steps.
- **Behavior cloning:** regression of the planner actions in D_P of the P2P-SAC run with the same seed. MLP 256×256 (ReLU, tanh output), Adam 3×10⁻⁴, 50 epochs, squared error on actions. The epoch with the lowest loss on 10% held-out data is kept; evaluation success is never used to select it. BC has no learning curve, so Table II has no N80 for it.
- **The planner alone:** the reference row of Table II and the estimates q̂ and K̂ of Assumption 2.
- **PPO:** Stable-Baselines3 defaults (2048 steps per rollout, 10 epochs, minibatch 64, GAE 0.95, clip 0.2), with ReLU activations as stated for all learners.

### 2.4 Simulation (Sec. VI-A)

- **Locomotion policy:** Isaac Lab's pretrained RSL-RL checkpoint for `Isaac-Velocity-Flat-Unitree-Go2-v0` (Isaac Lab 2.3.0, MD5 `50f9fe1f7ae965ab3e7dc975156f3adb`), not trained for this paper. Its observation noise stays on (Isaac Lab's default).
- **Episodes are ended only by the navigation task.** Isaac Lab's own 20-second time-out and its contact termination are disabled (they would reset the robot inside a step). Trunk contact (> 1 N) is read from the contact sensor and ends the episode as a failure. Every step is checked for simulator-internal resets (`internal_reset`, always 0).
- **Failure set F:** trunk contact, trunk height below 0.1 m, distance to an obstacle center below r + 0.30 + 0.02 m, or a distance below 0.35 m to a wall.
- **Start distribution ρ₀:** x uniform in [−1.65, 1.65], y uniform in [−1.65, 0] m, rejected if closer than 0.55 + 0.5 m to an obstacle center; heading 90°.
- **One environment per run** (`num_envs = 1`). Three runs share one GPU (an NVIDIA A40 on Alvis or a GH200 on Arrhenius). State the GPU used for the reported experiments; every run records it in `manifest.json`.
- **Determinism.** Seeds are set for Python, NumPy, PyTorch and Isaac Lab. GPU physics in Isaac Sim is not bitwise deterministic, so Isaac runs are reproducible statistically, not bit for bit. The kinematic test model is bitwise reproducible.

### 2.5 Evaluation (Sec. VI-C)

- The policy acts alone with its deterministic or mean action; the planner never acts in evaluations.
- **Periodic evaluation:** every 10⁵ steps, 20 episodes, from start positions drawn once from ρ₀ with a fixed seed. These are the same for all methods and seeds. Evaluation episodes are never stored or counted as training steps. With one environment, an evaluation runs at the end of the training episode in progress.
- **Final evaluation (Table II):** the final policy, never a "best" checkpoint, on 100 other start positions.
- **N80:** the first evaluation step at which the success rate is ≥ 80%; its resolution is the evaluation period. A seed that never reaches 80% is reported as "> budget" and counted in the table, for example "1.2 ± 0.3 (4/5)".
- **Mean ± standard deviation over seeds.** Path ratio, duration and speed are averaged over successful episodes only. The shaded band of Fig. 2 is one standard deviation, clipped to [0, 100] (`band: std` in `configs/analysis/paper.yaml`; `ci95` gives 95% confidence intervals instead).

---

## 3. Values the manuscript leaves open (`\pending`)

Current defaults of the code. They are starting values, not tuned; change them with the flags shown. The value used by each run is in its `config.json`.

| Quantity | Manuscript | Code default | Flag |
|---|---|---|---|
| Planner budget t_cpu | `\pending` | 20 iterations per query | `--reap-iterations` |
| Transition length T_d | `\pending` | 5×10⁴ steps | `--t-d` |
| Minibatch size n_b | `\pending` | 256 | `--batch-size` |
| Evaluation period / episodes | `\pending` | 10⁵ steps / 20 episodes (final: 100) | `--eval-every`, `--eval-episodes`, `--eval-final-episodes` |
| Discount γ | not stated | 0.99 | `--gamma` |
| Target-network rate τ, learning starts, updates per step | not stated | 0.005, 100, 1 | `--tau`, `--learning-starts`, `--gradient-steps` |
| SAC entropy | not stated | learned, target −2 | `--ent-coef`, `--target-entropy` |
| TD3/DDPG exploration noise | not stated | N(0, 0.1²) in normalized units (0.07 m/s) | `--exploration-noise` |
| TD3 policy delay, target noise | not stated | 2, 0.2 (clip 0.5) | `--td3-*` |
| Adam ε | not stated | library default (10⁻⁸; PPO 10⁻⁵) | `--adam-eps` |
| Baseline buffer size | not stated | 10⁶ (P2P-AC: C = 2×10⁶, C_P = 10⁶, so D_O = 10⁶) | `--buffer-size` |
| Planner weights, ω, step sizes | not stated | q = 1, r = 0.5, ω = 200, h_p = 1, h_d = 0.05, λ₀ = 0.01 | `--planner-q`, `--planner-r`, `--reap-*` |

[configuration.md](configuration.md) lists every option. Its "Paper" column shows which values the paper states.

---

## 4. Manuscript text to correct

1. **Fig. 1 (`w_1.pdf`)** shows the name PriPG-RL and elements that are not part of P2P-AC: the gate G_φ, β_t, "Privileged I_t", M_t and s_t = h_s(x_t).
2. **Running header** (`\markboth`) reads "P2P-SAC: Planner-to-Policy Soft Actor–Critic with an Anytime-Feasible MPC Planner".
3. **Before submission:** switch `\showstorytrue` to `\showstoryfalse`, resolve the 23 `{\color{gray} …}` passages, and replace the 90 `\pending` with values from `paper_results/numbers.tex` (`\res{…}`).
4. **Arena:** the text says x ∈ [−2.2, 2.0] m; the code uses [−2.0, 2.0] m.
5. **Start positions:** "the lower half of the arena" is y ≤ 0 m (Sec. 2.4 above).
6. **Failure set:** add leaving the arena (0.35 m wall margin) and give the collision radius (Sec. 2.4).
7. **Planner speed:** "bounded by 0.5 m/s" should read "each velocity component is bounded by 0.5 m/s".
8. **Problem (11):** remove Ω or state that no terminal set is used (Sec. 2.1).
9. **Budget:** say that t_cpu is an iteration count and give the measured time per query and the CPU (Sec. 2.1).
10. **Discretization of the flow:** one paragraph in Sec. IV or the supplement (Sec. 2.1).
11. **Assumption 1:** state it for states in S_P and report how often it fails in the experiments (Sec. 2.1).
12. **State:** the 4-D state of Eq. (18) has no heading or velocity, so the transition is only approximately Markov, which Proposition 4 assumes. One sentence acknowledging this (hidden heading and velocity are part of the mismatch w_t).
13. **Sec. III-D:** "often across many parallel environments"; every run uses one environment.
14. **Locomotion controller:** cite it as Isaac Lab's pretrained checkpoint (Sec. 2.4).
15. **Accelerated SAC:** say that its policy acts and the planner only provides labels.
16. **Hyperparameters:** state the values of Section 3.
17. **Hardware (Sec. VI-H):**
   - `p2pac/deploy/go2_deploy.py` commands Unitree's built-in sport-mode controller (`SportClient.Move`), not the Isaac Lab locomotion policy; say so.
   - Say how the pose reaches the policy (OptiTrack relay).
   - Check the accuracy "0.02 mm" against the camera specification.
   - Report the number of trials and the success rate with a Wilson interval (`p2pac.deploy.hardware_analysis`).
   - In the hardware figure the goal is drawn at y ≈ 2.5 m (text: 2.8 m), the start (y ≈ −2.2 m) lies outside ρ₀, and the obstacles are boxes rather than cylinders. Describe the actual hardware layout.

---

## 5. How the theory is checked in the logs

`python -m p2pac.analysis.theory --runs runs` computes, for every P2P-AC run:

| Statement | Quantity in `theory_checks.csv` | Expected |
|---|---|---|
| Definition 1(i): every planner action is feasible | `max_constraint_in_SP` | ≤ 0 |
| Assumption 1 | `assumption1_violation_frac` (queries with z ∉ S_P) | small; report it |
| Assumption 2 | `q_hat`, `K_hat_max`, `K_hat_p95` from the planner-phase episodes | — |
| Proposition 2 | `N_G` against `prop2_bound_E_NG`; `n_G_observed` against `prop2iii_bound_E_nG` | within the bounds |
| Corollary 2 | `cor2_observed_hit_prob` against `cor2_predicted_hit_prob` | close |
| Proposition 3 | `planner_queries_after_T_r`, `kappa_max_after_T_r` | 0 and 0 |
| Corollary 1 | `python -m p2pac.analysis.planner_gap` (gap against budget, exponential fit) | decreasing |

**Things to look at when the Isaac results come in:**

- `internal_reset` must be 0 everywhere. `scripts/isaac_smoke_test.py` checks it before any experiment.
- The success rate around T_r (Fig. 2, Sec. VI-F). On the kinematic test model, P2P-SAC's success rate dropped after the planner was removed and recovered later in some seeds but not others. Whether this happens in Isaac, and how strongly, is part of the answer to the question "how does the policy behave when the planner is removed?". The learner log has what is needed to understand it: κ, the entropy coefficient, and Q-values on D_P and D_O rows.
