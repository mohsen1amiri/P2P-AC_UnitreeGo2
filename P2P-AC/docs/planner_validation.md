# Validation of the planner implementation

This page explains how the planner of Sec. IV (`p2pac/planners/reap.py`) turns the paper's continuous-time flow into code, and shows the checks that it works.

All numbers below come from the **kinematic test model**: a point robot whose velocity follows the command with a 0.1 s lag, plus small noise. They validate the implementation. They are **not** results of the paper; the paper's planner numbers come from `configs/experiments/planner_reference.yaml` on Isaac Lab.

## 1. From the continuous flow to an algorithm

Eq. (14) evolves the planned inputs u and the multipliers λ in continuous time. The obvious implementation, a fixed-size Euler step for both, does not work reliably. Once an obstacle constraint becomes active the barrier becomes very steep (its curvature grows with β·λ). A fixed step then overshoots, the iterates oscillate around the solution instead of converging (the exponential convergence of Proposition 1 is lost), and some steps leave the feasible set, where the barrier is undefined.

The code therefore makes one iteration as follows:

1. **Primal step.** u ← u − h_p ∇_u B(u, λ). The step is halved until two conditions hold:
   - every constraint is strictly satisfied, which keeps the anytime-feasibility property of Proposition 2;
   - the barrier function has decreased sufficiently (Armijo test), as it does along the continuous flow.
2. **Dual step.** λ ← max(0, λ + h_d (∇_λ B(u, λ) + Ψ)), evaluated at the *new* u. This is the Arrow–Hurwicz scheme; `--reap-dual-update jacobi` evaluates it at the old u, which is the plain Euler step.

Two further choices:

- the multipliers start at λ₀ = 0.01 rather than 0, so the barrier is active from the first iteration;
- ω > β (200 > 100) keeps the modified barrier well defined near the constraint boundary.

| Property of the continuous flow | How the code keeps it | Option |
|---|---|---|
| Iterates stay feasible (Prop. 2: anytime feasibility) | step halving until all constraints are strictly satisfied | always on |
| B decreases along the primal flow | Armijo sufficient-decrease test | `--reap-armijo` |
| Stable primal-dual coupling | dual step at the new primal point | `--reap-dual-update gauss_seidel` |
| Barrier active from the start | λ₀ > 0 | `--reap-lambda-init 0.01` |
| Modified barrier well defined | ω > β | `--reap-omega 200` |

The paper (Sec. IV or the supplement) should describe this discretization; a reader who implements Eq. (14) with plain Euler steps will not reproduce the planner.

**Parameters.** β = 100 is from Sec. VI-A. ω = 200, h_p = 1, h_d = 0.05 and λ₀ = 0.01 are values for which the iteration converges on the kinematic model; they are starting values, not tuned on the Isaac task. The other parameters are N = 15, each velocity component bounded by 0.5 m/s, q = 1, r = 0.5, and the Riccati terminal weight. The default budget is 20 iterations per query.

## 2. Closed-loop test at different budgets

Command (about 10 minutes on a laptop; writes [`planner_validation.csv`](planner_validation.csv)):

```bash
python scripts/validate_planner.py --episodes 30 --budgets 1 2 5 10 20 50 100
```

The planner drives the kinematic model from 30 start positions drawn from ρ₀ (seed 0) until the goal, a failure or T_max = 8000 steps.

| Iterations per query | Reached goal | Collided | Mean steps (successful) | Largest constraint value (states in S_P) | Queries outside S_P | ms per query* |
|---|---|---|---|---|---|---|
| 1 | 0% | 100% | – | −1.1e−7 | 53% | 0.15 |
| 2 | 0% | 100% | – | −1.4e−7 | 55% | 0.22 |
| 5 | 17% | 83% | 734 | −2.3e−12 | 42% | 0.46 |
| 10 | 100% | 0% | 721 | −1.2e−15 | 0.09% | 0.86 |
| **20 (default)** | **100%** | **0%** | **457** | −6.9e−17 | 0.19% | 1.61 |
| 50 | 100% | 0% | 408 | −2.5e−15 | 0.47% | 3.71 |
| 100 | 100% | 0% | 406 | −1.3e−13 | 0.39% | 6.53 |

\* Measured on a laptop that was busy with other jobs. The paper should report the times measured on the cluster (`planner_reference` runs).

What the table shows:

- **Every action is feasible at every budget** (Definition 1(i)). For states in S_P, the largest constraint value of the returned plan is ≤ 0 up to rounding.
- **Feasible does not mean successful.** With 1–2 iterations the plan barely moves away from its starting point u = 0, so the robot crawls. The difference between the model and the real motion pushes it into the planner's safety margin (outside S_P), and it eventually collides. This is the difference between Definition 1 (feasibility for the planning model) and Assumption 2 (reaching the goal with the real system). The budget sweep of the paper should therefore include budgets where the planner reaches the goal.
- **A larger budget gives shorter paths.** Mean episode length falls from 721 steps (10 iterations) to 406 steps (100 iterations), as Corollary 1 predicts.

## 3. Distance to the optimal solution against the budget (Corollary 1)

```bash
python -m p2pac.analysis.planner_gap --run runs/main/p2p-sac/seed_0 --budgets 1 2 5 10 20 50 100 200
```

This command takes states the robot visited (from the run's step log) and computes two distances of the planner's first action:

- to the exact MPC solution (SLSQP on the same problem);
- to the converged flow (5000 iterations).

Medians on the kinematic smoke run:

| Iterations | 1 | 2 | 5 | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|
| distance to converged flow (m/s) | 0.438 | 0.377 | 0.312 | 0.265 | 0.241 | 0.152 | 0.066 | 0.024 |
| distance to exact MPC solution (m/s) | 0.443 | 0.384 | 0.332 | 0.278 | 0.241 | 0.163 | 0.081 | 0.062 |

The fit is median distance to the converged flow ≈ 0.342 · exp(−0.0141 · iterations), the exponential decay of Corollary 1. The distance to the exact solution levels off at about 0.06 m/s, because the barrier solution differs from the exact optimum by a term of order 1/β (the approximation error in Definition 1(ii)). Run the command on an Isaac P2P-SAC run for the paper.

## 4. Unit tests

`tests/test_planners.py` checks:

- feasibility for random states in S_P at budgets 0–100;
- that states outside S_P are flagged and still get an action;
- that the distance to the optimum does not grow with the budget;
- the constraint construction and the world-to-body mapping.
