# P2P-AC: Planner-to-Policy Actor–Critic for the Unitree Go2

> **P2P-AC: Planner-to-Policy Actor–Critic Reinforcement Learning via Anytime-Feasible MPC**
>
> Mohsen Amiri¹‡, Mohsen Amiri²‡, Ali Beikmohammadi¹, Sindri Magnússon¹, and Mehdi Hosseinzadeh², *Senior Member, IEEE*
>
> ¹ Department of Computer and System Science, Stockholm University, Sweden\
> ² School of Mechanical and Material Engineering, Washington State University, USA\
> ‡ Equal contribution

https://github.com/user-attachments/assets/5a33b944-228e-4857-9487-6d0ff60dda69

*Part 1: PPO, SAC and P2P-SAC (the proposed method) after 5 million training steps in NVIDIA Isaac Lab. Part 2: the P2P-SAC policy in Isaac Lab and in zero-shot deployment on a real Unitree Go2. (Video file: [docs/media/p2p-ac_go2.mov](docs/media/p2p-ac_go2.mov).)*

---

This repository contains the complete code for the paper **"P2P-AC: Planner-to-Policy Actor–Critic Reinforcement Learning via Anytime-Feasible MPC"**. With it you can train every method of the paper, reproduce every figure, table and number from the saved logs, and run the trained policy on a real Unitree Go2.

The folder is self-contained:

- the robot simulator setup (Isaac Lab, the locomotion policy, the container recipe);
- the training code and the experiment definitions;
- the cluster scripts;
- the analysis that turns logs into the paper's results.

Nothing outside this folder is needed except the Isaac Sim container image, which the HPC provides or one command downloads (Section 4). Nothing is installed on the HPC, and the setup works on both x86-64 (Alvis) and ARM (Arrhenius) GPU nodes.

---

## Contents

1. [What is in this folder](#1-what-is-in-this-folder)
2. [How a training run works](#2-how-a-training-run-works)
3. [Quick start on a laptop (no GPU, about 15 minutes)](#3-quick-start-on-a-laptop-no-gpu-about-15-minutes)
4. [Set up the GPU cluster (once)](#4-set-up-the-gpu-cluster-once)
5. [Run a single training run](#5-run-a-single-training-run)
6. [Run all experiments of the paper](#6-run-all-experiments-of-the-paper)
7. [How the SLURM jobs work](#7-how-the-slurm-jobs-work)
8. [Change settings and try your own variants](#8-change-settings-and-try-your-own-variants)
9. [From logs to the paper's figures, tables and numbers](#9-from-logs-to-the-papers-figures-tables-and-numbers)
10. [Where the results are saved](#10-where-the-results-are-saved)
11. [Evaluate saved policies again](#11-evaluate-saved-policies-again)
12. [Hardware experiments with the Unitree Go2](#12-hardware-experiments-with-the-unitree-go2)
13. [Tests](#13-tests)
14. [Troubleshooting](#14-troubleshooting)
15. [More documentation](#15-more-documentation)
16. [Publishing this folder on GitHub](#publishing-this-folder-on-github)
17. [Acknowledgements](#acknowledgements)
18. [License and citation](#license-and-citation)

---

## 1. What is in this folder

| Folder / file | What it contains |
|---|---|
| `p2pac/` | The Python package: task, simulators, planner, P2P-AC and the baselines, logging, evaluation, analysis, hardware deployment. |
| `p2pac/train.py` | **The single entry point for every training run** (every method, algorithm, seed, ablation). |
| `configs/experiments/` | One file per group of experiments of the paper (which runs, which seeds, which settings). |
| `configs/analysis/paper.yaml` | Which runs produce which figure, table and number of the paper. |
| `scripts/container.sh` | Runs commands inside the Isaac Sim container (`p2pac_run ...`). |
| `scripts/make_sweep.py` | Turns an experiment file into a SLURM job array (or a list of commands). |
| `scripts/status.py` | Shows the state of all runs. |
| `scripts/setup/` | One-time setup: download the container image (if the HPC has none), install the Python packages into the folder, verify the local simulator assets. |
| `scripts/clusters/` | Settings for Arrhenius and Alvis (copied to `scripts/cluster.env`). |
| `scripts/isaac_smoke_test.py` | Checks that the simulator works correctly before you start experiments. |
| `third_party/IsaacLab/` | NVIDIA Isaac Lab v2.3.0, unmodified (see `VENDORED.md` there). |
| `assets/` | The frozen locomotion policy of the Go2 (Isaac Lab's pretrained checkpoint) and the robot model and textures Isaac Sim loads. |
| `container/` | The Isaac Sim 5.1.0 image (downloaded, not in Git), a library it needs, and the pre-downloaded Python packages, per CPU architecture. |
| `third_party/pyyaml/` | A YAML reader, so the job tools run with a plain `python3` on a login node. |
| `requirements/` | Exact Python package versions: `isaac-overlay.txt` (cluster) and `dev.txt` (laptop). |
| `tests/` | Automatic tests (run with `pytest`). |
| `docs/` | Detailed documentation (see [Section 15](#15-more-documentation)) and the video (`docs/media/`). |

The words used in this README:

| Word | Meaning |
|---|---|
| **run** | One training of one method with one seed, for example P2P-SAC with seed 0. Each run has its own folder. |
| **experiment** | A group of runs, for example `main` (Fig. 2 and Table II) or `ablation` (Table III). |
| **method** | `p2p` (P2P-AC), `base` (the RL algorithm alone), `accel` (Accelerated SAC), `bc` (behavior cloning), `planner` (the planner alone). |
| **algo** | The RL algorithm: `sac`, `td3`, `ddpg` or `ppo`. |
| **variant** | A label for a changed setting, for example `single_buffer` or `budget_5`. |
| **step** | One control step of 0.02 s (one action of the policy). Budgets and schedules are counted in steps. |

---

## 2. How a training run works

A P2P-AC run passes through three phases (Sec. V of the paper):

```
environment step t:   0 ─────────────── T_p ─────────────────── T_r = T_p + T_d ────────────► end (5·10⁶)

who chooses actions:  the planner        the RL policy              the RL policy
planner is asked:     yes                yes (only for labels)      never
imitation weight κ:   κ0 = 10            decreases linearly to 0    0
new data goes to:     D_P (kept forever) D_O (first in, first out)  D_O
each minibatch:       half from D_P and half from D_O (all from one buffer if the other is empty)

defaults: T_p = 100 000, T_d = 50 000
```

During a run the program also does the following:

- **Evaluates the policy alone** every 100 000 steps on 20 fixed start positions, with no planner and the deterministic action. This produces the learning curves.
- **Saves a checkpoint** every 250 000 steps and at T_p, T_r and the end. A stopped run continues exactly where it stopped.
- **Logs every environment step, every episode and the learner statistics**, so every result of the paper can be computed later from the logs.
- **Ends with a final evaluation** of the final policy on 100 other start positions (Table II).

The baselines use the same program:

| Method | What happens |
|---|---|
| `--method base` | The RL algorithm alone (no planner, one buffer). |
| `--method accel` | The policy acts from the start; the planner only provides imitation labels until T_r. |
| `--method planner` | The planner drives the robot. Only the final evaluation is run. |
| `--method bc` | A network is fitted to the planner data D_P of a P2P-SAC run and then evaluated. |

---

## 3. Quick start on a laptop (no GPU, about 15 minutes)

On a laptop, the robot is replaced by a simple **kinematic test model** (`--backend kinematic`). It is meant only for checking that the code, the logging and the analysis work. **It is not used for any result in the paper.**

```bash
# 1. Python environment (Python 3.11 recommended)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/dev.txt
pip install -e . --no-deps

# 2. Automatic tests (about 2 minutes)
pytest

# 3. One short P2P-SAC run on the test model (about 1 minute)
python -m p2pac.train --method p2p --algo sac --backend kinematic \
       --total-steps 5000 --t-p 1000 --t-d 500 --eval-every 1000 --eval-episodes 3 \
       --experiment try --output-root runs_smoke

# 4. All methods and ablations, 2 seeds each, then the analysis (about 10 minutes)
python scripts/make_sweep.py configs/experiments/smoke_kinematic.yaml
bash scripts/run_local.sh sweeps/smoke_kinematic/jobs.txt
python -m p2pac.analysis.aggregate --spec configs/analysis/smoke_kinematic.yaml --runs runs_smoke --out results_smoke
python scripts/status.py --runs runs_smoke
```

After step 3, look at `runs_smoke/try/p2p-sac/seed_0/`; [Section 10](#10-where-the-results-are-saved) explains every file. After step 4, `results_smoke/` contains the same kind of tables and curves the paper uses, computed from these short runs. (The runs are far too short to learn anything; the numbers are meaningless.)

---

## 4. Set up the GPU cluster (once)

The paper's experiments run in NVIDIA Isaac Sim, which needs a Linux machine with an NVIDIA GPU. This section describes **Arrhenius** (NAISS), the cluster in use now, and **Alvis** (C3SE/NAISS), where the original experiments were run. The same steps work on other SLURM clusters with Apptainer, and on a single Linux GPU machine without SLURM.

### 4.1 Rules: portable, self-contained, nothing installed

The setup follows these rules:

- **Nothing is installed on the HPC.** No system packages, no root, no `module` installs, no `pip install` into the system or your home directory.
- **Everything lives in the repository folder.** The only exception is one container image file (`.sif`), provided by the HPC or downloaded once as a normal user.
- **Nothing is written outside the folder.** Two exceptions:
  - two empty folders in your home directory (`~/.cache/ov`, `~/.nvidia-omniverse`), used as mount points;
  - Isaac Sim's temporary caches, which go to the node-local disk on Arrhenius.
- **Runs need no internet.** The robot model and textures, the locomotion policy and Isaac Lab are in the folder. The Python packages can be pre-downloaded into the folder as well (step 4).
- **Both CPU architectures are covered.** Alvis nodes are x86-64. The GPU nodes of Arrhenius are **ARM (aarch64)**, while its login node is not. Every architecture-specific file exists for both, and `scripts/container.sh` picks the right one on each node.

| What the HPC must provide | Arrhenius | Alvis |
|---|---|---|
| GPU nodes with the NVIDIA driver | GH200 Grace Hopper (aarch64 CPU + Hopper GPU) | A40 (x86-64) |
| Apptainer (or Singularity) | yes | yes |
| SLURM | yes (`-p gpu --gpus=N`, GPU project account ending in `-gpu`) | yes (`--gpus-per-node=A40:1`) |
| `bash` and standard tools (`mkdir`, `pgrep`, `ps`, `kill`, `date`) | yes | yes |
| `python3` ≥ 3.7, no packages needed (for `make_sweep.py`, `status.py`, `isaac_assets.py`) | system Python of the login node | same |
| `git` (optional: records the code version in every run) | yes | yes |

| What this folder brings | Where | In Git? |
|---|---|---|
| Isaac Lab 2.3.0 (unmodified) | `third_party/IsaacLab/` | yes |
| Go2 locomotion policy | `assets/locomotion_policy/` | yes |
| Robot model, ground and textures Isaac Sim loads | `assets/isaac_sim/` | yes |
| YAML reader for the job tools (plain `python3` has none) | `third_party/pyyaml/` | yes |
| OpenSSL 1.1 library the image lacks, per architecture | `container/compat/<arch>/` | yes |
| Cluster settings | `scripts/clusters/arrhenius.env`, `alvis.env` | yes |
| The extra Python packages, exact versions | listed in `requirements/isaac-overlay.txt`; installed into `.pipbase/<arch>/`; pre-downloaded in `container/wheelhouse/<arch>/` | the list yes; the files no (too large) |
| Isaac Sim 5.1.0 image | the HPC's copy, or `container/isaac-sim-5.1.0-<arch>.sif` (step 3) | no (several GB) |

### 4.2 Important for Arrhenius: Isaac Sim on GH200

**NVIDIA does not officially support Isaac Sim 5.1 on the GH200**, for two independent reasons ([Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)):

- "GPUs without RT Cores (A100, H100) are not supported", and the GH200's GPU is an H100-class Hopper GPU;
- "Isaac Sim aarch64 builds are currently only supported on DGX Spark".

NVIDIA does publish the container for aarch64 (`nvcr.io/nvidia/isaac-sim:5.1.0` exists for `arm64`). Users report that **headless training without cameras** works on H100 GPUs, and only camera rendering fails ([Isaac Lab issue #4271](https://github.com/isaac-sim/IsaacLab/issues/4271)). P2P-AC needs no cameras. But no run on GH200 itself has been confirmed. Therefore:

1. **Run the acceptance test (step 5) on an Arrhenius GPU node before anything else.** If it ends with `ALL CHECKS PASSED`, the simulator works there.
2. **Videos cannot be recorded on GH200** (`--video` needs camera rendering and RT cores). The video at the top of this page was not made on Arrhenius.
3. **Keep one cluster per comparison.** Run all methods and seeds of one figure or table on the same cluster. GPU physics results differ slightly between GPU models, and every run records its machine, CPU architecture and GPU in `manifest.json`.
4. **If the acceptance test fails on GH200,** run the experiments on Alvis (A40, x86-64), where the original experiments ran, with the same folder: only `scripts/cluster.env` changes.

### 4.3 Where each step runs (Arrhenius)

The Arrhenius login node and the GPU nodes have different CPU architectures. An aarch64 container cannot run on the login node, so everything that uses the container runs on a GPU node:

| Task | Where | Command |
|---|---|---|
| copy the folder, edit `scripts/cluster.env` | login node | step 1–2 |
| download the image | login node (or a GPU node) | `bash scripts/setup/get_container.sh --arch aarch64` |
| check the simulator assets | login node | `python3 scripts/setup/isaac_assets.py check` |
| install the Python packages, acceptance test | **GPU node** (interactive job) | steps 4–5 |
| generate and submit job arrays, follow runs | login node | `python3 scripts/make_sweep.py …`, `sbatch …`, `python3 scripts/status.py` |
| training, evaluation | **GPU nodes** (SLURM jobs) | automatic |
| analysis (tables, figures) | GPU node, or a laptop after copying `runs/` | Section 9 |

On Alvis everything can also run on the login node except training (the login node is x86-64 like the GPU nodes).

### Step 1 — Put the repository on the cluster

Use the project storage, not your home directory. On Arrhenius container images must not be stored under `$HOME` (30 GiB quota), and each run writes about 1 GB.

```bash
cd /nobackup/proj/disk/<project>                 # Arrhenius   (Alvis: /mimer/NOBACKUP/groups/<group>)
git clone <address of this repository> P2P-AC   # or copy the folder from your computer (see below)
cd P2P-AC
```

**Copying the folder** (`rsync -a P2P-AC/ <user>@login.hpc.arrhenius.naiss.se:/nobackup/proj/disk/<project>/P2P-AC/`) also brings `container/wheelhouse/`, so no internet is needed in step 4. A clone from GitHub does not contain the wheelhouse.

**The path must not contain spaces** (Isaac Sim cannot handle them).

### Step 2 — Cluster settings

```bash
cp scripts/clusters/arrhenius.env scripts/cluster.env        # Alvis: scripts/clusters/alvis.env
nano scripts/cluster.env                                      # set your project account
```

On Arrhenius, check three settings:

- `P2PAC_SLURM_ACCOUNT` must be your **GPU** project, whose name ends in `-gpu`.
- `P2PAC_SLURM_TIME` must not exceed the GPU partition's limit; check it with `sinfo -p gpu -o "%P %l"`. Runs are checkpointed and requeued automatically, so a shorter limit only means more restarts.
- `P2PAC_SLURM_CPUS_PER_TASK` and `P2PAC_SLURM_MEM` are starting values for one GH200 shared by three runs.

The file is read automatically by every script, is not stored in Git, and is explained in `scripts/clusters/README.md`.

### Step 3 — The Isaac Sim 5.1.0 container image

- **Arrhenius:** download the aarch64 image into the repository (normal user, no root; needs internet and about 30 GB of temporary space; 20–60 minutes):

  ```bash
  bash scripts/setup/get_container.sh --arch aarch64     # -> container/isaac-sim-5.1.0-aarch64.sif
  rm -rf .apptainer_cache                                # temporary files of the download
  ```

  If `apptainer pull --arch arm64` is not possible on the login node, run `bash scripts/setup/get_container.sh` inside an interactive GPU job (step 4) instead; it then downloads the image for that node's architecture.
- **Alvis:** nothing to do. `scripts/clusters/alvis.env` already points to C3SE's image `/apps/containers/IsaacSim/IsaacSim-NGC-5.1.0.sif`.

`scripts/container.sh` uses `container/isaac-sim-5.1.0-<arch>.sif` for the node's architecture, unless `ISAACSIM_SIF_<arch>` in `scripts/cluster.env` points elsewhere.

### Step 4 — Install the extra Python packages into the folder (about 5 minutes)

On Arrhenius this must run on a GPU node, because the packages are installed with the container's own Python:

```bash
interactive -A <project>-gpu -p gpu --gpus=1 -c 16 --mem=60G -t 01:00:00     # Arrhenius (or: salloc …)
cd /nobackup/proj/disk/<project>/P2P-AC
bash scripts/setup/install_overlay.sh
```

On Alvis, run `bash scripts/setup/install_overlay.sh` on the login node.

The script installs the exact versions listed in `requirements/isaac-overlay.txt` into `.pipbase/<arch>/` inside the repository. If `container/wheelhouse/<arch>/` exists, it installs from there without internet; otherwise it downloads from the Python Package Index. At the end it imports every package and prints `ok` for each. On aarch64 one package is left out, `triton`: it has no aarch64 build, and this project does not use it.

To create the wheelhouse (for HPC nodes without internet), run on any computer with internet:

```bash
python3 scripts/setup/download_wheels.py --arch aarch64     # and/or --arch x86_64
```

Then copy the folder to the HPC.

### Step 5 — Acceptance test (about 10 minutes, on a GPU node)

In the same interactive GPU job:

```bash
source scripts/container.sh
python3 scripts/setup/isaac_assets.py check                 # local robot model and textures complete
p2pac_run python scripts/isaac_smoke_test.py --headless --out isaac_smoke.json
```

It checks six things and must end with `ALL CHECKS PASSED`:

- the locomotion policy file is the right one;
- the robot is never reset by the simulator itself during a long episode;
- velocity commands reach the locomotion policy;
- the robot follows them;
- collisions are detected;
- the planner drives the robot to the goal with feasible actions.

**Do not start experiments until it passes.** On Arrhenius this test also answers whether Isaac Sim runs on GH200 at all (Section 4.2). Keep `isaac_smoke.json` and the job output; they document the result.

### Step 6 (optional) — A short real training run

```bash
p2pac_run python -m p2pac.train --method p2p --algo sac --seed 0 --experiment test \
          --total-steps 20000 --t-p 5000 --t-d 2500 --eval-every 5000 --eval-episodes 3 --headless
python3 scripts/status.py --experiment test
```

It writes `runs/test/p2p-sac/seed_0/` (see [Section 10](#10-where-the-results-are-saved)). The column `time/fps` of `logs/sb3/segment_00/progress.csv` gives the speed in environment steps per second, which you need to plan the time of the full experiments (5 million steps per run).

### How `scripts/container.sh` works

`source scripts/container.sh` (in **bash**) defines three commands:

| Command | What it does |
|---|---|
| `p2pac_run python <args>` | Runs Python inside the container with everything set up: Isaac Lab, the extra packages, this repository, the local assets, the GPU. |
| `p2pac_pip <args>` | Runs the container's pip; packages are installed into `.pipbase/<arch>/`. |
| `p2pac_shell` | Opens a shell inside the container (for debugging). |

On every call it does the following:

- **Architecture.** It reads the node's CPU architecture (`uname -m`: `x86_64` or `aarch64`) and selects the matching image, package folder (`.pipbase/<arch>/`) and compatibility library (`container/compat/<arch>/`). On aarch64 it also preloads `libgomp`, as NVIDIA's instructions for Isaac Sim on ARM require.
- **Same paths everywhere.** It mounts the repository at the same path inside the container, so all paths in logs and configurations are identical inside and outside.
- **Code version.** It records the code version with the host's `git` for the run's `manifest.json`, because the image has no `git`.
- **Fresh caches.** It gives every process a fresh Isaac Sim cache folder and deletes it afterwards.

### Folders that appear while running (never in Git)

| Folder | What it is |
|---|---|
| `.pipbase/<arch>/` | The extra Python packages (step 4). Recreate with `bash scripts/setup/install_overlay.sh`. |
| `.isaac_cache/` or `/scratch/local/$USER/p2pac_isaac_cache/` (Arrhenius) | Isaac Sim's working folders, one per process: compiled shaders, generated code, logs. Deleted when the process ends (`P2PAC_KEEP_CACHE=1` keeps them). Isaac Sim regenerates everything in them. |
| `.apptainer_cache/` | Temporary files of the image download (step 3); delete it afterwards. |
| `runs/`, `sweeps/` | Your results and generated job files. |

---

## 5. Run a single training run

Every run is started with `python -m p2pac.train`; on the cluster you prefix it with `p2pac_run`. Example (P2P-SAC, seed 0):

```bash
source scripts/container.sh
p2pac_run python -m p2pac.train --method p2p --algo sac --seed 0 --experiment main --headless
```

| Part of the command | Meaning |
|---|---|
| `--method p2p --algo sac` | What to train: P2P-SAC. Other examples: `--method base --algo td3` (TD3 alone), `--method accel --algo sac`, `--method base --algo ppo`. |
| `--seed 0` | Random seed. |
| `--experiment main` | Group name. Results go to `runs/main/p2p-sac/seed_0/`. |
| `--headless` | Isaac Sim without a window (always on a cluster). |

All other settings take their default values, which are listed with explanations in [docs/configuration.md](docs/configuration.md) and printed by `python -m p2pac.train --help`. To see the exact configuration a command would use, without running anything, add `--print-config`.

**More examples:**

```bash
# the planner alone with a budget of 20 iterations per query (reference row of Table II)
p2pac_run python -m p2pac.train --method planner --reap-iterations 20 --variant budget_20 \
          --experiment planner_reference --headless

# behavior cloning on the planner data of the P2P-SAC run with seed 0 (after that run has passed T_p)
p2pac_run python -m p2pac.train --method bc --bc-data runs/main/p2p-sac/seed_0 --seed 0 --experiment main --headless

# an ablation: P2P-SAC with a single buffer instead of D_P + D_O
p2pac_run python -m p2pac.train --method p2p --algo sac --buffer single --variant single_buffer \
          --seed 0 --experiment ablation --headless
```

**Following a run while it trains.**

```bash
python3 scripts/status.py --experiment main                       # state and latest success rate of all runs
cat runs/main/p2p-sac/seed_0/status.json                          # state, current step
column -s, -t < runs/main/p2p-sac/seed_0/eval/summary.csv | less  # learning curve so far
tail -n 3 runs/main/p2p-sac/seed_0/events.jsonl                   # latest events
```

**Stopping and continuing.** Start the same command again and the run continues from its last checkpoint. You lose at most the steps since that checkpoint. This applies whether the run was stopped by you, by a crash or by the time limit.

**The safety rules for run folders:**

- **A finished run is never repeated.** Starting it again prints `run already completed, nothing to do`.
- **An unfinished run is continued**, never restarted from zero.
- **Settings cannot change silently.** If you start an existing run with different settings, the program refuses and lists the differences. Use a new `--variant` (or a new `--experiment`) to get a new folder. `--allow-config-change` forces the change and records it.
- **Nothing is ever overwritten.** `--force-new` trains a finished run again in a new folder next to the old one (`seed_0__rerun_<time>`).

---

## 6. Run all experiments of the paper

Each experiment of the paper has a file in `configs/experiments/`. A sweep runs everything in one file: all runs and all seeds.

| Order | Experiment file | Paper | Runs | Approx. cost* |
|---|---|---|---|---|
| 1 | `main.yaml` | Fig. 2, Table II (8 methods: DDPG, P2P-DDPG, TD3, P2P-TD3, SAC, P2P-SAC, Accel. SAC, PPO) | 8 × 5 seeds = 40 | 14 GPU jobs × ~4 days |
| 2 | `planner_reference.yaml` | Table II (planner row), Fig. 3 (time per query, q̂), Assumption 2 | 7 budgets × 1 seed | a few GPU hours |
| 3 | `budget.yaml` | Fig. 3 (N80 of P2P-SAC against the planner budget) | 5 budgets × 5 seeds = 25 | 9 GPU jobs × ~4 days |
| 4 | `ablation.yaml` | Table III (5 ablations of P2P-SAC) | 5 × 5 seeds = 25 | 9 GPU jobs × ~4 days |
| 5 | `bc.yaml` | Table II (behavior cloning row); needs the P2P-SAC runs of `main.yaml` to have passed T_p | 5 | a few GPU hours |

\* Three runs share one GPU. On Alvis's A40 an Isaac run makes about 20 steps per second, so 5 million steps take about 3 days, plus evaluations. On Arrhenius, measure the speed with step 6 of Section 4 first.

**Use one cluster per comparison:** all runs of one figure or table on the same cluster (Section 4.2).

**On a SLURM cluster**, on the login node, for each experiment file:

```bash
python3 scripts/make_sweep.py configs/experiments/main.yaml      # writes sweeps/main/ (plain python3 is enough)
sbatch sweeps/main/submit.sbatch                                 # submits the job array
```

Then the same for `planner_reference.yaml`, `budget.yaml` and `ablation.yaml` (they can run at the same time), and `bc.yaml` once the P2P-SAC runs of `main` are past step 100 000.

`make_sweep.py` prints every run it created and where its results will go. Useful options:

- `--seeds 0 1` runs a subset of seeds;
- `--only p2p-sac base-sac` runs a subset of runs;
- `--runs-per-job 2` puts fewer runs on one GPU.

**Without SLURM** (a single GPU machine), run the commands of a sweep one after another:

```bash
python3 scripts/make_sweep.py configs/experiments/main.yaml
bash scripts/run_local.sh sweeps/main/jobs.txt --container
```

**When everything has finished**, compute the paper's results ([Section 9](#9-from-logs-to-the-papers-figures-tables-and-numbers)).

---

## 7. How the SLURM jobs work

`make_sweep.py` writes three files into `sweeps/<experiment>/`:

| File | Content |
|---|---|
| `jobs.txt` | One complete training command per line: every run × every seed. You can copy any line and run it by hand. |
| `jobs.json` | The same commands with the run folder each one writes to, and the experiment file used. |
| `submit.sbatch` | The SLURM job array. Each array task takes `runs_per_job` lines (default 3) and runs them together on one GPU, started 2 minutes apart. |

What happens on the cluster:

1. **Logs of each job.** They go to `sweeps/<experiment>/slurm/<jobid>_<task>.out` and `.err`.
2. **Time limit** (`P2PAC_SLURM_TIME`, e.g. 7 days on Alvis). Fifteen minutes before it, SLURM sends a signal. Every run writes a checkpoint and stops (`state: preempted`), and the job puts itself back in the queue (`scontrol requeue`). When it starts again it runs the same commands: finished runs are skipped, unfinished ones continue from their checkpoint.
3. **Submitting again is always safe.** `sbatch sweeps/main/submit.sbatch` can be repeated at any time, for example after fixing a failed run. Only what is not finished will run.
4. **Checking progress.** `squeue -u $USER` shows the jobs; `python3 scripts/status.py --experiment main` shows the runs.
5. **When a run fails** (`state: failed`), the reason is in `events.jsonl` of the run and in the job's `.err` file. Fix the cause and submit again; the run continues from its last checkpoint.

The cluster-specific SLURM settings (account, partition, GPU request, CPUs, memory, time limit) come from `scripts/cluster.env` (see `scripts/clusters/`). The experiment-specific ones (runs per job, start delay) are in the `slurm:` block of `configs/experiments/_isaac_common.yaml`, which an experiment file can override.

---

## 8. Change settings and try your own variants

Every setting is an option of `p2pac.train`. [docs/configuration.md](docs/configuration.md) lists all of them with their default, meaning and whether the paper states the value. There are three ways to set an option; when the same option is set twice, the later one in this list wins:

1. **The method preset** (`--method p2p` sets the P2P-AC defaults, `--method base` switches the planner off, …).
2. **A YAML file**, `--config my_settings.yaml`, with option names written with underscores:
   ```yaml
   t_d: 100000
   gamma: 0.995
   batch_size: 512
   ```
3. **A flag**, with option names written with dashes: `--t-d 100000 --gamma 0.995`.

**Rule:** whenever you change a setting, also give the run a new `--variant` name (or a new `--experiment`). Otherwise the run would land in the folder of an existing run with other settings, and the program refuses to start.

**A set of variants as an experiment.** Write a new file in `configs/experiments/`, for example `configs/experiments/td_study.yaml`:

```yaml
name: td_study
description: P2P-SAC with different transition lengths T_d.
include: [_isaac_common.yaml]          # the shared Isaac settings (budget, evaluation, SLURM)
common:
  experiment: td_study                 # results go to runs/td_study/
seeds: [0, 1, 2, 3, 4]
runs:
  - {method: p2p, algo: sac, variant: td_0,      t_d: 0}
  - {method: p2p, algo: sac, variant: td_100000, t_d: 100000}
  - {method: p2p, algo: sac, variant: td_200000, t_d: 200000}
```

Then `python3 scripts/make_sweep.py configs/experiments/td_study.yaml` and `sbatch sweeps/td_study/submit.sbatch`. To compare the variants, add them to a copy of `configs/analysis/paper.yaml` (see the comments in that file) and run the aggregation.

**Settings shared by all paper experiments** (training budget, T_max, evaluation period and episodes, checkpoint interval, γ) are in `configs/experiments/_isaac_common.yaml`. A change there applies to every experiment generated afterwards, and is recorded in every run's `config.json`.

---

## 9. From logs to the paper's figures, tables and numbers

All results are computed from the run folders only; nothing is rerun.

```bash
p2pac_run python -m p2pac.analysis.aggregate --spec configs/analysis/paper.yaml --runs runs --out paper_results
p2pac_run python -m p2pac.analysis.theory --runs runs --out paper_results/theory_checks.csv
p2pac_run python -m p2pac.analysis.planner_gap --run runs/main/p2p-sac/seed_0 --out paper_results/planner_gap.csv
```

These need the container's Python packages but no GPU. Run them:

- on Alvis, on the login node;
- on Arrhenius, inside an interactive GPU job, because the login node cannot run the aarch64 container;
- or on a laptop with the environment of Section 3, after copying the `runs/` folder there (without the large `checkpoints/` folders).

| In the paper | Experiment | Runs used | Output in `paper_results/` | How to use it in LaTeX |
|---|---|---|---|---|
| Fig. 2 (learning curves) | `main` | all 8 methods | `curves/data_<method>.csv` (`steps_M`, `suc_mean`, `suc_lo`, `suc_hi`, …) | copy to the manuscript's `results/` folder |
| Table II (main results) | `main`, `planner_reference` | 8 methods, `bc`, `planner__budget_20` | `tables/table2_main_rows.tex` (+ `.csv`) | `\input` inside the `tabular` |
| Fig. 3 (budget) | `budget`, `main`, `planner_reference` | `p2p-sac__budget_*`, `p2p-sac`, `planner__budget_*` | `budget_sweep.csv` (`tcpu`, `n80_mean`, `n80_std`, `q_hat`, …) | copy to `results/` |
| Table III (ablations) | `main`, `ablation` | `p2p-sac`, `p2p-sac__<ablation>` | `tables/table3_ablation_rows.tex` (+ `.csv`) | `\input` |
| Numbers in the text | all | — | `numbers.tex` | `\input{numbers.tex}`, then e.g. `\res{table2_main/p2p-sac/n80}` |
| Sec. VI-F (removal), Prop. 3 | `main` | P2P runs | `theory_checks.csv` (`planner_queries_after_T_r` = 0, `kappa_max_after_T_r` = 0) and the curves around T_r | text |
| Assumptions 1–2, Prop. 2, Cor. 2 | `main` | P2P runs | `theory_checks.csv` | text / supplement |
| Cor. 1 (planner accuracy against budget) | any P2P run | one run | `planner_gap.csv`, `planner_gap_fit.json` | text / supplement |
| Sec. VI-H (hardware) | — | hardware trials | `python -m p2pac.deploy.hardware_analysis` ([Section 12](#12-hardware-experiments-with-the-unitree-go2)) | figure, text |

How the numbers are computed:

- **Success, crash and efficiency metrics** come from each seed's final evaluation (100 episodes of the final policy), reported as mean ± standard deviation over seeds. Path ratio, duration and speed are averaged over successful episodes only.
- **N80** is the first evaluation step with at least 80% success. A seed that never reaches 80% is counted as "> budget" and the table shows how many seeds reached it, e.g. `1.20 ± 0.30 (4/5)`.
- **The band in Fig. 2** is ± one standard deviation over seeds, clipped to 0–100% (`band: std` in `configs/analysis/paper.yaml`; `band: ci95` gives 95% confidence intervals, and the caption must then say so).
- **`provenance.json`** lists exactly which run folders, seeds and configurations went into each figure and table. Unfinished runs are listed there and left out of the results.

---

## 10. Where the results are saved

Each run has its own folder `runs/<experiment>/<method>-<algo>[__<variant>]/seed_<k>/`:

```
seed_0/
├── status.json          state (running / completed / failed / preempted) and current step
├── config.json          every setting of this run
├── manifest.json        code version (git commit), package versions, machine, GPU, SLURM job
├── events.jsonl         timeline: evaluations, end of planner phase, planner removal, checkpoints, restarts
├── logs/
│   ├── steps/*.npz      every environment step (state, actions, planner action, reward, κ, phase, ...)
│   ├── episodes.csv     every training episode (outcome, length, return, path ratio, ...)
│   ├── learner.jsonl    losses, Q-values, entropy, buffer statistics
│   └── sb3/             Stable-Baselines3's own logs and TensorBoard files
├── eval/
│   ├── summary.csv      one row per evaluation  ← the learning curve
│   ├── episodes.csv     one row per evaluation episode
│   ├── final.json       final evaluation of the final policy  ← Table II
│   ├── final_episodes.csv
│   ├── starts.json, final_starts.json      the fixed start positions
│   └── trajectories/    paths of all evaluation episodes
├── policies/            the policy at every evaluation
├── final_model.zip      the final policy
├── checkpoints/         everything needed to continue training
└── artifacts/           environment and planner description, planner data D_P (at T_p)
```

[docs/outputs.md](docs/outputs.md) explains every file and every column, and shows how to load them in Python.

---

## 11. Evaluate saved policies again

The policy evaluated at every evaluation is saved, so you can evaluate it again later, for example with more episodes, without training again:

```bash
# the final policy on 200 new start positions
p2pac_run python -m p2pac.evaluate --run-dir runs/main/p2p-sac/seed_0 --what final --episodes 200 --tag final200 --headless

# every saved policy of the run (a new learning curve)
p2pac_run python -m p2pac.evaluate --run-dir runs/main/p2p-sac/seed_0 --what snapshots --episodes 50 --tag curve50 --headless
```

Results go to `runs/.../eval/reeval_<tag>/`. An existing tag is never overwritten.

---

## 12. Hardware experiments with the Unitree Go2

`p2pac/deploy/go2_deploy.py` runs a trained policy on the real robot (Sec. VI-H). The setup is as follows:

- The policy runs on a computer connected to the robot.
- It reads the robot's position and sends velocity commands at 50 Hz through the Unitree SDK2 (`unitree_sdk2py`, robot sport mode). The planner is not used.
- Every control step is logged.

```bash
pip install unitree_sdk2py                      # on the computer next to the robot

# 1. dry run without a robot (simulated with the kinematic model)
python -m p2pac.deploy.go2_deploy --policy runs/main/p2p-sac/seed_0/final_model.zip \
       --run-config runs/main/p2p-sac/seed_0/config.json --pose kinematic --out hardware/dry_run

# 2. on the robot, with positions from a motion-capture system sent over UDP ("t,x,y,yaw" lines)
python -m p2pac.deploy.go2_deploy --policy runs/main/p2p-sac/seed_0/final_model.zip \
       --run-config runs/main/p2p-sac/seed_0/config.json --pose udp --udp-port 9870 \
       --origin <x0> <y0> --rotation-deg <angle> --iface <network interface> --trials 10 --out hardware/p2p-sac

# 3. summary over all trials (success rate with a confidence interval, durations, data for the figure)
python -m p2pac.deploy.hardware_analysis --trials hardware/p2p-sac --out paper_results/hardware --figure-trial 0
```

Notes:

- **Arena frame.** `--origin` and `--rotation-deg` map the motion-capture frame onto the arena frame used in training (goal at (0, 2.8) m).
- **Pose sources.** `--pose unitree_odom` uses the robot's own odometry instead of motion capture. The program that sends motion-capture positions over UDP depends on your lab and is not included.
- **Low-level controller.** On the robot, Unitree's built-in sport-mode controller replaces the Isaac Lab locomotion policy used in simulation; the paper should say so ([docs/paper_vs_code.md](docs/paper_vs_code.md)).
- **Heading.** `--yaw-hold-gain` (default 0) can keep the robot facing +y, as assumed by the policy.

---

## 13. Tests

```bash
pytest                    # about 2 minutes on a laptop, no GPU
```

The tests check, among other things:

- With the planner switched off, P2P-SAC/TD3/DDPG produce **exactly** the same numbers as Stable-Baselines3's SAC/TD3/DDPG. This is Proposition 3 in code.
- The phases: the planner acts only for t < T_p, gives labels only for t < T_r, and is never asked after T_r; κ follows Eq. (17).
- D_P is never overwritten; minibatches are half D_P, half D_O.
- The logit-space loss and its gradients.
- The planner's actions are feasible at every budget.
- The reward, outcome classification and start-position sampling.
- Stopping a run with the SLURM signal and continuing it gives continuous logs.
- The configuration system, the sweep generator and the statistics (N80 with unreached seeds, confidence intervals).
- Portability:
  - the shipped simulator assets, locomotion policy and Isaac Lab are complete and unchanged;
  - the package list is valid for x86-64 and aarch64;
  - the SLURM headers for Arrhenius and Alvis are correct;
  - the job tools run without numpy or PyYAML;
  - the code version is recorded without `git` in the container.

`docs/planner_validation.md` contains the checks of the planner implementation.

---

## 14. Troubleshooting

| Problem | Cause and solution |
|---|---|
| `run already completed, nothing to do` | The run folder already holds a finished run. Use a new `--variant`/`--experiment`, or `--force-new` to train it again in a new folder. |
| `configuration differs from the existing run` | You started an existing, unfinished run with other settings. Use a new `--variant`, or `--allow-config-change` if the change is intended (it is recorded). |
| `no Isaac Sim image for this node (aarch64)` / `(x86_64)` | There is no image for this node's CPU architecture (Section 4, step 3). On Arrhenius the container runs only on GPU nodes (aarch64); run container commands there, not on the login node. |
| `exec format error` | The image was made for the other CPU architecture. Use `get_container.sh --arch <arch>` or set `ISAACSIM_SIF_<arch>` correctly. |
| `Invalid account or account/partition combination` (Arrhenius) | GPU jobs need the GPU project, whose name ends in `-gpu` (`P2PAC_SLURM_ACCOUNT`). |
| Isaac Sim fails to start on Arrhenius | Possibly the GH200 limitation (Section 4.2). Keep the output, check it is not one of the other problems in this table, and fall back to Alvis if needed. |
| `--video` fails on Arrhenius | Camera rendering needs GPUs with RT cores; GH200 has none. Record videos on Alvis. |
| `apptainer/singularity not found` | Load it first, e.g. `module load Apptainer` (add the line to `scripts/cluster.env`). |
| `please use bash` | `scripts/container.sh` works only in bash: type `bash`, then `source scripts/container.sh`. |
| `WARNING: Could not find any nv files` | Normal on a node without a GPU (e.g. the analysis on the Alvis login node). |
| `ModuleNotFoundError` inside the container | The extra packages are not installed: run `bash scripts/setup/install_overlay.sh`. |
| `The local copy of the Isaac Sim assets is incomplete` | Files in `assets/isaac_sim/` are missing (e.g. the folder was not copied). Run `python3 scripts/setup/isaac_assets.py download` on a node with internet, or `export P2PAC_ISAAC_ASSETS=nvidia` to load them from NVIDIA's server. |
| `The repository path contains a space` | Move the repository to a path without spaces (Isaac Sim cannot handle them). |
| Isaac Sim fails at start-up | Read the job's `.err` file. Run the acceptance test (Section 4, step 5) on an interactive GPU node to see the full output. |
| GPU out of memory with 3 runs per GPU | Use `--runs-per-job 2` in `make_sweep.py` (or `runs_per_job: 2` in the experiment file). |
| The smoke test reports internal resets | Do not run experiments: the simulator resets the robot by itself. Check that `third_party/IsaacLab` is unmodified (`VENDORED.md`). |
| A run shows `failed` | Read the last lines of its `events.jsonl` and the job's `.err` file. After fixing the cause, submit the sweep again: the run continues from its last checkpoint. |
| A job keeps getting requeued | Normal: each job runs at most `P2PAC_SLURM_TIME`; runs continue after every requeue until they are finished. |

---

## 15. More documentation

| Document | Content |
|---|---|
| [docs/configuration.md](docs/configuration.md) | Every option: default value, meaning, whether the paper states it. Generated from the code (`python -m p2pac.config --markdown > docs/configuration.md`). |
| [docs/outputs.md](docs/outputs.md) | Every file a run writes and every column in it, with Python examples. |
| [docs/paper_vs_code.md](docs/paper_vs_code.md) | How the code implements each part of the paper, implementation details the paper should state, values still open in the manuscript, and text to correct. |
| [docs/planner_validation.md](docs/planner_validation.md) | How the planner's continuous-time flow is implemented, and the checks that it is feasible and converges. |
| `configs/experiments/*.yaml` | The definition of each experiment, with a description at the top. |
| `scripts/clusters/README.md` | The cluster settings and how to adapt them to another cluster. |
| `third_party/IsaacLab/VENDORED.md`, `assets/README.md`, `container/README.md`, `third_party/pyyaml/VENDORED.md` | Where the bundled files come from and how to check them. |

**Software versions:**

- Isaac Sim 5.1.0 and Isaac Lab 2.3.0 (commit `3c6e67bb`);
- Python 3.11;
- Stable-Baselines3 2.7.1, Gymnasium 1.2.0, RSL-RL 3.0.1;
- the exact list of extra packages is in `requirements/isaac-overlay.txt`.

---

## Publishing this folder on GitHub

The folder is ready to be pushed as it is (about 60 MB in Git). `.gitignore` keeps out everything that is large or
specific to your cluster: `runs/`, `sweeps/`, `paper_results/`, `scripts/cluster.env`, `.pipbase/`, `.isaac_cache/`,
`.apptainer_cache/`, `container/wheelhouse/` (one package is larger than GitHub's 100 MB file limit) and container
images. A clone from GitHub therefore lacks only the wheelhouse: either install the packages from the internet
(Section 4, step 4), recreate it with `scripts/setup/download_wheels.py`, or copy the folder instead of cloning it.

```bash
cd P2P-AC
git init
git add -A
git commit -m "P2P-AC code"
git remote add origin git@github.com:<user>/<repository>.git
git push -u origin main
```

Before making the repository public:

- add its address to `CITATION.cff`;
- check `assets/README.md` about redistributing NVIDIA's files (the locomotion checkpoint and the simulator assets).

Archive the `runs/` folder of the final experiments separately, for example on Zenodo, so that others can recompute
every result with Section 9 without training. The `checkpoints/` folders can be left out of that archive.

---

## Acknowledgements

This work was supported by the United States National Science Foundation (awards ECCS-2515358 and CNS-2502856), the Swedish Research Council (grant 2024-04058), and Sweden's Innovation Agency (Vinnova). Computational resources were provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS) at C3SE, partially funded by the Swedish Research Council (grant 2022-06725).

---

## License and citation

The code is released under the MIT license (`LICENSE`). Isaac Lab is under its own BSD-3-Clause license (`third_party/IsaacLab/LICENSE`). If you use this code, please cite the paper (`CITATION.cff`).
