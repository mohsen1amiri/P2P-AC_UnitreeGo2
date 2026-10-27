# Complete IEEE TAI P2P-AC Arrhenius campaign

This directory contains actual Slurm sbatches and job lists for the validated repository at:

`/nobackup/proj/disk/naiss2024-22-1645/personal/mohsena/P2P-AC_UnitreeGo2/P2P-AC`

## Counts

- **145 RL trainings**: 5 million steps, 5 seeds, 7 experiment YAML groups, no duplicated reference variants.
- **5 behavior-cloning runs**: use planner replay buffers from the 5 main P2P-SAC runs.
- **3 new REAP-only reference runs** at 1, 5, 15 ms. The already archived 10 ms result is reused.
- **153 unique scientific runs total**. Work divided **51 / 51 / 51** across three GPU accounts.
- **290 separately scheduled policy evaluations**: 145 x 10-episode snapshot-curve jobs, and 145 x 100-episode final-policy jobs. Training itself is in `snapshot` mode.

## Scheduling and dependencies

- Every training sbatch: **1 GH200 per array task**, 5 independent Isaac Sim processes per GH200, **36 CPU cores**, **100 GB RAM**, **3-day max walltime**, and Slurm checkpoint/requeue support.
- Up to **2 simultaneous GPU jobs per account** for training and evaluation (6 GPUs maximum across all three accounts, when resources are available). Change `#SBATCH --array=...%2` to `%1` or `%4` if preferred.
- Reference jobs: one REAP-only evaluation per account.
- BC jobs: one GPU (all 5 concurrently, allocated to Alireza's account), scheduled after all three training arrays have completed successfully.
- Snapshot evaluations: after the corresponding training array completes; final evaluations: after snapshots finish.
- Outputs, checkpoints, and logs are written to the original shared repository path under Mohsen's storage project; Slurm GPU billing uses the respective accounts.

**Caution:** five processes on one GH200 have not been tested. Two passed; if five cause OOM, reduce the grouping to 2 and regenerate or adjust the job list and array together. Do not merely edit the printed number `5` without changing the `--array` bounds.

## After extracting this archive *in the repository root*

Verify the generated commands against your installed YAML configurations:

```bash
python3 p2pac_campaign/verify_campaign.py
```

Inspect available Slurm sbatches:

```bash
ls p2pac_campaign/*.sbatch
```

Submit all stages (including evaluation + BC dependencies):

```bash
bash p2pac_campaign/submit_all.sh
```

Submission IDs are stored under `p2pac_campaign/submission_logs/`. Output and error logs are under `p2pac_campaign/logs/<stage>_<account>/`. Submission files must be generated and submitted from the validated repository path given above, on the **login node**.

## Evaluation seed protocol

- Policy snapshots: 10 fixed episodes, `--starts-seed 1000003`, policy snapshots every 50,000 steps (including step 0 if saved).
- Final checkpoint: 100 distinct episodes, `--starts-seed 1000004` (matches `eval_seed + 1` of the paper's run protocol, overriding the separate evaluator's default of `eval_seed + 2`).
- N80 must be computed using **two consecutive evaluation rounds** with success at least 80%, not merely the first crossing; verify downstream analysis before paper reporting.

## Reproducibility

Commands and account assignment are in `jobs/*.json` and `manifest.json`. The complete set of 153 scientific jobs has unique run directories. Every run preserves the repository's task and algorithm values (we do not modify original source or experiment YAMLs).

**Potential account policy:** The storage remains in the original Mohsen project folder even when GPU hours are charged to Sindri's/Alireza's Slurm accounts. Check this against NAISS project storage policy if required; in particular shared access on login does not itself prove compliant cross-project quota charging.
