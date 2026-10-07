# Cluster settings

Copy the file for your cluster to `scripts/cluster.env` and set your project account in it:

```bash
cp scripts/clusters/arrhenius.env scripts/cluster.env     # Arrhenius (NAISS), GH200 GPU nodes, aarch64
cp scripts/clusters/alvis.env scripts/cluster.env         # Alvis (C3SE/NAISS), A40 GPU nodes, x86_64
```

For another cluster, start from either file. The variables are:

| Variable | Meaning |
|---|---|
| `P2PAC_SLURM_ACCOUNT` | `#SBATCH --account` |
| `P2PAC_SLURM_PARTITION` | `#SBATCH --partition` |
| `P2PAC_SLURM_GPUS` or `P2PAC_SLURM_GPUS_PER_NODE` | `#SBATCH --gpus=<n>` or `#SBATCH --gpus-per-node=<type:n>` |
| `P2PAC_SLURM_CPUS_PER_TASK`, `P2PAC_SLURM_MEM` | `#SBATCH --cpus-per-task`, `#SBATCH --mem` (clusters with shared nodes) |
| `P2PAC_SLURM_TIME` | `#SBATCH --time` (overrides the experiment file) |
| `P2PAC_SLURM_EXTRA` | further `#SBATCH` options, separated by `;` |
| `ISAACSIM_SIF_x86_64`, `ISAACSIM_SIF_aarch64` | the Isaac Sim 5.1.0 image for each CPU architecture (default: `container/isaac-sim-5.1.0-<arch>.sif`) |
| `P2PAC_CACHE_ROOT` | where the per-process Isaac Sim caches go (default: `<repo>/.isaac_cache`) |
| `P2PAC_EXTRA_BINDS` | extra folders to mount into the container |

A variable already set in your shell overrides the file.
