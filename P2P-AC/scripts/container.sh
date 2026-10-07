#!/bin/bash
# Run commands inside the Isaac Sim container with everything this project needs.
#
# Usage (from anywhere, in bash):
#   source scripts/container.sh
#   p2pac_run python -m p2pac.train --method p2p --algo sac --seed 0 --experiment main --headless
#   p2pac_run python scripts/isaac_smoke_test.py --headless
#   p2pac_pip list                      # pip of the container's Python, installing into the overlay folder
#   p2pac_shell                         # interactive shell inside the container (for debugging)
#
# What it does:
#   * runs the command in the Isaac Sim 5.1 image with the GPU enabled (--nv);
#   * works on x86-64 nodes (e.g. Alvis) and on ARM/aarch64 nodes (e.g. the GH200 nodes of Arrhenius): the image,
#     the overlay package folder and the compatibility library are chosen for the CPU architecture of the node
#     it runs on (uname -m);
#   * mounts this repository at the SAME path inside the container and starts the command in the repository
#     root, so every path (run directories, configs, logs) is identical inside and outside the container;
#   * puts the repository, the overlay packages and the vendored Isaac Lab (third_party/IsaacLab) on PYTHONPATH and
#     uses Isaac Sim's own Python (/isaac-sim/python.sh);
#   * gives every process its own writable Isaac Sim cache folder under $P2PAC_CACHE_ROOT (Isaac Sim instances
#     running at the same time must not share caches) and deletes it when the process ends;
#   * records the code version with the host's git (the image has no git) for the run manifest;
#   * mounts container/compat/<arch>/libcrypto.so.1.1 (OpenSSL 1.1, missing from the image) unless
#     P2PAC_MOUNT_LIBCRYPTO11=0; on aarch64 it preloads libgomp, as required by Isaac Sim on ARM
#     (P2PAC_PRELOAD_GOMP=0 switches this off).
#
# Settings (all optional; put them in scripts/cluster.env, which is read automatically; a variable that is
# already set in your shell overrides the file):
#   ISAACSIM_SIF          the Isaac Sim 5.1.0 image (.sif) to use on this node
#   ISAACSIM_SIF_x86_64   the image for x86-64 nodes      (used when ISAACSIM_SIF is not set)
#   ISAACSIM_SIF_aarch64  the image for aarch64 nodes     (used when ISAACSIM_SIF is not set)
#                         default for both: <repo>/container/isaac-sim-5.1.0-<arch>.sif
#   P2PAC_PIPBASE         folder with the overlay packages. Default: <repo>/.pipbase/<arch>
#   P2PAC_CACHE_ROOT      folder for the per-process Isaac Sim caches. Default: <repo>/.isaac_cache
#   P2PAC_EXTRA_BINDS     extra "host:container" mounts, separated by spaces
#   P2PAC_KEEP_CACHE=1    keep the per-process cache folders (for debugging)
#   OMP_NUM_THREADS       passed into the container only if you set it

if [ -z "${BASH_VERSION:-}" ]; then
  echo "[container.sh] please use bash: run 'bash' first, then 'source scripts/container.sh'" >&2
  return 1 2>/dev/null || exit 1
fi

P2PAC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export P2PAC_ROOT
P2PAC_ARCH="$(uname -m)"            # x86_64 or aarch64
export P2PAC_ARCH

# read scripts/cluster.env; ISAACSIM_* / P2PAC_* variables already set in the shell take precedence over the file
if [[ -f "$P2PAC_ROOT/scripts/cluster.env" ]]; then
  _p2pac_keep=()
  for _v in $(compgen -v | grep -E '^(ISAACSIM_|P2PAC_)'); do
    _p2pac_keep+=("$_v=${!_v}")
  done
  source "$P2PAC_ROOT/scripts/cluster.env"
  for _kv in ${_p2pac_keep[@]+"${_p2pac_keep[@]}"}; do export "$_kv"; done
  unset _p2pac_keep _v _kv
fi

_p2pac_sif_var="ISAACSIM_SIF_${P2PAC_ARCH}"
export ISAACSIM_SIF="${ISAACSIM_SIF:-${!_p2pac_sif_var:-$P2PAC_ROOT/container/isaac-sim-5.1.0-${P2PAC_ARCH}.sif}}"
unset _p2pac_sif_var
export P2PAC_PIPBASE="${P2PAC_PIPBASE:-$P2PAC_ROOT/.pipbase/$P2PAC_ARCH}"
export P2PAC_CACHE_ROOT="${P2PAC_CACHE_ROOT:-$P2PAC_ROOT/.isaac_cache}"
export P2PAC_ISAACLAB="$P2PAC_ROOT/third_party/IsaacLab"

if command -v apptainer >/dev/null 2>&1; then P2PAC_CNT=apptainer
elif command -v singularity >/dev/null 2>&1; then P2PAC_CNT=singularity
else echo "[container.sh] apptainer/singularity not found (on a cluster: 'module load Apptainer' or similar)" >&2; fi
export P2PAC_CNT

_p2pac_exec() {
  # _p2pac_exec <cache dir> <command ...>
  local root="$1"; shift
  if [[ ! -r "$ISAACSIM_SIF" ]]; then
    echo "[container.sh] no Isaac Sim image for this node ($P2PAC_ARCH): $ISAACSIM_SIF" >&2
    echo "               set ISAACSIM_SIF_$P2PAC_ARCH in scripts/cluster.env, or download it:" >&2
    echo "               bash scripts/setup/get_container.sh --arch $P2PAC_ARCH" >&2
    echo "               (on Arrhenius the GPU nodes are aarch64: run container commands on a GPU node)" >&2
    return 1
  fi
  mkdir -p "$P2PAC_PIPBASE" "$root"/{kit_cache,kit_data,kit_logs,ov_cache,nvidia-omniverse,user_data,omni_config,xdg_cache,xdg_data,xdg_config,pip_cache,tmp}
  # mount points for two Omniverse folders in your home directory (empty folders; their content goes to $root)
  mkdir -p "$HOME/.cache/ov" "$HOME/.nvidia-omniverse"
  local binds=(
    --bind "$P2PAC_ROOT:$P2PAC_ROOT"
    --bind "$P2PAC_PIPBASE:/ext/pip"
    --bind "$root:$root"
    --bind "$root/kit_cache:/isaac-sim/kit/cache"
    --bind "$root/kit_data:/isaac-sim/kit/data"
    --bind "$root/kit_logs:/isaac-sim/kit/logs"
    --bind "$root/user_data:/isaac-sim/user_data"
    --bind "$root/ov_cache:$HOME/.cache/ov"
    --bind "$root/nvidia-omniverse:$HOME/.nvidia-omniverse"
  )
  local lib="$P2PAC_ROOT/container/compat/$P2PAC_ARCH/libcrypto.so.1.1"
  if [[ "${P2PAC_MOUNT_LIBCRYPTO11:-1}" == "1" && -f "$lib" ]]; then
    binds+=(--bind "$lib:/usr/lib/${P2PAC_ARCH}-linux-gnu/libcrypto.so.1.1"
            --bind "$lib:/lib/${P2PAC_ARCH}-linux-gnu/libcrypto.so.1.1")
  fi
  local preload=""
  if [[ "$P2PAC_ARCH" == "aarch64" && "${P2PAC_PRELOAD_GOMP:-1}" == "1" ]]; then
    preload="LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1"
  fi
  local b
  for b in ${P2PAC_EXTRA_BINDS:-}; do binds+=(--bind "$b"); done
  local il="$P2PAC_ISAACLAB/source"
  "$P2PAC_CNT" exec --nv --cleanenv "${binds[@]}" --pwd "$P2PAC_ROOT" "$ISAACSIM_SIF" env \
    ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PYTHONUSERBASE=/ext/pip PYTHONNOUSERSITE=0 \
    PATH=/ext/pip/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH="$P2PAC_ROOT:/ext/pip/lib/python3.11/site-packages:$il:$il/isaaclab:$il/isaaclab_tasks:$il/isaaclab_rl" \
    P2PAC_ROOT="$P2PAC_ROOT" ISAACLAB_ROOT="$P2PAC_ISAACLAB" ISAACSIM_SIF="$ISAACSIM_SIF" \
    OMNI_CONFIG_PATH="$root/omni_config" XDG_CACHE_HOME="$root/xdg_cache" XDG_DATA_HOME="$root/xdg_data" \
    XDG_CONFIG_HOME="$root/xdg_config" PIP_CACHE_DIR="$root/pip_cache" TMPDIR="$root/tmp" GIT_PYTHON_REFRESH=quiet \
    ${OMP_NUM_THREADS:+OMP_NUM_THREADS=$OMP_NUM_THREADS} ${preload} \
    SLURM_JOB_ID="${SLURM_JOB_ID:-}" SLURM_ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-}" SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-}" \
    SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST:-}" SLURM_RESTART_COUNT="${SLURM_RESTART_COUNT:-0}" \
    P2PAC_GIT_STATE_DIR="$root/git_state" P2PAC_ISAAC_ASSETS="${P2PAC_ISAAC_ASSETS:-local}" P2PAC_ARCH="$P2PAC_ARCH" \
    "$@"
}

_p2pac_record_git() {
  # The Isaac Sim image has no git: record the code version here, on the host, for the run manifest.
  local d="$1/git_state"
  mkdir -p "$d"
  if command -v git >/dev/null 2>&1 && git -C "$P2PAC_ROOT" rev-parse HEAD >/dev/null 2>&1; then
    git -C "$P2PAC_ROOT" rev-parse HEAD > "$d/commit" 2>/dev/null
    git -C "$P2PAC_ROOT" rev-parse --abbrev-ref HEAD > "$d/branch" 2>/dev/null
    git -C "$P2PAC_ROOT" describe --always --dirty --tags > "$d/describe" 2>/dev/null
    git -C "$P2PAC_ROOT" status --porcelain > "$d/status" 2>/dev/null
    git -C "$P2PAC_ROOT" diff HEAD > "$d/diff.patch" 2>/dev/null
  fi
}

_p2pac_with_cache() {
  # run a command with a fresh per-process cache folder; remove the folder afterwards
  local root="$P2PAC_CACHE_ROOT/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_$$_${RANDOM}${RANDOM}"
  mkdir -p "$root"
  _p2pac_record_git "$root"
  _p2pac_exec "$root" "$@"
  local code=$?
  [[ "${P2PAC_KEEP_CACHE:-0}" == "1" ]] || rm -rf "$root"
  return $code
}

# p2pac_run python <args>   ('python' means Isaac Sim's interpreter /isaac-sim/python.sh)
p2pac_run() {
  if [[ "${1:-}" == "python" || "${1:-}" == "python3" ]]; then shift; fi
  _p2pac_with_cache /isaac-sim/python.sh "$@"
}

# p2pac_pip <pip args>      (installs go to $P2PAC_PIPBASE)
p2pac_pip() {
  if [[ "${1:-}" == "install" ]]; then shift; _p2pac_with_cache /isaac-sim/python.sh -m pip install --user "$@"
  else _p2pac_with_cache /isaac-sim/python.sh -m pip "$@"; fi
}

# p2pac_shell               (interactive shell inside the container)
p2pac_shell() { _p2pac_with_cache bash; }

export -f _p2pac_exec _p2pac_record_git _p2pac_with_cache p2pac_run p2pac_pip p2pac_shell
