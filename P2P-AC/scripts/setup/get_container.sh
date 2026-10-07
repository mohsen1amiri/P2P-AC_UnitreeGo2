#!/bin/bash
# Download the Isaac Sim 5.1.0 container image (only needed if your HPC does not provide it).
#
#   bash scripts/setup/get_container.sh                     # for the CPU architecture of this node
#   bash scripts/setup/get_container.sh --arch aarch64      # for ARM nodes (e.g. Arrhenius GH200), from any node
#   bash scripts/setup/get_container.sh --arch x86_64       # for x86-64 nodes (e.g. Alvis)
#
# Writes container/isaac-sim-5.1.0-<arch>.sif. Uses 'apptainer pull' (or 'singularity pull'), which works as a
# normal user: no root, no fakeroot, nothing installed. NVIDIA publishes nvcr.io/nvidia/isaac-sim:5.1.0 for both
# amd64 (x86_64) and arm64 (aarch64). All temporary files go to <repo>/.apptainer_cache (not to your home
# directory); delete that folder afterwards. Needs internet access (e.g. a login node) and ~30 GB of free space
# while converting.
# If NGC asks for credentials: create an API key at https://ngc.nvidia.com and run
#   export APPTAINER_DOCKER_USERNAME='$oauthtoken'; export APPTAINER_DOCKER_PASSWORD=<your API key>
set -euo pipefail
IMAGE_URI="docker://nvcr.io/nvidia/isaac-sim:5.1.0"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ARCH="$(uname -m)"
OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch) ARCH="$2"; shift 2;;
    *) OUT="$1"; shift;;
  esac
done
case "$ARCH" in
  x86_64|amd64) ARCH=x86_64; DOCKER_ARCH=amd64;;
  aarch64|arm64) ARCH=aarch64; DOCKER_ARCH=arm64;;
  *) echo "unknown architecture: $ARCH (use x86_64 or aarch64)" >&2; exit 1;;
esac
OUT="${OUT:-$REPO/container/isaac-sim-5.1.0-$ARCH.sif}"
if command -v apptainer >/dev/null 2>&1; then CNT=apptainer
elif command -v singularity >/dev/null 2>&1; then CNT=singularity
else echo "apptainer/singularity not found (on most HPCs: 'module load Apptainer', or ask which module provides it)" >&2; exit 1; fi
if [[ -e "$OUT" ]]; then echo "$OUT already exists; delete it first to download it again." >&2; exit 1; fi
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$REPO/.apptainer_cache/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$REPO/.apptainer_cache/tmp}"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR" SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$(dirname "$OUT")"
echo "Downloading $IMAGE_URI ($DOCKER_ARCH) -> $OUT (this takes a while) ..."
"$CNT" pull --arch "$DOCKER_ARCH" "$OUT" "$IMAGE_URI"
echo "Done. scripts/container.sh uses it automatically on $ARCH nodes"
echo "(or set ISAACSIM_SIF_$ARCH in scripts/cluster.env). You can now delete $REPO/.apptainer_cache"
