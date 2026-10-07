#!/bin/bash
# Install the Python packages this project needs on top of the Isaac Sim container (one time, ~5 minutes).
#
#   bash scripts/setup/install_overlay.sh
#
# Nothing is installed into the system, the container image or your home directory: the exact versions listed in
# requirements/isaac-overlay.txt are installed with the container's own pip into $P2PAC_PIPBASE (default:
# <repo>/.pipbase), which scripts/container.sh puts on the container's PYTHONPATH.
#
# Run it on the kind of node the experiments run on: the container and the packages must match the node's CPU
# architecture (x86_64 on Alvis; aarch64 on the GPU nodes of Arrhenius, so there: inside a GPU job).
# The packages go to .pipbase/<arch>, so x86_64 and aarch64 installations can live side by side.
#
# Source of the packages:
#   * container/wheelhouse/<arch>/ if it exists (made by scripts/setup/download_wheels.py): no internet needed;
#   * otherwise the Python Package Index (needs internet access from this node).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO/scripts/container.sh"
REQ="$REPO/requirements/isaac-overlay.txt"
WHEELS="$REPO/container/wheelhouse/$P2PAC_ARCH"
mkdir -p "$P2PAC_PIPBASE"
if compgen -G "$WHEELS/*.whl" >/dev/null; then
  echo "Installing from $WHEELS (offline) into $P2PAC_PIPBASE ..."
  p2pac_pip install --no-deps --no-index --find-links "$WHEELS" --no-warn-script-location -r "$REQ"
else
  echo "Installing from the Python Package Index into $P2PAC_PIPBASE ..."
  p2pac_pip install --no-deps --no-warn-script-location -r "$REQ"
fi
echo "Checking the installation ..."
p2pac_run python - <<'PY'
import importlib, importlib.metadata as md
for mod, dist in [("stable_baselines3", "stable-baselines3"), ("gymnasium", "gymnasium"), ("rsl_rl", "rsl-rl-lib"),
                  ("tensordict", "tensordict"), ("pandas", "pandas"), ("yaml", None), ("scipy", None),
                  ("isaaclab", None), ("isaaclab_tasks", None), ("isaaclab_rl", None), ("p2pac", None)]:
    importlib.import_module(mod)
    print(f"  ok  {mod:20s} {md.version(dist) if dist else ''}")
PY
echo "Overlay installed in $P2PAC_PIPBASE ($P2PAC_ARCH)"
