# Container files

The simulator runs inside the NVIDIA Isaac Sim 5.1.0 container image, started with Apptainer (or Singularity)
as a normal user. Nothing is installed on the HPC system: the image is a single file, and everything else this
project needs is inside the repository folder. Files that depend on the CPU architecture exist once per
architecture: `x86_64` (e.g. Alvis) and `aarch64` (ARM, e.g. the GH200 GPU nodes of Arrhenius).
`scripts/container.sh` picks the right ones on each node.

| File / folder | What it is | In Git? |
|---|---|---|
| `isaac-sim-5.1.0-<arch>.sif` | The Isaac Sim 5.1.0 image (several GB). Use the one your HPC provides (Alvis: `/apps/containers/IsaacSim/IsaacSim-NGC-5.1.0.sif`, via `ISAACSIM_SIF_x86_64` in `scripts/cluster.env`), or download it as a normal user with `bash scripts/setup/get_container.sh --arch <x86_64\|aarch64>` (`apptainer pull` of `nvcr.io/nvidia/isaac-sim:5.1.0`, which NVIDIA publishes for both architectures; no root needed). | no (too large) |
| `wheelhouse/<arch>/` | The Python packages of `requirements/isaac-overlay.txt` as wheel files (x86_64: 56 files, ~450 MB; aarch64: 55 files, ~270 MB, without `triton`, which has no aarch64 build and is not used), made by `python3 scripts/setup/download_wheels.py --arch <arch>`. When present, `scripts/setup/install_overlay.sh` installs from it without internet. | no (one file exceeds GitHub's 100 MB limit) |
| `compat/<arch>/libcrypto.so.1.1` | OpenSSL 1.1 crypto library, which the Isaac Sim 5.1 image does not contain. The experiments on Alvis were run with it mounted into the container. Both files come from the same official Rocky Linux 8 package release, `openssl-libs-1.1.1k-14.el8_10` (`x86_64` and `aarch64` builds; the x86_64 file is byte-identical to the one used in the experiments). `scripts/container.sh` mounts the one matching the node; `P2PAC_MOUNT_LIBCRYPTO11=0` switches this off. | yes |
