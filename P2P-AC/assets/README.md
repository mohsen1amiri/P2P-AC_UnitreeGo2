# Assets

Files the simulation loads, kept in the repository so that runs need no downloads.

| Folder | Content | Details |
|---|---|---|
| `locomotion_policy/` | The frozen Go2 locomotion policy (Isaac Lab's pretrained RSL-RL checkpoint). | [locomotion_policy/README.md](locomotion_policy/README.md) |
| `isaac_sim/` | The 12 NVIDIA Isaac Sim 5.1 asset files the Go2 task loads: the robot model and its meshes, the flat ground and its textures, the sky texture, the arrow marker (videos only), and the tile material of Isaac Lab's terrain configuration. 45 MB. | below |

**`isaac_sim/`.** Isaac Sim normally downloads these files from NVIDIA's asset server when a run starts. Here they
are stored locally, and `scripts/container.sh` points Isaac Sim to them (`P2PAC_ISAAC_ASSETS=local`, the default),
so compute nodes need no internet access and every run uses exactly the same files. They are byte-identical to the
files Isaac Sim downloaded during the original experiments.

- The list with sizes and SHA-256 checksums is `isaac_sim_manifest.json`.
- `python3 scripts/setup/isaac_assets.py check` verifies the copy.
- `python3 scripts/setup/isaac_assets.py download` restores it from NVIDIA's server (plain Python 3, nothing to install).
- To use NVIDIA's server instead for a run: `export P2PAC_ISAAC_ASSETS=nvidia`.

Every run records which source it used in `manifest.json` (`simulator.assets`).

**Licenses.** The files in `isaac_sim/` and `locomotion_policy/` are NVIDIA content distributed with Isaac Sim and
Isaac Lab. Before making the repository public, check that NVIDIA's terms allow redistributing them. If they do
not, delete the files from the repository; the scripts above download them again.
