# Frozen locomotion policy

The learned navigation policy outputs velocity commands; a frozen locomotion policy turns them into joint targets
for the Go2 (Sec. VI-A). It is **not trained in this project**. It is Isaac Lab's pretrained RSL-RL policy for the
task `Isaac-Velocity-Flat-Unitree-Go2-v0`, downloaded by Isaac Lab 2.3.0 (`--use_pretrained_checkpoint`) and
copied here so the repository is self-contained.

| File | What it is | MD5 |
|---|---|---|
| `Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt` | RSL-RL checkpoint loaded by `p2pac/envs/isaac_go2.py` (the only file the code uses) | `50f9fe1f7ae965ab3e7dc975156f3adb` |
| `Isaac-Velocity-Flat-Unitree-Go2-v0/exported/policy.pt` | the same policy exported as TorchScript (for reference) | `8c0a486e6a04de0bbc056698f994d2b9` |
| `Isaac-Velocity-Flat-Unitree-Go2-v0/exported/policy.onnx` | the same policy exported as ONNX (for reference) | `1117e10a3936f85f24ac09caf01490a2` |

Every run records the path and SHA-256 of the checkpoint it loaded (`artifacts/environment.json`), and
`scripts/isaac_smoke_test.py` checks the MD5. To use another locomotion policy, pass `--locomotion-checkpoint <file>`.

License: Isaac Lab is released by NVIDIA under the BSD-3-Clause license (`third_party/IsaacLab/LICENSE`). The
checkpoint is downloaded by Isaac Lab from NVIDIA's asset server; before making the repository public, check that
NVIDIA's terms allow redistributing it. If they do not, delete the file from the repository and let users obtain it
with Isaac Lab itself (`third_party/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py --task
Isaac-Velocity-Flat-Unitree-Go2-v0 --use_pretrained_checkpoint` downloads it to `.pretrained_checkpoints/`), then
pass `--locomotion-checkpoint <path>`.
