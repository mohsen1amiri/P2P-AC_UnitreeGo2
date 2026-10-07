# Vendored Isaac Lab

This folder is an unmodified copy of **Isaac Lab v2.3.0**, commit
`3c6e67bb5c7ada942a6d1884ab69338f57596f77` (https://github.com/isaac-sim/IsaacLab), the version used for all
experiments. It is included so that the repository is self-contained; nothing in it is installed. The container
helper (`scripts/container.sh`) mounts it and puts `source/isaaclab`, `source/isaaclab_tasks` and
`source/isaaclab_rl` on `PYTHONPATH`; Isaac Sim's extension system finds the remaining packages (for example
`isaaclab_assets`) through `apps/` and `source/`.

Omitted from the copy (not needed to run anything): `docs/`, `.github/`, `.vscode/` and the repository dot-files
(`.gitignore`, `.gitattributes`, `.pre-commit-config.yaml`, `.flake8`).

Note for Git: four JSON files of Isaac Lab's project template (`tools/template/templates/external/.vscode/*.json`) are
excluded by a `.gitignore` that Isaac Lab itself places in that folder, so `git add -A` skips them. They are not used
by this project. To commit them anyway: `git add -f third_party/IsaacLab/tools/template/templates/external/.vscode`.

To check that the copy is unmodified:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git /tmp/IsaacLab && git -C /tmp/IsaacLab checkout 3c6e67bb
diff -r /tmp/IsaacLab/source third_party/IsaacLab/source && echo identical
```

Isaac Lab is released under the BSD-3-Clause license (`LICENSE`); the mimic extension under Apache-2.0
(`LICENSE-mimic`).
