"""NVIDIA Isaac Lab backend: Unitree Go2 driven by a frozen velocity-tracking locomotion policy (Sec. VI-A).

The learned (or planner) policy issues body-frame velocity commands ``[v_x, v_y]`` at 50 Hz; a frozen
RSL-RL locomotion policy (the Isaac Lab pretrained checkpoint for ``Isaac-Velocity-Flat-Unitree-Go2-v0``, shipped
in ``assets/locomotion_policy/``) converts them into joint targets at 200 Hz physics (``sim.dt = 0.005``,
``decimation = 4``).

How the Isaac Lab task is adapted to the navigation MDP:

* **No internal terminations.** Isaac Lab's velocity task ends episodes after ``episode_length_s = 20`` s and on
  base contact, and then re-spawns the robot *inside* ``env.step``. The navigation task has its own episode
  logic (goal, failure, ``T_max``), so both Isaac terminations are disabled and ``episode_length_s`` is set
  longer than ``T_max``. Trunk contact is read from the contact sensor after every step. Any internal reset or
  teleport (> 0.2 m in one step) is detected and logged as ``internal_reset``; it must never happen.
* **Seeding.** ``env_cfg.seed`` is set, which seeds Isaac's startup randomisation (e.g. the base mass).
* **Commands.** The velocity command is written into the slice of the policy observation that belongs to the
  ``velocity_commands`` term (located from the observation manager, not hard-coded) and into the command
  manager. Command resampling, heading control and standing environments are disabled, so the command manager
  never overwrites the command.
* **Consistent reset.** After placing the robot at the start pose the scene is updated and the policy
  observation is recomputed, so the first locomotion action uses the new state.
* **Task frame.** Positions are expressed relative to the environment origin (``scene.env_origins``).

The module must be imported **after** the Isaac Sim application has been launched (see :mod:`p2pac.train`).
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Optional

import numpy as np

from p2pac.envs.base import RobotState
from p2pac.paths import DEFAULT_LOCOMOTION_CHECKPOINT as DEFAULT_CHECKPOINT

TASK_NAME = "Isaac-Velocity-Flat-Unitree-Go2-v0"


def sha256_of(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _yaw_from_quat_wxyz(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def build_env_cfg(task_cfg, *, seed: int, num_envs: int = 1, episode_length_s: float, obs_noise: bool, debug_vis: bool):
    """Create the Isaac Lab environment configuration of the navigation arena."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg

    cfg = UnitreeGo2FlatEnvCfg()
    cfg.seed = int(seed)
    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = 5.0
    cfg.sim.dt = 0.005
    cfg.decimation = 4
    cfg.episode_length_s = float(episode_length_s)

    # --- obstacles (true radius; the planner applies its own tightening) -------------------------
    for i, (ox, oy, r) in enumerate(task_cfg.obstacles):
        setattr(cfg.scene, f"obstacle_{i}", AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/obstacle_{i}",
            spawn=sim_utils.CylinderCfg(
                radius=float(r), height=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(float(ox), float(oy), 0.5)),
        ))

    # --- goal marker (visual only) -----------------------------------------------------------------
    cfg.scene.goal_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/goal_marker",
        spawn=sim_utils.SphereCfg(radius=0.15, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(task_cfg.goal_x, task_cfg.goal_y, 0.1)),
    )

    # --- arena walls (inner faces on the arena bounds, thickness 0.1 m) ----------------------------
    t, h = 0.1, 1.0
    xr, yr = task_cfg.x_max - task_cfg.x_min, task_cfg.y_max - task_cfg.y_min
    xc, yc = 0.5 * (task_cfg.x_min + task_cfg.x_max), 0.5 * (task_cfg.y_min + task_cfg.y_max)
    walls = {
        "wall_bottom": ((xr + t, t, h), (xc, task_cfg.y_min - t / 2, h / 2)),
        "wall_top": ((xr + t, t, h), (xc, task_cfg.y_max + t / 2, h / 2)),
        "wall_left": ((t, yr + t, h), (task_cfg.x_min - t / 2, yc, h / 2)),
        "wall_right": ((t, yr + t, h), (task_cfg.x_max + t / 2, yc, h / 2)),
    }
    for name, (size, pos) in walls.items():
        setattr(cfg.scene, name, AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.CuboidCfg(
                size=tuple(float(s) for s in size),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 1.0)),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(float(p) for p in pos)),
        ))

    if cfg.viewer is not None:
        cfg.viewer.resolution = (1280, 720)
        cfg.viewer.eye = (xc, yc, 10.0)
        cfg.viewer.lookat = (xc, yc, 0.0)

    # --- terminations: handled by the navigation task, never by Isaac Lab ---------------------------
    cfg.terminations.time_out = None
    cfg.terminations.base_contact = None

    # --- commands: never resampled, no heading control, no standing envs ---------------------------
    cmd = cfg.commands.base_velocity
    cmd.resampling_time_range = (1.0e9, 1.0e9)
    cmd.heading_command = False
    cmd.rel_standing_envs = 0.0
    cmd.debug_vis = bool(debug_vis)

    # Observation noise of the locomotion policy (Isaac Lab's default for this task: enabled).
    cfg.observations.policy.enable_corruption = bool(obs_noise)
    return cfg


class IsaacGo2Backend:
    """Robot backend wrapping an Isaac Lab ``ManagerBasedRLEnv`` and the frozen locomotion policy."""

    def __init__(
        self,
        task_cfg,
        *,
        seed: int,
        device: str = "cuda:0",
        render_mode: Optional[str] = None,
        locomotion_checkpoint: str = DEFAULT_CHECKPOINT,
        episode_length_s: Optional[float] = None,
        obs_noise: bool = True,
        spawn_height: float = 0.5,
        contact_threshold: float = 1.0,
        teleport_threshold: float = 0.2,
    ):
        import gymnasium as gym
        import torch
        from isaaclab.managers import SceneEntityCfg
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from rsl_rl.runners import OnPolicyRunner

        self._torch = torch
        self.device = device
        self.dt = float(task_cfg.dt)
        self.spawn_height = float(spawn_height)
        self.contact_threshold = float(contact_threshold)
        self.teleport_threshold = float(teleport_threshold)
        if episode_length_s is None:
            # Long enough that Isaac never times out before the task does (T_max * dt + margin).
            episode_length_s = (task_cfg.max_episode_steps + 100) * task_cfg.dt
        self.episode_length_s = float(episode_length_s)

        cfg = build_env_cfg(
            task_cfg, seed=seed, episode_length_s=self.episode_length_s,
            obs_noise=obs_noise, debug_vis=render_mode is not None,
        )
        self.env = gym.make(TASK_NAME, cfg=cfg, render_mode=render_mode)
        self.uenv = self.env.unwrapped
        if abs(self.uenv.step_dt - self.dt) > 1e-9:
            raise RuntimeError(f"Isaac step_dt={self.uenv.step_dt} differs from task dt={self.dt}")
        self.render_mode = render_mode
        if render_mode == "rgb_array":
            self._prime_renderer()

        # --- frozen locomotion policy (Isaac Lab pretrained checkpoint) -------------------------
        self.locomotion_checkpoint = os.path.abspath(locomotion_checkpoint)
        if not os.path.isfile(self.locomotion_checkpoint):
            raise FileNotFoundError(f"locomotion checkpoint not found: {self.locomotion_checkpoint}")
        self.locomotion_checkpoint_sha256 = sha256_of(self.locomotion_checkpoint)
        vec_env = RslRlVecEnvWrapper(self.env)
        agent_cfg = load_cfg_from_registry(TASK_NAME, "rsl_rl_cfg_entry_point")
        self._runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=device)
        self._runner.load(self.locomotion_checkpoint)
        self._policy = self._runner.get_inference_policy(device=device)

        # --- locate the velocity-command slice in the policy observation ---------------------------
        om = self.uenv.observation_manager
        names = list(om.active_terms["policy"])
        dims = [int(np.prod(d)) for d in om.group_obs_term_dim["policy"]]
        if "velocity_commands" not in names:
            raise RuntimeError(f"policy observation has no 'velocity_commands' term: {names}")
        k = names.index("velocity_commands")
        start = int(sum(dims[:k]))
        if dims[k] != 3:
            raise RuntimeError(f"velocity_commands term has dimension {dims[k]}, expected 3")
        self._cmd_slice = slice(start, start + 3)
        self.policy_obs_layout = list(zip(names, dims))

        # --- contact sensor on the trunk ------------------------------------------------------------
        self._contact_cfg = SceneEntityCfg("contact_forces", body_names="base")
        self._contact_cfg.resolve(self.uenv.scene)
        self._robot = self.uenv.scene["robot"]
        self._obs = None
        self._last_xy = np.zeros(2)

    # ------------------------------------------------------------------------------------------
    def _prime_renderer(self) -> None:
        """Isaac's renderer sometimes needs a few app updates before the first frame."""
        import omni.kit.app

        self.env.reset()
        app = omni.kit.app.get_app()
        for _ in range(50):
            try:
                app.update()
                self.uenv.sim.render()
                self.env.render()
                return
            except TypeError:
                continue
        raise RuntimeError("Renderer failed to initialise.")

    def _origin(self) -> np.ndarray:
        return self.uenv.scene.env_origins[0, :2].detach().cpu().numpy().astype(np.float64)

    def _read_state(self, contact: bool, internal_reset: bool) -> RobotState:
        d = self._robot.data
        pos = d.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        xy = pos[:2] - self._origin()
        yaw = _yaw_from_quat_wxyz(d.root_quat_w[0].detach().cpu().numpy())
        vb = d.root_lin_vel_b[0, :2].detach().cpu().numpy().astype(np.float64)
        jump = float(np.linalg.norm(xy - self._last_xy))
        if jump > self.teleport_threshold:
            internal_reset = True
        self._last_xy = xy.copy()
        return RobotState(xy=xy, height=float(pos[2]), yaw=yaw, vel_body=vb, contact=contact, internal_reset=internal_reset)

    # ------------------------------------------------------------------------------------------
    def reset_to(self, x: float, y: float, yaw: float) -> RobotState:
        torch = self._torch
        self._obs, _ = self.env.reset()
        o = self._origin()
        quat = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]          # (w, x, y, z)
        pose = torch.tensor([[x + o[0], y + o[1], self.spawn_height, *quat]], device=self.uenv.device, dtype=torch.float32)
        self._robot.write_root_pose_to_sim(pose)
        self._robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.uenv.device))
        self.uenv.sim.step(render=False)
        self.uenv.scene.update(dt=self.uenv.physics_dt)
        self._obs = self.uenv.observation_manager.compute()                 # observation after the teleport
        self._last_xy = np.array([x, y], dtype=np.float64)
        return self._read_state(contact=False, internal_reset=False)

    def step(self, vx: float, vy: float, yaw_rate: float) -> RobotState:
        torch = self._torch
        cmd = torch.tensor([vx, vy, yaw_rate], device=self.uenv.device, dtype=torch.float32)
        self._obs["policy"][:, self._cmd_slice] = cmd
        self.uenv.command_manager.get_command("base_velocity")[:] = cmd
        with torch.inference_mode():
            joint_actions = self._policy(self._obs)
        self._obs, _, terminated, time_outs, _ = self.env.step(joint_actions)
        internal_reset = bool(terminated.any().item() or time_outs.any().item())   # must never happen
        import isaaclab.envs.mdp as mdp

        contact = bool(mdp.illegal_contact(self.uenv, self.contact_threshold, self._contact_cfg).any().item())
        return self._read_state(contact=contact, internal_reset=internal_reset)

    def render(self) -> Optional[np.ndarray]:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def describe(self) -> dict:
        return {
            "backend": "isaac",
            "task_name": TASK_NAME,
            "dt": self.dt,
            "sim_dt": float(self.uenv.physics_dt),
            "decimation": int(self.uenv.cfg.decimation),
            "episode_length_s": self.episode_length_s,
            "obs_noise": bool(self.uenv.cfg.observations.policy.enable_corruption),
            "spawn_height": self.spawn_height,
            "contact_threshold": self.contact_threshold,
            "locomotion_checkpoint": self.locomotion_checkpoint,
            "locomotion_checkpoint_sha256": self.locomotion_checkpoint_sha256,
            "policy_obs_layout": self.policy_obs_layout,
            "velocity_command_slice": [self._cmd_slice.start, self._cmd_slice.stop],
            "env_origin": self._origin().tolist(),
        }
