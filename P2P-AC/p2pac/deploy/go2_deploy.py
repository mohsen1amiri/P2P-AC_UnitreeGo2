"""Deploy a trained policy on a physical Unitree Go2 (Sec. VI-H) with complete logging.

The learned policy runs on a remote computer and sends body-frame velocity commands ``[v_x, v_y]`` to the
robot's built-in locomotion controller (Unitree SDK2 ``SportClient.Move``) at 50 Hz. The planner is NOT used.

**Important differences to simulation** (state them in the paper; see docs/paper_vs_code.md):

* on hardware the low-level controller is Unitree's sport mode, not the Isaac Lab RSL-RL policy used in training;
* the observation ``[x, y, x_goal, y_goal]`` must be expressed in the *training arena frame*: the pose source is
  mapped through :class:`FrameCalibration` (origin and rotation of the motion-capture frame);
* the policy assumes the robot faces +y of the arena (yaw = 90 deg); an optional yaw-hold controller
  (``--yaw-hold-gain``) keeps that heading (0 = disabled, i.e. yaw-rate command 0 as in simulation).

Pose sources: ``udp`` (generic; a motion-capture relay sends ``"t,x,y,yaw\\n"`` packets -- the relay for the
authors' OptiTrack/Motive setup is site specific and must be provided), ``unitree_odom`` (``rt/sportmodestate``,
the robot's onboard odometry), and ``kinematic`` (dry run against the kinematic
test model, no robot needed).

Every trial writes ``<out>/trial_<k>/log.csv`` (one row per control step: time stamps, raw and arena-frame pose,
observation, policy action, sent command, measured velocity) and ``summary.json`` (outcome, duration, path
length, path ratio, mean speed, min clearance). ``python -m p2pac.deploy.hardware_analysis`` aggregates them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from p2pac.task import NavigationTask, Outcome, TaskConfig


@dataclass
class Pose:
    t: float
    x: float
    y: float
    yaw: float
    vx: float = math.nan     # measured body velocity if the source provides it
    vy: float = math.nan


@dataclass
class FrameCalibration:
    """Map raw source coordinates into the arena frame: ``p_arena = R(theta) (p_raw - origin)``."""

    origin_x: float = 0.0
    origin_y: float = 0.0
    rotation_deg: float = 0.0

    def apply(self, p: Pose) -> Pose:
        th = math.radians(self.rotation_deg)
        dx, dy = p.x - self.origin_x, p.y - self.origin_y
        x = math.cos(th) * dx - math.sin(th) * dy
        y = math.sin(th) * dx + math.cos(th) * dy
        return Pose(p.t, x, y, p.yaw + th, p.vx, p.vy)


class UdpPoseSource:
    """Receives ``"t,x,y,yaw"`` (ASCII, one packet per pose) on a UDP port; returns the newest pose."""

    def __init__(self, port: int, host: str = "0.0.0.0"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self._last: Optional[Pose] = None

    def read(self) -> Optional[Pose]:
        while True:
            try:
                data, _ = self.sock.recvfrom(256)
            except BlockingIOError:
                break
            t, x, y, yaw = (float(v) for v in data.decode().strip().split(",")[:4])
            self._last = Pose(t, x, y, yaw)
        return self._last


class UnitreeRobot:
    """Unitree SDK2 sport-mode client (commands) and ``rt/sportmodestate`` subscriber (odometry)."""

    def __init__(self, iface: str = ""):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

        ChannelFactoryInitialize(0, iface) if iface else ChannelFactoryInitialize(0)
        self.client = SportClient()
        self.client.SetTimeout(10.0)
        self.client.Init()
        self._state = None
        self.sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.sub.Init(self._cb, 10)

    def _cb(self, msg) -> None:
        self._state = msg

    def read(self) -> Optional[Pose]:
        s = self._state
        if s is None:
            return None
        return Pose(time.time(), float(s.position[0]), float(s.position[1]), float(s.imu_state.rpy[2]),
                    float(s.velocity[0]), float(s.velocity[1]))

    def stand_up(self) -> None:
        self.client.StandUp()

    def move(self, vx: float, vy: float, wz: float) -> None:
        self.client.Move(float(vx), float(vy), float(wz))

    def stop(self) -> None:
        self.client.StopMove()

    def damp(self) -> None:
        self.client.Damp()


class KinematicTwin:
    """Dry-run 'robot' + pose source backed by the kinematic test model."""

    def __init__(self, start: tuple[float, float], yaw: float, seed: int = 0):
        from p2pac.envs.kinematic import KinematicBackend

        self.b = KinematicBackend(seed=seed)
        self.st = self.b.reset_to(start[0], start[1], yaw)

    def read(self) -> Pose:
        return Pose(time.time(), float(self.st.xy[0]), float(self.st.xy[1]), float(self.st.yaw), *map(float, self.st.vel_body))

    def stand_up(self) -> None:
        pass

    def move(self, vx: float, vy: float, wz: float) -> None:
        self.st = self.b.step(vx, vy, wz)

    def stop(self) -> None:
        pass

    def damp(self) -> None:
        pass


def load_policy(path: str):
    """SB3 model (.zip, any algorithm of this repository) or BC policy (.pt)."""
    if path.endswith(".pt"):
        from p2pac.algos.bc import load_bc_policy

        return load_bc_policy(path)
    from stable_baselines3 import PPO

    from p2pac.algos import P2PDDPG, P2PSAC, P2PTD3

    for cls in (P2PSAC, P2PTD3, P2PDDPG, PPO):
        try:
            return cls.load(path, device="cpu")
        except Exception:
            continue
    raise RuntimeError(f"could not load policy {path}")


FIELDS = ["k", "wall_time", "pose_time", "raw_x", "raw_y", "raw_yaw", "x", "y", "yaw", "obs_x", "obs_y", "obs_gx", "obs_gy",
          "action_vx", "action_vy", "cmd_vx", "cmd_vy", "cmd_wz", "meas_vx", "meas_vy", "dist_to_goal", "clearance", "loop_dt"]


def run_trial(args, task: NavigationTask, policy, robot, pose_source, calib: FrameCalibration, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    dt = 1.0 / args.hz
    g = task.cfg.goal
    f = open(os.path.join(out_dir, "log.csv"), "w", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    robot.stand_up()
    time.sleep(args.settle_s)
    t0 = time.time()
    path, last_xy, start_xy, min_clear = 0.0, None, None, math.inf
    outcome, k = Outcome.TIMEOUT, 0
    try:
        while k < args.max_steps:
            tick = time.time()
            raw = pose_source.read()
            if raw is None:
                robot.stop()
                time.sleep(dt)
                continue
            p = calib.apply(raw)
            xy = np.array([p.x, p.y])
            if start_xy is None:
                start_xy = xy.copy()
            if last_xy is not None:
                path += float(np.linalg.norm(xy - last_xy))
            last_xy = xy
            clear = task.clearance(xy)
            min_clear = min(min_clear, clear)
            if task.distance_to_goal(xy) < task.cfg.goal_tolerance:
                outcome = Outcome.GOAL
                break
            if task.in_obstacle(xy) or task.out_of_bounds(xy):
                outcome = Outcome.COLLISION_OBSTACLE if task.in_obstacle(xy) else Outcome.OUT_OF_BOUNDS
                break
            obs = np.array([p.x, p.y, g[0], g[1]], dtype=np.float32)
            action, _ = policy.predict(obs, deterministic=True)
            action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:2], -args.v_max, args.v_max)
            wz = 0.0
            if args.yaw_hold_gain > 0:
                err = (task.cfg.spawn_yaw - p.yaw + math.pi) % (2 * math.pi) - math.pi
                wz = float(np.clip(args.yaw_hold_gain * err, -args.wz_max, args.wz_max))
            robot.move(action[0], action[1], wz)
            w.writerow({"k": k, "wall_time": tick - t0, "pose_time": raw.t, "raw_x": raw.x, "raw_y": raw.y, "raw_yaw": raw.yaw,
                        "x": p.x, "y": p.y, "yaw": p.yaw, "obs_x": obs[0], "obs_y": obs[1], "obs_gx": obs[2], "obs_gy": obs[3],
                        "action_vx": action[0], "action_vy": action[1], "cmd_vx": action[0], "cmd_vy": action[1], "cmd_wz": wz,
                        "meas_vx": raw.vx, "meas_vy": raw.vy, "dist_to_goal": task.distance_to_goal(xy), "clearance": clear,
                        "loop_dt": time.time() - tick})
            k += 1
            time.sleep(max(0.0, dt - (time.time() - tick)))
    finally:
        robot.stop()
        f.close()
    duration = time.time() - t0
    direct = float(np.linalg.norm(g - start_xy)) if start_xy is not None else math.nan
    summary = {"outcome": outcome.name.lower(), "success": int(outcome == Outcome.GOAL), "steps": k, "duration_s": duration,
               "path_length": path, "path_ratio": path / direct if direct and direct > 0 else math.nan,
               "mean_speed": path / duration if duration > 0 else math.nan, "min_clearance": min_clear,
               "start_x": float(start_xy[0]) if start_xy is not None else math.nan,
               "start_y": float(start_xy[1]) if start_xy is not None else math.nan}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="final_model.zip / policy snapshot / bc_policy.pt")
    ap.add_argument("--run-config", default=None, help="config.json of the training run (task geometry); default: paper task")
    ap.add_argument("--out", required=True, help="output directory (one sub-directory per trial)")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--pose", choices=("udp", "unitree_odom", "kinematic"), default="udp")
    ap.add_argument("--udp-port", type=int, default=9870)
    ap.add_argument("--iface", default="", help="network interface to the robot (Unitree SDK2)")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--v-max", type=float, default=0.7)
    ap.add_argument("--wz-max", type=float, default=0.6)
    ap.add_argument("--yaw-hold-gain", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--settle-s", type=float, default=1.0)
    ap.add_argument("--origin", type=float, nargs=2, default=(0.0, 0.0), help="arena origin in the pose-source frame")
    ap.add_argument("--rotation-deg", type=float, default=0.0)
    ap.add_argument("--dry-run-start", type=float, nargs=2, default=(0.0, -1.5), help="start for --pose kinematic")
    args = ap.parse_args(argv)

    cfg = TaskConfig()
    if args.run_config:
        from p2pac.config import DEFAULTS, task_config

        with open(args.run_config) as fh:
            d = dict(DEFAULTS)
            d.update(json.load(fh))
        cfg = task_config(argparse.Namespace(**d))
    task = NavigationTask(cfg)
    policy = load_policy(args.policy)
    calib = FrameCalibration(args.origin[0], args.origin[1], args.rotation_deg)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "deployment.json"), "w") as fh:
        json.dump({"args": vars(args), "calibration": asdict(calib), "task": cfg.to_dict()}, fh, indent=2, default=str)
    existing = len([d for d in os.listdir(args.out) if d.startswith("trial_")])
    for i in range(args.trials):
        if args.pose == "kinematic":
            robot = pose_source = KinematicTwin(tuple(args.dry_run_start), cfg.spawn_yaw, seed=i)
            args.settle_s = 0.0
        else:
            robot = UnitreeRobot(args.iface)
            pose_source = UdpPoseSource(args.udp_port) if args.pose == "udp" else robot
        s = run_trial(args, task, policy, robot, pose_source, calib, os.path.join(args.out, f"trial_{existing + i:03d}"))
        print(f"[deploy] trial {existing + i}: {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
