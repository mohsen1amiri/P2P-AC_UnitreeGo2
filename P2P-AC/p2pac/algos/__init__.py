"""Learning algorithms: P2P-AC instances, plain actor-critic baselines, Accelerated SAC, PPO and BC."""

from __future__ import annotations

from typing import Any

import torch as th

from p2pac.algos.buffers import AnnotatedReplayBuffer, PlannerReplayBuffer
from p2pac.algos.p2p import ALGORITHM_CLASSES, P2PDDPG, P2PSAC, P2PTD3, PlannerToPolicyMixin
from p2pac.algos.schedule import P2PConfig

ACTIVATIONS = {"relu": th.nn.ReLU, "tanh": th.nn.Tanh, "elu": th.nn.ELU}


def p2p_config_from_args(args) -> P2PConfig:
    """Translate the resolved command-line configuration into a :class:`P2PConfig`."""
    if args.method == "base":
        return P2PConfig.disabled()
    # Label noise (Accel. SAC baseline) is given in physical units (m/s); the standard deviation in the normalised
    # action space is noise / D_u (identical for both axes when vx_max = vy_max).
    half_range = min(args.vx_max, args.vy_max)
    label_noise = args.label_noise_std / half_range if args.label_noise_std > 0 else 0.0
    return P2PConfig(
        t_p=args.t_p,
        t_d=args.t_d,
        kappa0=args.kappa0,
        execute_planner=args.execute_planner,
        imitation_space=args.imitation_space,
        logit_clip_eps=args.logit_clip_eps,
        label_noise_std=label_noise,
    )


def policy_kwargs_from_args(args, on_policy: bool = False) -> dict[str, Any]:
    """Network / optimiser settings shared by all learners (Sec. VI-B: two hidden layers of 256 ReLU units, Adam).

    ``adam_eps=None`` keeps the SB3 default of the algorithm (1e-8, PyTorch's default, for SAC/TD3/DDPG; 1e-5 for
    PPO). ``optimizer_kwargs`` must then not be passed at all, because SB3
    only applies PPO's ``eps=1e-5`` when ``optimizer_kwargs is None``.
    """
    arch = list(args.net_arch)
    net_arch = dict(pi=arch, vf=arch) if on_policy else dict(pi=arch, qf=arch)
    kwargs: dict[str, Any] = dict(net_arch=net_arch, activation_fn=ACTIVATIONS[args.activation], optimizer_class=th.optim.Adam)
    if args.adam_eps is not None:
        kwargs["optimizer_kwargs"] = dict(eps=args.adam_eps)
    return kwargs


def build_model(args, vec_env, device: str, seed: int, tensorboard_log: str | None = None):
    """Create a fresh model for ``args.algo`` / ``args.method``."""
    if args.algo == "ppo":
        from stable_baselines3 import PPO

        if args.method != "base":
            raise ValueError("PPO is only used as an on-policy baseline (method=base)")
        return PPO(
            "MlpPolicy", vec_env,
            learning_rate=args.learning_rate, n_steps=args.ppo_n_steps, batch_size=args.ppo_batch_size,
            n_epochs=args.ppo_n_epochs, gamma=args.gamma, gae_lambda=args.ppo_gae_lambda, clip_range=args.ppo_clip_range,
            ent_coef=args.ppo_ent_coef, vf_coef=args.ppo_vf_coef, max_grad_norm=args.ppo_max_grad_norm,
            policy_kwargs=policy_kwargs_from_args(args, on_policy=True),
            tensorboard_log=tensorboard_log, seed=seed, device=device, verbose=0,
        )

    cls = ALGORITHM_CLASSES[args.algo]
    p2p_cfg = p2p_config_from_args(args)
    if args.method != "base" and args.buffer == "dual":
        rb_cls, rb_kwargs = PlannerReplayBuffer, dict(planner_capacity=args.planner_buffer_capacity,
                                                     planner_fraction=args.planner_batch_fraction)
    else:
        rb_cls, rb_kwargs = AnnotatedReplayBuffer, {}
    common = dict(
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        replay_buffer_class=rb_cls,
        replay_buffer_kwargs=rb_kwargs,
        policy_kwargs=policy_kwargs_from_args(args),
        tensorboard_log=tensorboard_log,
        seed=seed,
        device=device,
        verbose=0,
        p2p_config=p2p_cfg,
    )
    if args.algo == "sac":
        target_entropy = "auto" if args.target_entropy is None else float(args.target_entropy)
        from p2pac.config import parse_ent_coef

        return P2PSAC("MlpPolicy", vec_env, ent_coef=parse_ent_coef(args.ent_coef), target_entropy=target_entropy, **common)

    import numpy as np
    from stable_baselines3.common.noise import NormalActionNoise

    n_act = vec_env.action_space.shape[0]
    # exploration noise in the *normalised* action space (SB3 adds it to the scaled action)
    noise = NormalActionNoise(mean=np.zeros(n_act), sigma=args.exploration_noise * np.ones(n_act)) if args.exploration_noise > 0 else None
    if args.algo == "td3":
        return P2PTD3("MlpPolicy", vec_env, action_noise=noise, policy_delay=args.td3_policy_delay,
                      target_policy_noise=args.td3_target_noise, target_noise_clip=args.td3_target_noise_clip, **common)
    if args.algo == "ddpg":
        return P2PDDPG("MlpPolicy", vec_env, action_noise=noise, **common)
    raise ValueError(f"unknown algorithm {args.algo!r}")


def model_class(algo: str):
    if algo == "ppo":
        from stable_baselines3 import PPO

        return PPO
    return ALGORITHM_CLASSES[algo]


__all__ = [
    "ACTIVATIONS", "AnnotatedReplayBuffer", "P2PConfig", "P2PDDPG", "P2PSAC", "P2PTD3", "PlannerReplayBuffer",
    "PlannerToPolicyMixin", "build_model", "model_class", "p2p_config_from_args", "policy_kwargs_from_args",
]
