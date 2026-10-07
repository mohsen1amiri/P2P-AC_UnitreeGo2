"""P2P-AC: Planner-to-Policy Actor-Critic (Sec. V, Algorithm 1) on top of Stable-Baselines3 2.7.1.

:class:`PlannerToPolicyMixin` adds to an SB3 off-policy algorithm

* behaviour substitution, Eq. (16): for ``t < T_p`` the executed action is the planner action ``u^dagger_t``;
* planner queries for ``t < T_r`` only (``iota_t`` records availability), with one planner instance per
  parallel environment, reset at the beginning of every episode;
* storage of ``(u^dagger_t, iota_t)`` with every transition, in the normalised action space;
* the imitation term ``kappa_t E[iota * l(s, u^dagger)]`` of Eq. (20) in the actor loss.

:class:`P2PSAC`, :class:`P2PTD3` and :class:`P2PDDPG` re-implement ``train()`` of SB3 2.7.1 *line by line*
(same order of updates, same losses, same RNG consumption) and add only the imitation term. With the planner
disabled (``P2PConfig.disabled()``) they are bitwise identical to SB3's SAC/TD3/DDPG, which is tested in
``tests/test_algorithm_equivalence.py``; for ``t >= T_r`` (``kappa_t = 0``) the update is therefore exactly the
underlying algorithm (Proposition 3).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Optional

import numpy as np
import torch as th
from stable_baselines3 import DDPG, SAC, TD3
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F

from p2pac.algos.buffers import AnnotatedSamples, PendingExtras
from p2pac.algos.schedule import P2PConfig


class StatAccumulator:
    """Accumulates scalar learner statistics between two flushes of the learner log."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._last: dict[str, float] = {}

    def add(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            if v is None:
                continue
            v = float(v)
            if math.isnan(v):
                continue
            self._sums[k] += v
            self._counts[k] += 1

    def set_last(self, **kwargs: float) -> None:
        self._last.update({k: float(v) for k, v in kwargs.items()})

    def pop(self) -> dict[str, float]:
        out = {k: self._sums[k] / self._counts[k] for k in self._sums if self._counts[k] > 0}
        out.update(self._last)
        out["n_updates_in_window"] = max(self._counts.values()) if self._counts else 0
        self._sums.clear()
        self._counts.clear()
        return out


class PlannerToPolicyMixin:
    """Behaviour substitution, planner bookkeeping and the imitation term (Algorithm 1)."""

    # attributes created by _p2p_init; runtime-only ones are excluded from SB3's pickle
    p2p: P2PConfig

    _RUNTIME_ATTRS = ("_planners", "_action_map", "_pending_extras", "step_record", "learner_stats")

    def _p2p_init(self, p2p_config: Optional[P2PConfig]) -> None:
        self.p2p = p2p_config if p2p_config is not None else P2PConfig.disabled()
        self._planners: Optional[list] = None
        self._action_map: Optional[Callable] = None
        self._pending_extras: Optional[PendingExtras] = None
        self.step_record: dict[str, Any] = {}
        self.learner_stats = StatAccumulator()
        self.n_planner_queries = 0
        self.n_planner_queries_after_removal = 0  # must stay 0 (Proposition 3 (i))
        self.n_planner_executed_steps = 0

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + list(self._RUNTIME_ATTRS)  # type: ignore[misc]

    # ------------------------------------------------------------------------------------------
    # planner management
    # ------------------------------------------------------------------------------------------
    def attach_planner(self, planner_factory: Optional[Callable[[], Any]], action_map: Optional[Callable]) -> None:
        """Create one planner per parallel environment (planners are runtime objects, never pickled)."""
        if planner_factory is None:
            if self.p2p.uses_planner:
                raise ValueError("this P2P configuration needs a planner but none was attached")
            self._planners, self._action_map = None, None
            return
        n_envs = self.n_envs  # type: ignore[attr-defined]
        self._planners = [planner_factory() for _ in range(n_envs)]
        self._action_map = action_map

    def reset_planners(self, env_indices: Optional[list[int]] = None) -> None:
        if self._planners is None:
            return
        for i in (range(len(self._planners)) if env_indices is None else env_indices):
            self._planners[i].reset()

    @property
    def has_planner(self) -> bool:
        return self._planners is not None

    # ------------------------------------------------------------------------------------------
    # data collection (Algorithm 1, lines 3-6)
    # ------------------------------------------------------------------------------------------
    def _sample_action(self, learning_starts: int, action_noise=None, n_envs: int = 1):
        # Line 3: behaviour action of the AC algorithm (computed first so that the RNG stream of the
        # underlying algorithm is unchanged).
        action, buffer_action = super()._sample_action(learning_starts, action_noise, n_envs)  # type: ignore[misc]
        t = int(self.num_timesteps)  # type: ignore[attr-defined]
        p = action.shape[-1]
        planner_scaled = np.zeros((n_envs, p), dtype=np.float32)
        iota = np.zeros(n_envs, dtype=np.float32)
        executed = np.zeros(n_envs, dtype=np.int8)
        rec: dict[str, Any] = {
            "t": t,
            "behavior_action": np.array(action, dtype=np.float32, copy=True),
            "planner_action": np.full((n_envs, p), np.nan, dtype=np.float32),
            "planner_iterations": np.zeros(n_envs, dtype=np.int32),
            "planner_ms": np.full(n_envs, np.nan, dtype=np.float32),
            "planner_max_constraint": np.full(n_envs, np.nan, dtype=np.float32),
            "planner_outside_domain": np.zeros(n_envs, dtype=np.int8),
            "planner_action_clipped": np.zeros(n_envs, dtype=np.int8),
            "kappa": self.p2p.kappa(t),
            "phase": self.p2p.phase(t),
        }
        # Line 4: query the planner while it is needed (t < T_r in the paper).
        if self.p2p.queries_planner(t):
            if self._planners is None:
                raise RuntimeError("planner required at t=%d but not attached" % t)
            if t >= self.p2p.t_r:
                self.n_planner_queries_after_removal += 1
            obs = self._last_obs  # type: ignore[attr-defined]
            u_dag = np.zeros((n_envs, p), dtype=np.float64)
            for i in range(n_envs):
                out = self._planners[i].plan(np.asarray(obs[i, :2], dtype=np.float64))   # z_t = h_z(s_t)
                u_i, clipped = self._action_map(out)
                u_dag[i] = u_i
                rec["planner_iterations"][i] = out.iterations
                rec["planner_ms"][i] = 1e3 * out.wall_time_s
                rec["planner_max_constraint"][i] = out.max_constraint
                rec["planner_outside_domain"][i] = int(out.outside_domain)
                rec["planner_action_clipped"][i] = int(clipped)
            self.n_planner_queries += n_envs
            planner_scaled = self.policy.scale_action(u_dag).astype(np.float32)  # type: ignore[attr-defined]
            iota[:] = 1.0
            rec["planner_action"] = u_dag.astype(np.float32)
            # Line 5 / Eq. (16): behaviour substitution during the planner phase.
            if self.p2p.executes_planner(t):
                action = u_dag.astype(action.dtype)
                buffer_action = planner_scaled.copy()
                executed[:] = 1
                self.n_planner_executed_steps += n_envs
        rec["executed_by_planner"] = executed.copy()
        rec["executed_action"] = np.array(action, dtype=np.float32, copy=True)
        self.step_record = rec
        self._pending_extras = PendingExtras(
            planner_actions=planner_scaled,
            planner_mask=iota,
            executed_by_planner=executed,
            step_index=t,
            to_planner_buffer=self.p2p.routes_to_planner_buffer(t),
        )
        return action, buffer_action

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos) -> None:
        # Line 6: store the tuple of Sec. V-B (D_P if t < T_p, else D_O).
        if self._pending_extras is not None and hasattr(replay_buffer, "set_pending"):
            replay_buffer.set_pending(self._pending_extras)
        self._pending_extras = None
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)  # type: ignore[misc]
        done_idx = [i for i, d in enumerate(dones) if d]
        if done_idx:
            self.reset_planners(done_idx)

    def _setup_learn(self, *args, **kwargs):
        out = super()._setup_learn(*args, **kwargs)  # type: ignore[misc]
        # A new episode starts whenever learn() resets the environment.
        self.reset_planners()
        return out

    # ------------------------------------------------------------------------------------------
    # imitation term (Sec. V-C)
    # ------------------------------------------------------------------------------------------
    def _imitation_loss(self, data: AnnotatedSamples, pre_activation: th.Tensor, sampled_action: Optional[th.Tensor]) -> th.Tensor:
        """Per-sample imitation loss ``l(s, u^dagger)``, shape (B, 1).

        * ``logit``: ``(1/p) ||xi_theta(s) - xi^dagger||^2`` with ``xi^dagger`` of Eq. (18) (paper);
        * ``action``: ``(1/p) ||tanh(xi_theta(s)) - D_u^{-1}(u^dagger - u_c)||^2`` (ablation "imitation on actions");
        * ``action_sample``: ``(1/p) ||u_tilde - D_u^{-1}(u^dagger - u_c) - noise||^2`` with the sampled action
          ``u_tilde ~ pi_theta(.|s)`` (the Accelerated SAC baseline).
        """
        cfg = self.p2p
        target = data.planner_actions
        if cfg.label_noise_std > 0.0:
            target = (target + th.randn_like(target) * cfg.label_noise_std).clamp(-1.0, 1.0)
        if cfg.imitation_space == "logit":
            eps = cfg.logit_clip_eps
            xi_dag = th.atanh(target.clamp(-(1.0 - eps), 1.0 - eps))
            return ((pre_activation - xi_dag) ** 2).mean(dim=1, keepdim=True)
        if cfg.imitation_space == "action":
            return ((th.tanh(pre_activation) - target) ** 2).mean(dim=1, keepdim=True)
        assert sampled_action is not None
        return ((sampled_action - target) ** 2).mean(dim=1, keepdim=True)

    def _batch_stats(self, data: AnnotatedSamples, current_q: th.Tensor) -> dict[str, float]:
        with th.no_grad():
            fp = data.from_planner_buffer
            n_p = float(fp.sum().item())
            n = float(fp.shape[0])
            goal_p = float((data.goal * fp).sum().item())
            q = current_q.detach()
            stats = {
                "batch_frac_planner_buffer": n_p / n,
                "batch_frac_iota": float(data.planner_mask.mean().item()),
                "batch_goal_rows_from_planner_buffer": goal_p,
                "batch_has_goal_from_planner_buffer": float(goal_p > 0),
                "batch_goal_rows": float(data.goal.sum().item()),
                "q_mean": float(q.mean().item()),
            }
            if n_p > 0:
                stats["q_mean_planner_rows"] = float((q * fp).sum().item() / n_p)
            if n_p < n:
                stats["q_mean_online_rows"] = float((q * (1 - fp)).sum().item() / (n - n_p))
        return stats


# =================================================================================================
# SAC
# =================================================================================================
class P2PSAC(PlannerToPolicyMixin, SAC):
    """P2P-SAC. ``train`` is SB3 2.7.1 ``SAC.train`` plus the imitation term (and diagnostics)."""

    def __init__(self, *args, p2p_config: Optional[P2PConfig] = None, **kwargs):
        self._p2p_init(p2p_config)
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        t = int(self.num_timesteps)
        kappa = self.p2p.kappa(t)

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # ---- P2P-AC imitation term, Eq. (20): kappa_t * E[iota * l(s, u^dagger)] ------------------
            extra: dict[str, float] = {}
            if kappa > 0.0 and isinstance(replay_data, AnnotatedSamples) and float(replay_data.planner_mask.sum()) > 0:
                pre, _, _ = self.actor.get_action_dist_params(replay_data.observations)   # xi_theta(s)
                ell = self._imitation_loss(replay_data, pre, actions_pi)
                imitation = (replay_data.planner_mask * ell).mean()
                actor_loss = actor_loss + kappa * imitation
                mask = replay_data.planner_mask
                extra["imitation_loss"] = float(imitation.item())
                extra["imitation_loss_on_labelled"] = float((ell * mask).sum().item() / max(float(mask.sum().item()), 1.0))
                extra["logit_abs_mean"] = float(pre.detach().abs().mean().item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            if isinstance(replay_data, AnnotatedSamples):
                extra.update(self._batch_stats(replay_data, current_q_values[0]))
            self.learner_stats.add(
                critic_loss=critic_losses[-1], actor_loss_ac=actor_losses[-1], actor_loss_total=float(actor_loss.item()),
                ent_coef=ent_coefs[-1], ent_coef_loss=ent_coef_losses[-1] if ent_coef_losses else None,
                entropy=float(-log_prob.detach().mean().item()), **extra,
            )

        self._n_updates += gradient_steps
        self.learner_stats.set_last(kappa=kappa, phase=self.p2p.phase(t), n_updates=self._n_updates, timesteps=t)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        self.logger.record("p2p/kappa", kappa)


# =================================================================================================
# TD3 / DDPG
# =================================================================================================
def _td3_train(self, gradient_steps: int, batch_size: int) -> None:
    """SB3 2.7.1 ``TD3.train`` plus the imitation term in the (delayed) actor update."""
    self.policy.set_training_mode(True)
    self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
    t = int(self.num_timesteps)
    kappa = self.p2p.kappa(t)

    actor_losses, critic_losses = [], []
    for _ in range(gradient_steps):
        self._n_updates += 1
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
        discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

        with th.no_grad():
            noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
            next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

        current_q_values = self.critic(replay_data.observations, replay_data.actions)
        critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        assert isinstance(critic_loss, th.Tensor)
        critic_losses.append(critic_loss.item())

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        extra: dict[str, float] = {}
        if self._n_updates % self.policy_delay == 0:
            use_imitation = kappa > 0.0 and isinstance(replay_data, AnnotatedSamples) and float(replay_data.planner_mask.sum()) > 0
            if use_imitation:
                # Split the actor forward pass to expose the pre-activation xi_theta(s) (mu[-1] is the tanh).
                features = self.actor.extract_features(replay_data.observations, self.actor.features_extractor)
                pre = self.actor.mu[:-1](features)
                pi_action = self.actor.mu[-1](pre)
            else:
                pi_action = self.actor(replay_data.observations)
            actor_loss = -self.critic.q1_forward(replay_data.observations, pi_action).mean()
            actor_losses.append(actor_loss.item())
            if use_imitation:
                ell = self._imitation_loss(replay_data, pre, pi_action)
                imitation = (replay_data.planner_mask * ell).mean()
                actor_loss = actor_loss + kappa * imitation
                mask = replay_data.planner_mask
                extra["imitation_loss"] = float(imitation.item())
                extra["imitation_loss_on_labelled"] = float((ell * mask).sum().item() / max(float(mask.sum().item()), 1.0))
                extra["logit_abs_mean"] = float(pre.detach().abs().mean().item())
            extra["actor_loss_ac"] = actor_losses[-1]
            extra["actor_loss_total"] = float(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
            polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
            polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        if isinstance(replay_data, AnnotatedSamples):
            extra.update(self._batch_stats(replay_data, current_q_values[0]))
        self.learner_stats.add(critic_loss=critic_losses[-1], **extra)

    self.learner_stats.set_last(kappa=kappa, phase=self.p2p.phase(t), n_updates=self._n_updates, timesteps=t)
    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    if len(actor_losses) > 0:
        self.logger.record("train/actor_loss", np.mean(actor_losses))
    self.logger.record("train/critic_loss", np.mean(critic_losses))
    self.logger.record("p2p/kappa", kappa)


class P2PTD3(PlannerToPolicyMixin, TD3):
    """P2P-TD3 (imitation applied at the delayed actor updates of TD3)."""

    def __init__(self, *args, p2p_config: Optional[P2PConfig] = None, **kwargs):
        self._p2p_init(p2p_config)
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        _td3_train(self, gradient_steps, batch_size)


class P2PDDPG(PlannerToPolicyMixin, DDPG):
    """P2P-DDPG (SB3's DDPG is TD3 with one critic, ``policy_delay = 1`` and no target smoothing)."""

    def __init__(self, *args, p2p_config: Optional[P2PConfig] = None, **kwargs):
        self._p2p_init(p2p_config)
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        _td3_train(self, gradient_steps, batch_size)


ALGORITHM_CLASSES = {"sac": P2PSAC, "td3": P2PTD3, "ddpg": P2PDDPG}
