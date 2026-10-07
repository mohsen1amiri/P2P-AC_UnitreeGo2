"""Behaviour-cloning baseline (Sec. VI-B): regress the planner actions on the data of the planner phase.

The data set is the planner buffer ``D_P`` exported by a P2P-AC run at ``t = T_p``
(``artifacts/planner_buffer.npz``), i.e. exactly the ``T_p`` planner transitions the P2P-AC learner saw. The
policy has the actor architecture of the AC learners (MLP + tanh squashing to the action box) and is deployed
without any RL. Model selection uses a held-out validation split of the regression loss (never the task
success), so the baseline is not tuned on the evaluation metric.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch as th
from torch import nn

from p2pac.algos import ACTIVATIONS


@dataclass
class BCConfig:
    net_arch: tuple = (256, 256)
    activation: str = "relu"
    learning_rate: float = 3e-4
    batch_size: int = 256
    epochs: int = 50
    val_fraction: float = 0.1
    loss: str = "action"           # "action" (MSE on normalised actions) | "logit" (Eq. (19))
    logit_clip_eps: float = 1e-4
    seed: int = 0


class BCPolicy(nn.Module):
    """Deterministic policy ``u = u_c + D_u tanh(xi_theta(s))``."""

    def __init__(self, obs_dim: int, act_dim: int, net_arch, activation: str, action_low, action_high):
        super().__init__()
        layers, d = [], obs_dim
        for h in net_arch:
            layers += [nn.Linear(d, h), ACTIVATIONS[activation]()]
            d = h
        layers.append(nn.Linear(d, act_dim))
        self.net = nn.Sequential(*layers)
        self.register_buffer("low", th.as_tensor(np.asarray(action_low, dtype=np.float32)))
        self.register_buffer("high", th.as_tensor(np.asarray(action_high, dtype=np.float32)))

    def logits(self, obs: th.Tensor) -> th.Tensor:
        return self.net(obs)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Normalised action in [-1, 1]."""
        return th.tanh(self.logits(obs))

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """SB3-like interface used by the evaluation protocol (returns env-unit actions)."""
        with th.no_grad():
            o = th.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(-1, self.net[0].in_features)
            a = self.forward(o)
            u = self.low + 0.5 * (a + 1.0) * (self.high - self.low)
        out = u.cpu().numpy()
        return (out[0] if np.asarray(obs).ndim == 1 else out), None


def load_planner_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Observations and normalised planner actions of the labelled planner transitions."""
    d = np.load(path)
    obs = d["observations"].reshape(-1, d["observations"].shape[-1])
    act = d["planner_actions"].reshape(-1, d["planner_actions"].shape[-1])
    mask = d["planner_mask"].reshape(-1) > 0
    return obs[mask].astype(np.float32), act[mask].astype(np.float32)


def train_bc(obs: np.ndarray, act_norm: np.ndarray, action_low, action_high, cfg: BCConfig, out_dir: str, log=print) -> BCPolicy:
    """Train the BC policy; returns the parameters with the lowest validation loss."""
    rng = np.random.default_rng(cfg.seed)
    th.manual_seed(cfg.seed)
    n = obs.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(round(cfg.val_fraction * n)))
    val, tr = perm[:n_val], perm[n_val:]
    model = BCPolicy(obs.shape[1], act_norm.shape[1], cfg.net_arch, cfg.activation, action_low, action_high)
    opt = th.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    o_t, a_t = th.as_tensor(obs), th.as_tensor(act_norm)
    eps = cfg.logit_clip_eps

    def loss_fn(idx):
        if cfg.loss == "logit":
            tgt = th.atanh(a_t[idx].clamp(-(1 - eps), 1 - eps))
            return ((model.logits(o_t[idx]) - tgt) ** 2).mean()
        return ((model(o_t[idx]) - a_t[idx]) ** 2).mean()

    best, best_state, rows = np.inf, None, []
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(tr)
        tr_losses = []
        for i in range(0, len(order), cfg.batch_size):
            idx = th.as_tensor(order[i:i + cfg.batch_size])
            loss = loss_fn(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())
        model.eval()
        with th.no_grad():
            v = float(loss_fn(th.as_tensor(val)).item())
        rows.append({"epoch": epoch, "train_loss": float(np.mean(tr_losses)), "val_loss": v})
        if v < best:
            best, best_state = v, {k: x.clone() for k, x in model.state_dict().items()}
        log(f"[bc] epoch {epoch:3d} train {np.mean(tr_losses):.5f} val {v:.5f}")
    model.load_state_dict(best_state)
    os.makedirs(out_dir, exist_ok=True)
    th.save({"state_dict": model.state_dict(), "config": asdict(cfg), "obs_dim": obs.shape[1], "act_dim": act_norm.shape[1],
             "action_low": list(map(float, action_low)), "action_high": list(map(float, action_high))},
            os.path.join(out_dir, "bc_policy.pt"))
    with open(os.path.join(out_dir, "bc_training.json"), "w") as f:
        json.dump({"history": rows, "best_val_loss": best, "n_train": int(len(tr)), "n_val": int(n_val)}, f, indent=2)
    return model


def load_bc_policy(path: str) -> BCPolicy:
    ckpt = th.load(path, map_location="cpu", weights_only=False)
    c = ckpt["config"]
    model = BCPolicy(ckpt["obs_dim"], ckpt["act_dim"], c["net_arch"], c["activation"], ckpt["action_low"], ckpt["action_high"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
