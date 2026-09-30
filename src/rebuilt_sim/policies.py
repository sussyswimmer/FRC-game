"""Neural-network policies and a loader for trained models (needs the ``train`` extra: torch).

``ActorCritic`` is the network ``scripts/train_selfplay.py`` trains. ``load_policy`` turns
either a self-play checkpoint (.pt) or a Stable-Baselines3 model (.zip) into a function
``obs -> action`` that the watch/evaluate/play scripts can use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal


def _mlp(inp: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, discrete: bool, hidden: int = 256) -> None:
        super().__init__()
        self.discrete = discrete
        self.actor = _mlp(obs_dim, hidden, act_dim)
        self.critic = _mlp(obs_dim, hidden, 1)
        if not discrete:
            self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.meta = {"obs_dim": obs_dim, "act_dim": act_dim, "discrete": discrete, "hidden": hidden}

    def dist(self, obs: torch.Tensor):
        out = self.actor(obs)
        if self.discrete:
            return Categorical(logits=out)
        return Normal(out, self.log_std.exp().expand_as(out))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        d = self.dist(obs)
        if deterministic:
            a = d.probs.argmax(-1) if self.discrete else d.mean
        else:
            a = d.sample()
        logp = d.log_prob(a) if self.discrete else d.log_prob(a).sum(-1)
        return a, logp, self.value(obs)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        d = self.dist(obs)
        if self.discrete:
            return d.log_prob(actions), d.entropy(), self.value(obs)
        return d.log_prob(actions).sum(-1), d.entropy().sum(-1), self.value(obs)

    def save(self, path, extra: dict | None = None) -> None:
        torch.save({"state_dict": self.state_dict(), "meta": self.meta, "extra": extra or {}}, path)

    @classmethod
    def load(cls, path) -> "ActorCritic":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = cls(**ck["meta"])
        net.load_state_dict(ck["state_dict"])
        net.eval()
        return net


def _box_mode(act_dim: int) -> str:
    return "continuous_aim" if act_dim == 9 else "continuous"


def load_policy(path: str | Path, deterministic: bool = True):
    """Return (policy_fn, action_mode) for a .pt self-play checkpoint or an SB3 .zip model."""
    path = Path(path)
    if path.suffix == ".zip":
        from stable_baselines3 import PPO

        model = PPO.load(path, device="cpu")
        mode = "macro" if hasattr(model.action_space, "n") else _box_mode(model.action_space.shape[0])

        def fn(obs: np.ndarray):
            action, _ = model.predict(obs, deterministic=deterministic)
            return action
        return fn, mode
    net = ActorCritic.load(path)
    mode = "macro" if net.discrete else _box_mode(net.meta["act_dim"])

    def fn(obs: np.ndarray):
        a, _, _ = net.act(torch.as_tensor(obs, dtype=torch.float32), deterministic=deterministic)
        return a.numpy()
    return fn, mode
