"""Play a match with trained policies in some seats and scripted bots in the rest.

Used by the watch, play and evaluate scripts. A policy is any ``obs -> action`` function
(see ``rebuilt_sim.policies.load_policy``); it decides every ``decision_dt`` seconds and
its last action is held in between, exactly like during training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .env import EnvConfig, MatchRunner

AGENTS = ("blue_0", "blue_1", "blue_2", "red_0", "red_1", "red_2")


def run_config_for(model_path: str | Path, mode: str) -> EnvConfig:
    """EnvConfig matching how a model was trained (reads config.json from its run folder)."""
    path = Path(model_path).resolve()
    for folder in (path.parent, path.parent.parent):
        cfg_file = folder / "config.json"
        if cfg_file.exists():
            env = json.loads(cfg_file.read_text()).get("env", {})
            return EnvConfig(action_mode=mode, decision_dt=float(env.get("decision_dt", 0.1)))
    return EnvConfig(action_mode=mode, decision_dt=0.25 if mode == "macro" else 0.1)


class PolicyMatch:
    """One match: ``policies`` maps seat index (0-5) -> obs->action function; every other seat
    is a scripted bot. ``tiers`` fixes robot types by seat; the rest are drawn at random.

    The same ``seed`` and ``tiers`` give the same robots, start positions and random streams,
    so a seat played by a policy and the same seat played by its scripted bot are compared on
    like-for-like matches.
    """

    def __init__(self, cfg: EnvConfig, policies: dict, tiers: dict[int, str] | None = None,
                 seed: int | None = None) -> None:
        self.cfg = cfg
        self.policies = dict(policies)
        self.runner = MatchRunner(cfg, sorted(self.policies))
        self.runner.reset(np.random.default_rng(seed), tiers)
        self.match = self.runner.match
        self.actions: dict[int, object] = {}
        self.next_decision = 0.0

    def step(self) -> None:
        m = self.match
        if m.t >= self.next_decision - 1e-9:
            for i, fn in self.policies.items():
                self.actions[i] = fn(self.runner.observe(i))
            self.next_decision += self.cfg.decision_dt
        cmds = self.runner.bots.commands()
        for i, a in self.actions.items():
            cmds[i] = self.runner.to_command(i, a)
        m.step(cmds)

    def run(self) -> None:
        while not self.match.done:
            self.step()
