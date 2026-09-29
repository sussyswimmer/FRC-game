"""PettingZoo parallel environment: several robots learn at once (e.g. 3v3 self-play).

    from rebuilt_sim.pz_env import RebuiltParallelEnv
    env = RebuiltParallelEnv()                                   # all six robots are agents
    env = RebuiltParallelEnv(learners=("blue_0", "blue_1", "blue_2"))  # red is scripted bots

Agents see the field from their own alliance's side, so one shared policy can drive both
colors. Each agent's reward is its alliance's score margin plus its own shaping terms.
"""

from __future__ import annotations

import functools
from dataclasses import replace

import numpy as np
from pettingzoo import ParallelEnv

from .env import OBSERVATION_SPACE, EnvConfig, MatchRunner, action_space

AGENTS = ("blue_0", "blue_1", "blue_2", "red_0", "red_1", "red_2")


class RebuiltParallelEnv(ParallelEnv):
    metadata = {"name": "rebuilt_v0", "render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config: EnvConfig | None = None, learners: tuple[str, ...] = AGENTS,
                 render_mode: str | None = None, **overrides) -> None:
        cfg = config or EnvConfig(learner_tiers=("strong", "mid", "elite"))
        if overrides:
            cfg = replace(cfg, **overrides)
        self.cfg = cfg
        unknown = set(learners) - set(AGENTS)
        if unknown:
            raise ValueError(f"unknown agents {sorted(unknown)}")
        self.possible_agents = [a for a in AGENTS if a in learners]
        self.index = {a: AGENTS.index(a) for a in self.possible_agents}
        self.agents: list[str] = []
        self.render_mode = render_mode
        self.runner: MatchRunner | None = None
        self._rng = np.random.default_rng()
        self._viewer = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return OBSERVATION_SPACE

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return action_space(self.cfg.action_mode)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self.runner = MatchRunner(self.cfg, [self.index[a] for a in self.agents])
        tiers = None
        if options and "tiers" in options:  # {"blue_0": "elite", ...}
            tiers = {self.index[a]: t for a, t in options["tiers"].items() if a in self.index}
        self.runner.reset(self._rng, tiers)
        obs = {a: self.runner.observe(self.index[a]) for a in self.agents}
        infos = {a: self.runner.info(self.index[a]) for a in self.agents}
        return obs, infos

    def step(self, actions: dict):
        run = self.runner
        run.step({self.index[a]: act for a, act in actions.items()})
        done = run.match.done
        obs = {a: run.observe(self.index[a]) for a in self.agents}
        rewards = {a: run.reward(self.index[a]) for a in self.agents}
        terms = {a: done for a in self.agents}
        truncs = {a: False for a in self.agents}
        infos = {a: run.info(self.index[a]) for a in self.agents}
        if done:
            self.agents = []
        if self.render_mode == "human":
            self.render()
        return obs, rewards, terms, truncs, infos

    @property
    def match(self):
        return self.runner.match

    def render(self):
        from .viewer import Viewer

        if self._viewer is None:
            self._viewer = Viewer(headless=self.render_mode == "rgb_array")
        return self._viewer.draw(self.runner.match, return_array=self.render_mode == "rgb_array")

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
