"""Gymnasium environment: you control one robot, scripted bots drive the other five.

    import gymnasium as gym
    import rebuilt_sim.env  # registers the ids
    env = gym.make("Rebuilt-Strategy-v0")                       # 8 macro actions
    env = gym.make("Rebuilt-Control-v0")                        # continuous drive/intake/shoot/climb
    env = gym.make("Rebuilt-Aim-HiFi-v0")                       # high fidelity, you also set the shot

One episode is one full match (AUTO, pause, TELEOP, final grace) = 1660 decisions at 10 Hz.
Rewards and opponents are configurable through ``EnvConfig`` (see docs/02-simulator.md);
``EnvConfig(hifi=HiFiConfig())`` switches on the high-fidelity physics
(docs/03-driving-and-aiming.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .bots import ScriptedPolicy
from .constants import Alliance
from .controller import N_MACROS, Macro, MacroController
from .obs import OBS_SIZE, OBS_SIZE_HIFI, observe, vel_to_frame
from .robot import DAY1_TIER_WEIGHTS, TIERS, RobotCommand
from .sim import HiFiConfig, Match, MatchConfig

ACTION_MODES = ("macro", "continuous", "continuous_aim")


@dataclass(frozen=True)
class RewardConfig:
    score: float = 0.1  # per point of change in (own alliance score - opponent score)
    self_fuel: float = 0.0  # extra per FUEL this robot scored itself
    pickup: float = 0.005  # per FUEL this robot collected (shaping; set to 0 once it learns)
    wasted: float = -0.05  # per FUEL this robot put into a HUB that wasn't counting
    own_foul: float = 0.0  # extra per foul this robot committed (the opponent already gets the points)
    win: float = 1.0  # +win / -win at the end of the match (0 for a tie)
    ranking_point: float = 0.5  # per bonus ranking point (ENERGIZED, SUPERCHARGED, TRAVERSAL) earned


@dataclass(frozen=True)
class EnvConfig:
    action_mode: str = "macro"  # "macro" Discrete(8), "continuous" Box(6), "continuous_aim" Box(9) (high fidelity)
    decision_dt: float = 0.1  # seconds of match time per env step
    sim_dt: float = 0.05  # physics step (the high-fidelity physics uses hifi.dt)
    learner_tiers: tuple[str, ...] = ("strong",)  # robot the agent drives, sampled each episode
    learner_slots: tuple[int, ...] = (0, 1, 2, 3, 4, 5)  # 0-2 blue, 3-5 red, sampled each episode
    bot_tier_weights: dict = field(default_factory=lambda: dict(DAY1_TIER_WEIGHTS))
    event_level: str = "regional"
    random_starts: bool = True
    reward: RewardConfig = field(default_factory=RewardConfig)
    hifi: HiFiConfig | None = None  # high-fidelity physics for driving and aiming (docs/03-driving-and-aiming.md)

    def __post_init__(self) -> None:
        if self.action_mode not in ACTION_MODES:
            raise ValueError(f"unknown action_mode {self.action_mode!r}")
        if self.action_mode == "continuous_aim" and (self.hifi is None or not self.hifi.ballistics):
            raise ValueError("continuous_aim sets the shooter, so it needs the high-fidelity ballistics "
                             "(hifi=HiFiConfig())")


class MatchRunner:
    """Owns one match plus the scripted bots; applies learner actions and computes rewards."""

    def __init__(self, cfg: EnvConfig, learners: list[int]) -> None:
        self.cfg = cfg
        self.learners = list(learners)
        self.match: Match | None = None
        self._prev: dict[int, tuple] = {}
        self._finished: set[int] = set()

    def reset(self, rng: np.random.Generator, tiers: dict[int, str] | None = None) -> None:
        """Start a new match. ``tiers`` fixes the robot type of any seat (0-5); other bot seats are
        drawn from ``bot_tier_weights`` and other learner seats from ``learner_tiers``."""
        cfg = self.cfg
        names = list(cfg.bot_tier_weights)
        p = np.array([cfg.bot_tier_weights[n] for n in names], dtype=float)
        drawn = list(rng.choice(names, size=6, p=p / p.sum()))
        given = tiers or {}
        for i in range(6):
            if i in given:
                drawn[i] = given[i]
            elif i in self.learners:
                drawn[i] = str(rng.choice(cfg.learner_tiers))
        match_seed, bot_seed = (int(s) for s in rng.integers(2**31, size=2))
        self.match = Match([TIERS[t] for t in drawn],
                           MatchConfig(dt=cfg.sim_dt, event_level=cfg.event_level, random_starts=cfg.random_starts,
                                       hifi=cfg.hifi),
                           seed=match_seed)
        self.ctl = MacroController(self.match)
        bots = [i for i in range(6) if i not in self.learners]
        self.bots = ScriptedPolicy(self.match, robots=bots, controller=self.ctl, seed=bot_seed)
        self._prev = {i: self._snapshot(i) for i in self.learners}
        self._finished = set()

    # ---------------------------------------------------------------- stepping
    def step(self, actions: dict[int, object]) -> None:
        m = self.match
        n_sub = max(1, round(self.cfg.decision_dt / m.cfg.dt))
        for _ in range(n_sub):
            if m.done:
                break
            cmds = self.bots.commands()
            for i, a in actions.items():
                cmds[i] = self.to_command(i, a)
            m.step(cmds)

    def to_command(self, i: int, action) -> RobotCommand:
        r = self.match.robots[i]
        if self.cfg.action_mode == "macro":
            return self.ctl.command(r, Macro(int(action)))
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        vx, vy = vel_to_frame(r.alliance, a[0] * r.spec.max_speed, a[1] * r.spec.max_speed)
        cmd = RobotCommand(vx=float(vx), vy=float(vy), omega=float(a[2] * r.spec.max_omega),
                           intake=bool(a[3] > 0), shoot=bool(a[4] > 0),
                           climb=r.spec.climb_level if a[5] > 0 else 0)
        if self.cfg.action_mode == "continuous_aim":  # -1..1 spans each setting's range
            sh = r.spec.shooter
            cmd.shot_speed = float(_span(sh.speed_range, a[6]))
            cmd.hood = float(_span(sh.hood_range, a[7]))
            cmd.turret = float(a[8] * sh.turret_range) if r.spec.turret else 0.0
        return cmd

    def observe(self, i: int) -> np.ndarray:
        return observe(self.match, self.ctl, i)

    # ---------------------------------------------------------------- rewards
    def _snapshot(self, i: int) -> tuple:
        m, r = self.match, self.match.robots[i]
        diff = m.scores[r.alliance].total - m.scores[r.alliance.other].total
        st = r.stats
        return diff, st.fuel_scored, st.fuel_collected, st.fuel_wasted, st.fouls_minor + st.fouls_major

    def reward(self, i: int) -> float:
        w = self.cfg.reward
        now = self._snapshot(i)
        prev = self._prev[i]
        self._prev[i] = now
        d = [n - p for n, p in zip(now, prev)]
        rew = w.score * d[0] + w.self_fuel * d[1] + w.pickup * d[2] + w.wasted * d[3] + w.own_foul * d[4]
        if self.match.done and i not in self._finished:
            self._finished.add(i)
            a = self.match.robots[i].alliance
            own, opp = self.match.scores[a].total, self.match.scores[a.other].total
            rew += w.win * (1.0 if own > opp else -1.0 if own < opp else 0.0)
            rp = self.match.summary().rp[a]
            rew += w.ranking_point * (rp.energized + rp.supercharged + rp.traversal)
        return float(rew)

    def info(self, i: int) -> dict:
        m = self.match
        r = m.robots[i]
        a = r.alliance
        info = {"alliance": "blue" if a == Alliance.BLUE else "red", "tier": r.spec.name, "t": m.t}
        if m.done:
            s = m.summary()
            info.update({
                "own_score": s.scores[a].total, "opp_score": s.scores[a.other].total,
                "own_fuel": s.scores[a].fuel, "win": float(s.winner == a), "tie": float(s.winner is None),
                "ranking_points": s.rp[a].total, "robot_fuel_scored": r.stats.fuel_scored,
                "robot_fouls": r.stats.fouls_minor + r.stats.fouls_major, "tower_points": r.stats.tower_points,
            })
        return info


def _span(bounds: tuple[float, float], a: float) -> float:
    lo, hi = bounds
    return lo + (a + 1.0) * 0.5 * (hi - lo)


def action_space(mode: str) -> spaces.Space:
    if mode == "macro":
        return spaces.Discrete(N_MACROS)
    # continuous: vx, vy, turn, intake, shoot, climb; continuous_aim adds exit speed, hood, turret
    return spaces.Box(low=-1.0, high=1.0, shape=(9 if mode == "continuous_aim" else 6,), dtype=np.float32)


OBSERVATION_SPACE = spaces.Box(low=-2.0, high=2.0, shape=(OBS_SIZE,), dtype=np.float32)
OBSERVATION_SPACE_HIFI = spaces.Box(low=-2.0, high=2.0, shape=(OBS_SIZE_HIFI,), dtype=np.float32)


def observation_space(cfg: EnvConfig) -> spaces.Box:
    return OBSERVATION_SPACE if cfg.hifi is None else OBSERVATION_SPACE_HIFI


class RebuiltEnv(gym.Env):
    """Single-agent REBUILT match: the agent drives one robot, bots drive the rest."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None, **overrides) -> None:
        cfg = config or EnvConfig()
        if overrides:
            cfg = replace(cfg, **overrides)
        self.cfg = cfg
        self.render_mode = render_mode
        self.action_space = action_space(cfg.action_mode)
        self.observation_space = observation_space(cfg)
        self.runner: MatchRunner | None = None
        self.learner = cfg.learner_slots[0]
        self._viewer = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        self.learner = int(rng.choice(self.cfg.learner_slots))
        if options and "learner_slot" in options:
            self.learner = int(options["learner_slot"])
        tiers = {self.learner: options["learner_tier"]} if options and "learner_tier" in options else None
        self.runner = MatchRunner(self.cfg, [self.learner])
        self.runner.reset(rng, tiers)
        return self.runner.observe(self.learner), self.runner.info(self.learner)

    def step(self, action):
        run = self.runner
        run.step({self.learner: action})
        obs = run.observe(self.learner)
        reward = run.reward(self.learner)
        terminated = run.match.done
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, run.info(self.learner)

    @property
    def match(self) -> Match:
        return self.runner.match

    def render(self):
        from .viewer import Viewer  # pygame is optional

        if self._viewer is None:
            self._viewer = Viewer(headless=self.render_mode == "rgb_array")
        return self._viewer.draw(self.runner.match, highlight=self.learner,
                                 return_array=self.render_mode == "rgb_array")

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


gym.register(id="Rebuilt-Strategy-v0", entry_point="rebuilt_sim.env:RebuiltEnv",
             kwargs={"action_mode": "macro"})
gym.register(id="Rebuilt-Control-v0", entry_point="rebuilt_sim.env:RebuiltEnv",
             kwargs={"action_mode": "continuous"})
# high-fidelity physics (docs/03-driving-and-aiming.md)
gym.register(id="Rebuilt-Strategy-HiFi-v0", entry_point="rebuilt_sim.env:RebuiltEnv",
             kwargs={"action_mode": "macro", "hifi": HiFiConfig()})
gym.register(id="Rebuilt-Control-HiFi-v0", entry_point="rebuilt_sim.env:RebuiltEnv",
             kwargs={"action_mode": "continuous", "hifi": HiFiConfig()})
gym.register(id="Rebuilt-Aim-HiFi-v0", entry_point="rebuilt_sim.env:RebuiltEnv",
             kwargs={"action_mode": "continuous_aim", "hifi": HiFiConfig()})
