"""Scripted drivers that play like the teams on the 2026 Istanbul Regional broadcast.

Each ``BotDriver`` picks a macro action every ``reaction`` seconds; ``MacroController``
turns it into drive commands. The observed elite pattern (docs/01-video-analysis.md §4.3):
sweep the center FUEL in AUTO, hoard while the HUB is off, unload the moment it turns on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C
from .controller import Macro, MacroController, path_length
from .robot import RobotCommand
from .rules import Period


@dataclass(frozen=True)
class BotStyle:
    auto: str = "preload"  # "sweep" | "depot" | "preload" | "none"
    auto_return: float = 11.0  # s into AUTO when a sweeping robot heads back to shoot
    min_load: int = 6  # FUEL in the hopper before driving to shoot while the HUB is on
    hoard: bool = True  # keep collecting while the HUB is off
    defend: float = 0.0  # chance to play defense during an off window with a full hopper
    climb_at: float | None = None  # seconds left on the clock when it leaves to climb
    use_outpost: bool = False  # refill from the human player when the field is bare nearby
    reaction: float = 0.3  # s between decisions
    awareness: float = 1.0  # chance the drive team plays around the HUB schedule at all this match
    distraction: float = 0.0  # chance per decision of losing the next reaction period (hesitation)
    foul_rate: float = 0.0  # incidental fouls per minute the physics doesn't model (Day 1: ~0.03)


STYLES: dict[str, BotStyle] = {
    "elite": BotStyle(auto="sweep", auto_return=10.5, min_load=18, use_outpost=True, distraction=0.05,
                      foul_rate=0.01),
    "strong": BotStyle(auto="sweep", auto_return=11.5, min_load=10, use_outpost=True, awareness=0.9,
                       distraction=0.2, foul_rate=0.02),
    "mid": BotStyle(auto="depot", min_load=5, awareness=0.6, distraction=0.25, foul_rate=0.03),
    "low": BotStyle(auto="preload", min_load=3, reaction=0.6, awareness=0.3, distraction=0.4, foul_rate=0.05),
    "climber": BotStyle(auto="depot", min_load=5, climb_at=16.0, awareness=0.6, distraction=0.25, foul_rate=0.03),
    "elite_climber": BotStyle(auto="sweep", auto_return=10.5, min_load=15, climb_at=14.0, use_outpost=True,
                              distraction=0.05, foul_rate=0.01),
    "broken": BotStyle(auto="none"),
    "defender": BotStyle(auto="preload", min_load=4, defend=1.0, distraction=0.1, foul_rate=0.05),
}
MAJOR_SHARE = 0.4  # Day 1: 5 of 11 foul awards were 15+ points


class BotDriver:
    def __init__(self, style: BotStyle, rng: np.random.Generator) -> None:
        self.style = style
        self.rng = rng
        self.aware = rng.random() < style.awareness
        self.window: float | None = None  # when our HUB next turns on; one defense decision per off window
        self.defend_now = False

    def decide(self, m, r, ctl: MacroController) -> Macro:
        st = self.style
        if st.auto == "none" or r.spec.max_speed <= 0:
            return Macro.IDLE
        mem = ctl.mem[r.index]
        period = m.period
        if period in (Period.PAUSE, Period.POST, Period.DONE):
            return mem.macro
        if self.rng.random() < st.distraction:
            return Macro.IDLE
        if period == Period.AUTO:
            return self._auto(m, r, ctl)
        clock = C.MATCH_END - m.t
        if st.climb_at is not None and r.spec.climb_level > 0 and clock <= st.climb_at:
            return Macro.CLIMB
        cap = r.spec.capacity
        if not self.aware:  # ignores the HUB schedule: shoots whenever loaded, even into an inactive HUB
            if (mem.macro == Macro.SHOOT and r.fuel > 0) or r.fuel >= min(st.min_load, cap):
                return Macro.SHOOT
            return self._refill(m, r, ctl)
        active = m.hub_active(r.alliance)
        ttg = m.time_to_toggle(r.alliance)
        to_spot = path_length(r, *ctl.shoot_spot(r)) / max(r.spec.max_speed, 0.1) + 0.8
        if active:
            ending = ttg is not None and ttg < to_spot + 1.5
            if mem.macro == Macro.SHOOT and r.fuel > 0:
                return Macro.SHOOT  # keep unloading
            if r.fuel >= min(st.min_load, cap) or (r.fuel > 0 and ending):
                return Macro.SHOOT
            return self._refill(m, r, ctl)
        # our HUB is off
        window = m.schedule.next_toggle(r.alliance, m.t)
        if window != self.window:
            self.window = window
            self.defend_now = self.rng.random() < st.defend
        if ttg is not None and ttg < to_spot + 0.5 and r.fuel >= min(st.min_load, cap):
            return Macro.STAGE
        if self.defend_now and r.fuel >= min(st.min_load, cap):
            return Macro.DEFEND
        if r.fuel < cap and st.hoard:
            return self._refill(m, r, ctl)
        return Macro.DEFEND if self.defend_now else Macro.STAGE

    def _auto(self, m, r, ctl: MacroController) -> Macro:
        st, t, mem = self.style, m.t, ctl.mem[r.index]
        if st.auto == "sweep":
            if t < st.auto_return and r.fuel < 0.85 * r.spec.capacity:
                return Macro.COLLECT
            return Macro.SHOOT if r.fuel > 0 else Macro.COLLECT
        if st.auto == "depot":
            if not mem.notes.get("preload_done"):
                if r.fuel > 0:
                    return Macro.SHOOT
                mem.notes["preload_done"] = True
            if r.fuel < min(6, r.spec.capacity) and t < 14.0 and m.depot_count(r.alliance) > 0:
                return Macro.COLLECT_DEPOT
            return Macro.SHOOT if r.fuel > 0 else Macro.COLLECT_DEPOT
        return Macro.SHOOT if r.fuel > 0 else Macro.IDLE

    def _refill(self, m, r, ctl: MacroController) -> Macro:
        if m.depot_count(r.alliance) >= 4 and region_is_own(r):
            return Macro.COLLECT_DEPOT
        if self.style.use_outpost and m.chute_count[r.alliance] > 0:
            grid, _ = ctl._fuel_grid()
            if grid.sum() < 40:
                return Macro.COLLECT_OUTPOST
        return Macro.COLLECT


def region_is_own(r) -> bool:
    return r.x < C.ALLIANCE_ZONE_DEPTH if r.alliance == 0 else r.x > C.FIELD_LENGTH - C.ALLIANCE_ZONE_DEPTH


class ScriptedPolicy:
    """Scripted control of some robots in a match. Call it every step to get their commands.

    ``robots``: indices it controls (default: all six). Pass a shared ``controller`` when
    a learner also drives robots in the same match so teammates coordinate targets.
    """

    def __init__(self, match, robots=None, styles: dict[int, BotStyle] | None = None,
                 controller: MacroController | None = None, seed: int | None = None) -> None:
        self.match = match
        self.ctl = controller or MacroController(match)
        # one decision stream and one foul stream per robot seat, so a seat's luck doesn't depend
        # on which other seats are scripted (keeps evaluate.py comparisons like-for-like)
        streams = np.random.SeedSequence(seed).spawn(12)
        self.robots = list(range(6)) if robots is None else list(robots)
        styles = styles or {}
        self.drivers = {i: BotDriver(styles.get(i) or STYLES.get(match.robots[i].spec.name, STYLES["mid"]),
                                     np.random.default_rng(streams[i])) for i in self.robots}
        self.foul_rngs = {i: np.random.default_rng(streams[6 + i]) for i in self.robots}
        self.macros = {i: Macro.IDLE for i in self.robots}
        self.next_decision = {i: 0.0 for i in self.robots}

    def macro(self, i: int) -> Macro:
        m = self.match
        if m.t >= self.next_decision[i] - 1e-9:
            self.macros[i] = self.drivers[i].decide(m, m.robots[i], self.ctl)
            self.next_decision[i] = m.t + self.drivers[i].style.reaction
        return self.macros[i]

    def commands(self, match=None) -> list[RobotCommand | None]:
        m = self.match
        out: list[RobotCommand | None] = [None] * 6
        playing = m.period in (Period.AUTO, Period.TRANSITION, Period.SHIFT1, Period.SHIFT2, Period.SHIFT3,
                               Period.SHIFT4, Period.ENDGAME)
        for i in self.robots:
            out[i] = self.ctl.command(m.robots[i], self.macro(i))
            rate = self.drivers[i].style.foul_rate
            frng = self.foul_rngs[i]
            if playing and rate > 0 and frng.random() < rate / 60.0 * m.cfg.dt:
                m.call_foul(i, "incidental foul (scripted bot)", major=frng.random() < MAJOR_SHARE)
        return out

    __call__ = commands
