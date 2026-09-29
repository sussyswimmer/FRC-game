"""Match rules: clock, HUB activity schedule, grace windows, scoring and ranking points.

Everything here is a pure function of match time and recorded events, so it can be tested
against the numbers observed on the 2026 Istanbul Regional broadcast
(see docs/01-video-analysis.md, section 5.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from . import constants as C
from .constants import Alliance


class Period(IntEnum):
    AUTO = 0
    PAUSE = 1
    TRANSITION = 2
    SHIFT1 = 3
    SHIFT2 = 4
    SHIFT3 = 5
    SHIFT4 = 6
    ENDGAME = 7
    POST = 8  # clock at 0:00, last grace window still open
    DONE = 9


PERIOD_STARTS = (
    (Period.AUTO, 0.0),
    (Period.PAUSE, C.AUTO_LEN),
    (Period.TRANSITION, C.TELEOP_START),
    (Period.SHIFT1, C.SHIFT1_START),
    (Period.SHIFT2, C.SHIFT1_START + C.SHIFT_LEN),
    (Period.SHIFT3, C.SHIFT1_START + 2 * C.SHIFT_LEN),
    (Period.SHIFT4, C.SHIFT1_START + 3 * C.SHIFT_LEN),
    (Period.ENDGAME, C.ENDGAME_START),
    (Period.POST, C.MATCH_END),
    (Period.DONE, C.FINAL_TIME),
)
_START = {p: s for p, s in PERIOD_STARTS}
_END = {p: e for (p, _), (_, e) in zip(PERIOD_STARTS, PERIOD_STARTS[1:])}

SHIFTS = (Period.SHIFT1, Period.SHIFT2, Period.SHIFT3, Period.SHIFT4)


def period_at(t: float) -> Period:
    result = Period.AUTO
    for p, start in PERIOD_STARTS:
        if t >= start:
            result = p
    return result


def period_bounds(p: Period) -> tuple[float, float]:
    return _START[p], _END.get(p, float("inf"))


def display_clock(t: float) -> float:
    """Seconds shown on the arena clock: counts down 20 -> 0 in AUTO, holds 2:20 in the pause,
    then 140 -> 0 in TELEOP."""
    if t < C.AUTO_LEN:
        return C.AUTO_LEN - t
    if t < C.TELEOP_START:
        return C.MATCH_END - C.TELEOP_START
    return max(0.0, C.MATCH_END - t)


def is_teleop(t: float) -> bool:
    return C.TELEOP_START <= t < C.MATCH_END


class HubSchedule:
    """Which HUBs are active when.

    Both HUBs are active in AUTO, the TRANSITION SHIFT and END GAME. The alliance that
    scored more FUEL in AUTO has its HUB inactive in SHIFT 1 (and SHIFT 3); the other
    alliance is inactive in SHIFT 2 and SHIFT 4. A tie in AUTO is broken at random.
    """

    def __init__(self) -> None:
        self.first_inactive: Alliance | None = None

    def resolve_auto(self, blue_auto_fuel: int, red_auto_fuel: int, coin: float) -> Alliance:
        if blue_auto_fuel > red_auto_fuel:
            self.first_inactive = Alliance.BLUE
        elif red_auto_fuel > blue_auto_fuel:
            self.first_inactive = Alliance.RED
        else:
            self.first_inactive = Alliance.BLUE if coin < 0.5 else Alliance.RED
        return self.first_inactive

    def inactive_shifts(self, alliance: Alliance) -> tuple[Period, Period]:
        if self.first_inactive is None:
            raise RuntimeError("HUB order is not known until AUTO scoring closes")
        if alliance == self.first_inactive:
            return Period.SHIFT1, Period.SHIFT3
        return Period.SHIFT2, Period.SHIFT4

    def is_active(self, alliance: Alliance, t: float) -> bool:
        p = period_at(t)
        if p in SHIFTS:
            if self.first_inactive is None:
                return True
            return p not in self.inactive_shifts(alliance)
        return p in (Period.AUTO, Period.PAUSE, Period.TRANSITION, Period.ENDGAME)

    def deactivation_times(self, alliance: Alliance) -> tuple[float, ...]:
        """Times at which this alliance's HUB switches from active to inactive."""
        times = [C.MATCH_END]
        if self.first_inactive is not None:
            times += [_START[p] for p in self.inactive_shifts(alliance)]
        return tuple(sorted(times))

    def counts(self, alliance: Alliance, t_sensed: float) -> bool:
        """Whether FUEL passing this alliance's HUB counter at ``t_sensed`` scores.

        FUEL counts while the HUB is active and for GRACE seconds after it deactivates
        (including after the clock reaches 0:00 at the end of the match).
        """
        if t_sensed < C.MATCH_END and self.is_active(alliance, t_sensed):
            return True
        return any(d <= t_sensed <= d + C.GRACE for d in self.deactivation_times(alliance))

    def next_toggle(self, alliance: Alliance, t: float) -> float | None:
        """Time of the next change in this alliance's HUB status after ``t``.

        None if the status won't change again before the match ends, or if the HUB order
        is not decided yet (AUTO scoring still open).
        """
        if self.first_inactive is None:
            return None
        now = self.is_active(alliance, t)
        for p, start in PERIOD_STARTS:
            if start > t and p <= Period.ENDGAME and self.is_active(alliance, start) != now:
                return start
        return None


def fuel_counts_for_auto(t_sensed: float) -> bool:
    """FUEL counted up to GRACE seconds after AUTO ends is credited to AUTO."""
    return t_sensed <= C.AUTO_LEN + C.GRACE


@dataclass
class AllianceScore:
    auto_fuel: int = 0
    teleop_fuel: int = 0
    auto_tower: int = 0
    teleop_tower: int = 0
    penalty_points: int = 0  # points received from the opponent's fouls
    minor_fouls: int = 0  # fouls this alliance committed
    major_fouls: int = 0
    wasted_fuel: int = 0  # FUEL that went through this HUB while it was not counting

    @property
    def fuel(self) -> int:
        return self.auto_fuel + self.teleop_fuel

    @property
    def tower(self) -> int:
        return self.auto_tower + self.teleop_tower

    @property
    def total(self) -> int:
        return self.fuel + self.tower + self.penalty_points


@dataclass
class RankingPoints:
    win: int = 0
    energized: int = 0
    supercharged: int = 0
    traversal: int = 0

    @property
    def total(self) -> int:
        return self.win + self.energized + self.supercharged + self.traversal


def ranking_points(own: AllianceScore, opp: AllianceScore, level: str = "regional") -> RankingPoints:
    energized, supercharged, traversal = C.RP_THRESHOLDS[level]
    if own.total > opp.total:
        win = C.WIN_RP
    elif own.total == opp.total:
        win = C.TIE_RP
    else:
        win = 0
    return RankingPoints(
        win=win,
        energized=int(own.fuel >= energized),
        supercharged=int(own.fuel >= supercharged),
        traversal=int(own.tower >= traversal),
    )


@dataclass
class Foul:
    t: float
    committed_by: Alliance
    robot: int
    rule: str
    major: bool


@dataclass
class FoulLedger:
    fouls: list[Foul] = field(default_factory=list)

    def call(self, scores: list[AllianceScore], t: float, alliance: Alliance, robot: int, rule: str, major: bool) -> None:
        self.fouls.append(Foul(t, alliance, robot, rule, major))
        offender, victim = scores[alliance], scores[alliance.other]
        if major:
            offender.major_fouls += 1
            victim.penalty_points += C.MAJOR_FOUL
        else:
            offender.minor_fouls += 1
            victim.penalty_points += C.MINOR_FOUL
