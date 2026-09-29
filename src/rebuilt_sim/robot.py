"""Robot capabilities, state and commands.

A ``RobotSpec`` is what a team built: speed, hopper size, intake and shooter rates,
shooting accuracy, height (TRENCH or not) and climbing. ``TIERS`` holds presets
calibrated against the 2026 Istanbul Regional Day 1 data (docs/01-video-analysis.md §4.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import IntEnum

from . import constants as C
from .constants import Alliance


@dataclass(frozen=True)
class RobotSpec:
    name: str = "mid"
    radius: float = 0.46  # collision radius including bumpers
    height: float = 0.70  # <= TRENCH_CLEARANCE (0.565 m) can drive under the TRENCH
    mass: float = 60.0  # kg with bumpers and battery
    max_speed: float = 3.8  # m/s
    max_accel: float = 7.0  # m/s^2
    max_omega: float = 6.0  # rad/s
    bump_speed: float = 1.6  # speed cap while driving over a BUMP
    capacity: int = 18  # FUEL the hopper holds
    intake_rate: float = 5.0  # FUEL/s it can swallow
    intake_width: float = 0.60  # m across the front
    intake_reach: float = 0.18  # m beyond the bumper
    shoot_rate: float = 3.0  # FUEL/s
    accuracy: float = 0.72  # hit probability at or inside sweet_range
    sweet_range: float = 2.2  # m from the HUB center
    falloff: float = 0.12  # hit probability lost per meter beyond sweet_range
    min_range: float = 0.9  # m from the HUB center
    max_range: float = 4.0
    turret: bool = False  # without a turret the robot must face its HUB to shoot
    moving_penalty: float = 0.06  # hit probability lost per m/s of speed while shooting
    climb_level: int = 0  # highest TOWER LEVEL reachable (0 = no climber)
    climb_time: float = 6.0  # s to reach climb_level
    climb_success: float = 0.9
    jam_rate: float = 0.0  # intake/shooter jams per minute of match time
    jam_time: float = 4.0  # s a jam takes to clear
    preload: int = C.MAX_PRELOAD

    @property
    def can_trench(self) -> bool:
        return self.height <= C.TRENCH_CLEARANCE

    def hit_probability(self, distance: float, speed: float) -> float:
        p = self.accuracy - self.falloff * max(0.0, distance - self.sweet_range) - self.moving_penalty * speed
        return min(1.0, max(0.02, p))

    def climb_duration(self, level: int) -> float:
        return self.climb_time * (0.4 + 0.6 * level / max(1, self.climb_level))


_MID = dict(
    max_speed=3.6, max_accel=6.5, capacity=10, intake_rate=2.0, intake_width=0.50, shoot_rate=1.4,
    accuracy=0.50, sweet_range=2.0, falloff=0.14, max_range=3.5, jam_rate=1.0, jam_time=6.0,
)

TIERS: dict[str, RobotSpec] = {
    # ~225-250 FUEL/match on its own: big hopper, turret, drives under the TRENCH (team 9483)
    "elite": RobotSpec(
        name="elite", height=0.55, max_speed=4.5, max_accel=8.5, capacity=44, intake_rate=7.0,
        intake_width=0.75, intake_reach=0.20, shoot_rate=5.5, accuracy=0.90, sweet_range=3.2,
        falloff=0.05, max_range=6.5, turret=True, moving_penalty=0.03, jam_rate=0.3, jam_time=3.0,
    ),
    # 40-70 FUEL/match (6431, 9077, 4481, 3646)
    "strong": RobotSpec(
        name="strong", height=0.55, max_speed=4.0, max_accel=7.5, capacity=18, intake_rate=3.2,
        intake_width=0.60, shoot_rate=2.2, accuracy=0.65, sweet_range=2.4, falloff=0.10,
        max_range=4.5, moving_penalty=0.05, jam_rate=0.6, jam_time=5.0,
    ),
    # 15-30 FUEL/match
    "mid": RobotSpec(name="mid", **_MID),
    # 15 FUEL/match or less: about half the Day 1 field
    "low": RobotSpec(
        name="low", max_speed=3.0, max_accel=5.0, capacity=6, intake_rate=1.2, intake_width=0.40,
        shoot_rate=0.8, accuracy=0.35, sweet_range=1.6, falloff=0.18, max_range=2.8, jam_rate=1.5,
        jam_time=8.0,
    ),
    # mid-level scorer that climbs to LEVEL 1, and makes it about half the time (team 10428 on Day 1)
    "climber": RobotSpec(name="climber", climb_level=1, climb_time=6.0, climb_success=0.5, **_MID),
    # a later-season build: strong scorer plus a LEVEL 3 climb
    "elite_climber": RobotSpec(
        name="elite_climber", height=0.55, max_speed=4.4, max_accel=8.5, capacity=40, intake_rate=8.0,
        intake_width=0.70, shoot_rate=6.0, accuracy=0.90, sweet_range=3.0, falloff=0.06,
        max_range=6.0, turret=True, moving_penalty=0.03, climb_level=3, climb_time=7.0, jam_rate=0.3,
        jam_time=3.0,
    ),
    # didn't move all match (P1 red, P4 blue scored 0)
    "broken": RobotSpec(name="broken", max_speed=0.0, max_accel=0.0, capacity=0, intake_rate=0.0,
                        shoot_rate=0.0, preload=0),
}

# Day 1 early-season mix: 1 elite, ~5 strong, ~12 mid, ~16 low among 34 teams
DAY1_TIER_WEIGHTS = {"elite": 0.03, "strong": 0.13, "mid": 0.28, "climber": 0.03, "low": 0.47, "broken": 0.06}


def spec(name: str, **overrides) -> RobotSpec:
    return replace(TIERS[name], **overrides) if overrides else TIERS[name]


class ClimbState(IntEnum):
    GROUND = 0
    CLIMBING = 1
    CLIMBED = 2
    DESCENDING = 3


@dataclass
class RobotCommand:
    """Field-relative drive command plus mechanism requests."""

    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    intake: bool = False
    shoot: bool = False
    climb: int = 0  # requested TOWER LEVEL (0 = none)


@dataclass
class RobotStats:
    fuel_collected: int = 0
    fuel_shot: int = 0
    fuel_scored: int = 0  # shot by this robot and counted for its alliance
    fuel_wasted: int = 0  # shot by this robot into a HUB that was not counting
    fouls_minor: int = 0
    fouls_major: int = 0
    tower_points: int = 0


@dataclass
class Robot:
    spec: RobotSpec
    alliance: Alliance
    slot: int  # 0..2 within the alliance
    index: int  # 0..5 in the match (blue 0-2, red 3-5)
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    fuel: int = 0
    intake_budget: float = 0.0
    shoot_cooldown: float = 0.0
    jam_timer: float = 0.0  # > 0 while the intake/shooter is jammed
    climb_state: ClimbState = ClimbState.GROUND
    climb_target: int = 0
    climb_timer: float = 0.0
    climb_level: int = 0
    auto_tower: bool = False
    awarded_level3: bool = False  # G420: fouled while climbing in END GAME
    crossed_center_foul: bool = False
    g407_cooldown: float = 0.0
    last_cmd: RobotCommand = field(default_factory=RobotCommand)
    stats: RobotStats = field(default_factory=RobotStats)

    @property
    def name(self) -> str:
        return f"{'blue' if self.alliance == Alliance.BLUE else 'red'}_{self.slot}"

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def mobile(self) -> bool:
        return self.climb_state == ClimbState.GROUND and self.spec.max_speed > 0
