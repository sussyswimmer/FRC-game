"""Robot capabilities, state and commands.

A ``RobotSpec`` is what a team built: speed, hopper size, intake and shooter rates,
shooting accuracy, height (TRENCH or not) and climbing. ``TIERS`` holds presets
calibrated against the 2026 Istanbul Regional Day 1 data (docs/01-video-analysis.md §4.4).
The high-fidelity physics also uses the hardware in ``DriveSpec`` and ``ShooterSpec`` and
the bumper footprint (docs/03-driving-and-aiming.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import IntEnum

from . import constants as C
from .constants import Alliance


@dataclass(frozen=True)
class DriveSpec:
    """Swerve drivetrain hardware (high fidelity). Four modules, one drive motor each."""

    motor: str = "kraken_x60"  # drive motor, a key of drivetrain.MOTORS
    gear_ratio: float = 6.75  # motor turns per wheel turn (SDS MK4i "L2")
    wheel_radius: float = 2.0 * C.IN
    current_limit: float = 45.0  # A, stator current limit per drive motor
    wheel_cof: float = 1.1  # tread on carpet
    steer_rate: float = 25.0  # rad/s, fastest a module can turn


@dataclass(frozen=True)
class ShooterSpec:
    """FUEL launcher (high fidelity): what the robot's aim software can set, and how precisely."""

    speed_range: tuple[float, float] = (4.5, 10.0)  # m/s FUEL exit speed
    hood_range: tuple[float, float] = (0.85, 1.30)  # rad launch elevation (equal ends: fixed hood)
    height: float = 0.50  # m, exit height above the carpet
    spinup: float = 0.40  # s, flywheel time constant
    hood_rate: float = 2.0  # rad/s
    turret_rate: float = 8.0  # rad/s (robots with a turret)
    turret_range: float = math.pi  # turret travel either side of the heading
    speed_sd: float = 0.03  # shot-to-shot exit speed spread, as a fraction
    angle_sd: float = 0.02  # rad, launch direction spread (elevation and yaw)
    recovery: float = 0.03  # fraction of flywheel speed each shot takes out
    aim_tolerance: float = 0.06  # rad of aiming error the aim software accepts before it fires
    lead: bool = False  # aim software leads shots taken on the move


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
    # high fidelity only (docs/03-driving-and-aiming.md)
    length: float = 0.86  # m, bumper to bumper along the heading (a 27.5 in frame plus bumpers)
    width: float = 0.86  # m, bumper to bumper across
    drive: DriveSpec = DriveSpec()
    shooter: ShooterSpec = ShooterSpec()

    @property
    def can_trench(self) -> bool:
        return self.height <= C.TRENCH_CLEARANCE

    def hit_probability(self, distance: float, speed: float) -> float:
        p = self.accuracy - self.falloff * max(0.0, distance - self.sweet_range) - self.moving_penalty * speed
        return min(1.0, max(0.02, p))

    def climb_duration(self, level: int) -> float:
        return self.climb_time * (0.4 + 0.6 * level / max(1, self.climb_level))


# High-fidelity hardware. Drive current limits make each tier accelerate at its max_accel;
# shooter spreads are tuned so hit rates match the tier's accuracy (docs/03-driving-and-aiming.md).
_KRAKEN_FOC_L3 = DriveSpec(motor="kraken_x60_foc", gear_ratio=6.12, current_limit=55.0, steer_rate=30.0)
_KRAKEN_L2 = DriveSpec(motor="kraken_x60", gear_ratio=6.75, current_limit=44.0)
_NEO_L2 = DriveSpec(motor="neo", gear_ratio=6.75, current_limit=40.0, steer_rate=20.0)
_NEO_L1 = DriveSpec(motor="neo", gear_ratio=8.14, current_limit=26.0, steer_rate=15.0)
_ELITE_SHOOTER = ShooterSpec(speed_range=(4.5, 12.0), hood_range=(0.70, 1.40), spinup=0.25, hood_rate=4.0,
                             turret_rate=10.0, speed_sd=0.039, angle_sd=0.026, recovery=0.02, aim_tolerance=0.03,
                             lead=True)
_STRONG_SHOOTER = ShooterSpec(speed_range=(4.5, 10.0), hood_range=(0.85, 1.30), spinup=0.40, speed_sd=0.090,
                              angle_sd=0.053, aim_tolerance=0.06)
_MID_SHOOTER = ShooterSpec(speed_range=(4.5, 9.0), hood_range=(1.00, 1.30), height=0.62, spinup=0.60,
                           hood_rate=1.5, speed_sd=0.124, angle_sd=0.078, recovery=0.04, aim_tolerance=0.08)
_LOW_SHOOTER = ShooterSpec(speed_range=(4.5, 8.5), hood_range=(1.25, 1.25), height=0.62, spinup=0.80,
                           speed_sd=0.13, angle_sd=0.085, recovery=0.05, aim_tolerance=0.10)

_MID = dict(
    max_speed=3.6, max_accel=6.5, capacity=10, intake_rate=2.0, intake_width=0.50, shoot_rate=1.4,
    accuracy=0.50, sweet_range=2.0, falloff=0.14, max_range=3.5, jam_rate=1.0, jam_time=6.0,
    length=0.90, width=0.82, drive=_NEO_L2, shooter=_MID_SHOOTER,
)

TIERS: dict[str, RobotSpec] = {
    # ~225-250 FUEL/match on its own: big hopper, turret, drives under the TRENCH (team 9483)
    "elite": RobotSpec(
        name="elite", height=0.55, max_speed=4.5, max_accel=8.5, capacity=44, intake_rate=7.0,
        intake_width=0.75, intake_reach=0.20, shoot_rate=5.5, accuracy=0.90, sweet_range=3.2,
        falloff=0.05, max_range=6.5, turret=True, moving_penalty=0.03, jam_rate=0.3, jam_time=3.0,
        length=0.84, width=0.84, drive=_KRAKEN_FOC_L3, shooter=_ELITE_SHOOTER,
    ),
    # 40-70 FUEL/match (6431, 9077, 4481, 3646)
    "strong": RobotSpec(
        name="strong", height=0.55, max_speed=4.0, max_accel=7.5, capacity=18, intake_rate=3.2,
        intake_width=0.60, shoot_rate=2.2, accuracy=0.65, sweet_range=2.4, falloff=0.10,
        max_range=4.5, moving_penalty=0.05, jam_rate=0.6, jam_time=5.0, drive=_KRAKEN_L2, shooter=_STRONG_SHOOTER,
    ),
    # 15-30 FUEL/match
    "mid": RobotSpec(name="mid", **_MID),
    # 15 FUEL/match or less: about half the Day 1 field
    "low": RobotSpec(
        name="low", max_speed=3.0, max_accel=5.0, capacity=6, intake_rate=1.2, intake_width=0.40,
        shoot_rate=0.8, accuracy=0.35, sweet_range=1.6, falloff=0.18, max_range=2.8, jam_rate=1.5,
        jam_time=8.0, length=0.92, width=0.82, drive=_NEO_L1, shooter=_LOW_SHOOTER,
    ),
    # mid-level scorer that climbs to LEVEL 1, and makes it about half the time (team 10428 on Day 1)
    "climber": RobotSpec(name="climber", climb_level=1, climb_time=6.0, climb_success=0.5, **_MID),
    # a later-season build: strong scorer plus a LEVEL 3 climb
    "elite_climber": RobotSpec(
        name="elite_climber", height=0.55, max_speed=4.4, max_accel=8.5, capacity=40, intake_rate=8.0,
        intake_width=0.70, shoot_rate=6.0, accuracy=0.90, sweet_range=3.0, falloff=0.06,
        max_range=6.0, turret=True, moving_penalty=0.03, climb_level=3, climb_time=7.0, jam_rate=0.3,
        jam_time=3.0, length=0.84, width=0.84, drive=_KRAKEN_FOC_L3, shooter=_ELITE_SHOOTER,
    ),
    # didn't move all match (P1 red, P4 blue scored 0)
    "broken": RobotSpec(name="broken", max_speed=0.0, max_accel=0.0, capacity=0, intake_rate=0.0,
                        shoot_rate=0.0, preload=0, length=0.90, width=0.82),
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
    # manual aiming (high fidelity). None: the robot's aim software sets the shooter itself.
    shot_speed: float | None = None  # m/s exit speed
    hood: float | None = None  # rad launch elevation
    turret: float | None = None  # rad, relative to the heading (robots with a turret)


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
    rect: bool = False  # collides as its rectangular bumper footprint (high fidelity) instead of a circle
    flywheel: float = 0.0  # m/s exit speed the shooter is spinning at (high fidelity)
    hood: float = 0.0  # rad launch elevation
    turret: float = 0.0  # rad, relative to the heading
    shot_ready: bool = False  # the aim software has a solution and the shooter is on it
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

    @property
    def extent_x(self) -> float:
        """Half the bumper footprint's size along the field's x axis (zone rules, TOWER contact)."""
        if not self.rect:
            return self.spec.radius
        return (abs(self.spec.length * math.cos(self.heading)) + abs(self.spec.width * math.sin(self.heading))) / 2

    @property
    def reach(self) -> float:
        """Radius of the smallest circle around the center that holds the whole footprint."""
        return math.hypot(self.spec.length, self.spec.width) / 2 if self.rect else self.spec.radius

    @property
    def front(self) -> float:
        """Distance from the center to the front bumper, where the intake is."""
        return self.spec.length / 2 if self.rect else self.spec.radius
