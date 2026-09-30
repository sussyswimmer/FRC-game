"""The match simulator: robots, FUEL, HUBs, TOWERs, human players and the referee.

``Match.step(commands)`` advances the world by ``dt`` seconds. Time 0 is the start of
AUTO; the match is over (``done``) once the last scoring grace window closes at
``constants.FINAL_TIME``. All 504 FUEL are tracked individually so the resource
loop (neutral zone -> hopper -> HUB -> exits -> neutral zone) is physical.

``MatchConfig(hifi=HiFiConfig())`` switches on the high-fidelity physics for driving and
aiming: swerve modules, rectangular bumpers, command latency, pose estimation and 3D FUEL
ballistics (docs/03-driving-and-aiming.md). The rules and the referee are the same.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import collision
from . import constants as C
from .ballistics import Shooters
from .constants import Alliance
from .drivetrain import SwerveDrive
from .field import BAND_X, FIELD
from .robot import TIERS, ClimbState, Robot, RobotCommand, RobotSpec
from .sensors import PoseEstimator
from .rules import (
    AllianceScore,
    FoulLedger,
    HubSchedule,
    Period,
    RankingPoints,
    fuel_counts_for_auto,
    period_at,
    ranking_points,
)

# FUEL states (re-exported: tests and tools import them from here)
GROUND, HELD, FLIGHT, HUB, CHUTE = C.GROUND, C.HELD, C.FLIGHT, C.HUB, C.CHUTE

_CELL = 2 * C.FUEL_RADIUS
_NX = int(math.ceil(C.FIELD_LENGTH / _CELL))
_NY = int(math.ceil(C.FIELD_WIDTH / _CELL))
_PNY = _NY + 2  # contact grid is padded by one cell on each side so neighbors never go out of range
_CELL_CAP = 4  # FUEL per grid cell considered for ball-ball contact
_NEIGHBOR_OFFSETS = np.array([ox * _PNY + oy for ox in (-1, 0, 1) for oy in (-1, 0, 1)], dtype=np.int64)
_OBSTACLE_X = (C.TOWER_DEPTH, BAND_X[0], BAND_X[1], C.FIELD_LENGTH - C.TOWER_DEPTH)


@dataclass(frozen=True)
class HiFiConfig:
    """High-fidelity physics for driving and aiming (docs/03-driving-and-aiming.md). Each part
    can be switched off on its own, e.g. to measure how much it changes a trained policy."""

    dt: float = 0.02  # physics step: WPILib's 50 Hz robot loop
    swerve: bool = True  # swerve modules: steering, motor torque and current limits, traction, wheel slip
    footprints: bool = True  # rectangular bumpers that turn when hit off-center, instead of circles
    ballistics: bool = True  # FUEL flies in 3D into the HUB opening, instead of a hit probability
    latency: float = 0.04  # s between a command and the motors acting on it
    sensor_noise: float = 1.0  # scale of pose-estimate and perception noise; 0 = robots know the truth
    fuel_every: int = 2  # FUEL rolling physics runs every Nth step (every 0.04 s) to save time


@dataclass
class MatchConfig:
    dt: float = 0.05  # physics step (s); high fidelity uses hifi.dt
    event_level: str = "regional"  # sets ranking-point thresholds
    hp_feed_rate: float = 4.0  # FUEL/s a human player pushes through the OUTPOST chute
    fuel_friction: float = 2.5  # m/s^2 rolling deceleration of FUEL on carpet
    fuel_restitution: float = 0.35
    robot_restitution: float = 0.1
    random_starts: bool = False  # jitter starting spots along the ROBOT STARTING LINE
    hifi: HiFiConfig | None = None  # high-fidelity physics (None: the fast strategy-level physics)

    def __post_init__(self) -> None:
        if self.hifi is not None:
            self.dt = self.hifi.dt


@dataclass
class ScoreEvent:
    t: float
    alliance: Alliance
    robot: int
    auto: bool


@dataclass
class MatchSummary:
    scores: list[AllianceScore]
    rp: list[RankingPoints]
    winner: Alliance | None
    first_inactive: Alliance | None
    fouls: list = field(default_factory=list)
    robot_stats: list = field(default_factory=list)


class Match:
    def __init__(
        self,
        specs: list[RobotSpec],
        config: MatchConfig | None = None,
        seed: int | None = None,
        starts: list[tuple[float, float]] | None = None,
    ) -> None:
        if len(specs) != 6:
            raise ValueError("a match needs 6 robot specs: blue 0-2 then red 0-2")
        self.cfg = config or MatchConfig()
        # Separate random streams: field noise, the AUTO tie coin, and one per robot (shots, jams,
        # climbs), so changing one robot's behavior doesn't reshuffle every other robot's luck.
        # High fidelity adds a sensor stream and a perception stream.
        streams = np.random.SeedSequence(seed).spawn(10)
        field_ss, coin_ss, *robot_ss = streams[:8]
        self.rng = np.random.default_rng(field_ss)
        self.coin_rng = np.random.default_rng(coin_ss)
        self.robot_rngs = [np.random.default_rng(s) for s in robot_ss]
        self.field = FIELD
        self.step_count = 0
        self.t = 0.0
        self.done = False
        self.scores = [AllianceScore(), AllianceScore()]
        self.fouls = FoulLedger()
        self.schedule = HubSchedule()
        self.rp: list[RankingPoints] | None = None
        self.score_log: list[ScoreEvent] = []
        self.contacts: set[tuple[int, int]] = set()

        self.robots: list[Robot] = []
        for i, s in enumerate(specs):
            alliance = Alliance.BLUE if i < 3 else Alliance.RED
            self.robots.append(Robot(spec=s, alliance=alliance, slot=i % 3, index=i))
        self._place_robots(starts)

        n = 6
        self.pin_timer = np.zeros((n, n))
        self.pin_fouls = np.zeros((n, n), dtype=int)
        self.pin_last = np.full((n, n), -np.inf)  # when each pin (pinner, victim) last applied
        self._g403_contacts: set[tuple[int, int]] = set()
        self._g420_contacts: set[tuple[int, int]] = set()
        self.hp_budget = [0.0, 0.0]
        self._cell_table = np.full(((_NX + 2) * _PNY, _CELL_CAP), -1, dtype=np.int64)
        self._robot_boxes = {tall: [tuple(b) for b in self.field.obstacle_arrays[tall]] for tall in (False, True)}
        self._init_fuel()

        # high fidelity (docs/03-driving-and-aiming.md); all None / off for the strategy physics
        hifi = self.hifi = self.cfg.hifi
        self.drive: SwerveDrive | None = None
        self.collider: collision.Collider | None = None
        self.shooters: Shooters | None = None
        self.sensors: PoseEstimator | None = None
        self.obs_rng = np.random.default_rng(streams[9])  # perception noise in observations
        self._fuel_every = 1
        self._pending: list[deque] = []
        self.lookahead = 0.0  # s a robot's code looks ahead to cover command latency
        if hifi is not None:
            self.lookahead = hifi.latency + 2 * self.cfg.dt
            self._fuel_every = max(1, int(hifi.fuel_every))
            lag = int(round(hifi.latency / self.cfg.dt))
            self._pending = [deque([None] * lag) for _ in self.robots] if lag > 0 else []
            if hifi.swerve:
                self.drive = SwerveDrive(self.robots, self.cfg.dt)
            if hifi.footprints:
                for r in self.robots:
                    r.rect = True
                self.collider = collision.Collider(self)
            if hifi.ballistics:
                self.f_z = np.zeros(C.FUEL_TOTAL)  # height of FUEL in flight
                self.f_vz = np.zeros(C.FUEL_TOTAL)
                self.shooters = Shooters(self)
            if hifi.sensor_noise > 0:
                self.sensors = PoseEstimator(self, np.random.default_rng(streams[8]), hifi.sensor_noise)

    # ------------------------------------------------------------------ setup
    def _place_robots(self, starts: list[tuple[float, float]] | None) -> None:
        for r in self.robots:
            if starts is not None:
                x, y = starts[r.index]
            else:
                x, y = self.field.start_positions(r.alliance, r.spec.radius)[r.slot]
                if self.cfg.random_starts:
                    y = float(np.clip(y + self.rng.uniform(-0.5, 0.5), 0.6, C.FIELD_WIDTH - 0.6))
            r.x, r.y = x, y
            r.heading = 0.0 if r.alliance == Alliance.BLUE else math.pi

    def _init_fuel(self) -> None:
        n = C.FUEL_TOTAL
        self.f_pos = np.zeros((n, 2))
        self.f_vel = np.zeros((n, 2))
        self.f_state = np.full(n, GROUND, dtype=np.int8)
        self.f_owner = np.full(n, -1, dtype=np.int8)  # robot (HELD), alliance (HUB/CHUTE/FLIGHT target)
        self.f_shooter = np.full(n, -1, dtype=np.int8)
        self.f_hit = np.zeros(n, dtype=bool)
        self.f_time = np.zeros(n)  # arrival (FLIGHT) or counter time (HUB)
        self.f_t0 = np.zeros(n)  # launch time, for drawing arcs
        self.f_land = np.zeros((n, 2))
        self.f_settle = np.zeros(n, dtype=np.int8)
        k = 0
        for r in self.robots:  # preloads
            p = min(r.spec.preload, r.spec.capacity, C.MAX_PRELOAD)
            self.f_state[k:k + p] = HELD
            self.f_owner[k:k + p] = r.index
            r.fuel = p
            k += p
        for a in Alliance:  # OUTPOST chutes hold 24 each, off the field
            self.f_state[k:k + C.OUTPOST_CHUTE_FUEL] = CHUTE
            self.f_owner[k:k + C.OUTPOST_CHUTE_FUEL] = a
            k += C.OUTPOST_CHUTE_FUEL
        self.chute_count = [C.OUTPOST_CHUTE_FUEL, C.OUTPOST_CHUTE_FUEL]
        for a in Alliance:  # DEPOTs: 24 each on the floor
            box = self.field.depots[a]
            xs = np.linspace(box.x0, box.x1, 4 + 2)[1:-1]
            ys = np.linspace(box.y0, box.y1, 6 + 2)[1:-1]
            pts = np.array([(x, y) for x in xs for y in ys])[: C.DEPOT_FUEL]
            self.f_pos[k:k + len(pts)] = pts
            k += len(pts)
        # NEUTRAL ZONE block: every remaining FUEL, filled center-out in a 12 x 34 grid
        remaining = n - k
        gx, gy = C.NEUTRAL_BLOCK_GRID
        w, h = C.NEUTRAL_BLOCK_SIZE
        xs = C.CENTER_X - w / 2 + (np.arange(gx) + 0.5) * w / gx
        ys = C.CENTER_Y - h / 2 + (np.arange(gy) + 0.5) * h / gy
        slots = np.array([(x, y) for y in ys for x in xs])
        order = np.argsort(np.abs(slots[:, 1] - C.CENTER_Y), kind="stable")
        self.f_pos[k:] = slots[order[:remaining]]

    # ------------------------------------------------------------------ queries
    @property
    def period(self) -> Period:
        return period_at(self.t)

    def hub_active(self, alliance: Alliance) -> bool:
        return self.schedule.is_active(alliance, self.t) and self.t < C.MATCH_END

    def time_to_toggle(self, alliance: Alliance) -> float | None:
        nt = self.schedule.next_toggle(alliance, self.t)
        return None if nt is None else nt - self.t

    def ground_fuel(self) -> np.ndarray:
        return self.f_pos[self.f_state == GROUND]

    def fuel_in_flight(self) -> np.ndarray:
        """(k, 3) array of x, y, height-fraction for drawing FUEL in the air."""
        idx = np.flatnonzero(self.f_state == FLIGHT)
        if idx.size == 0:
            return np.zeros((0, 3))
        if self.shooters is not None:
            return np.column_stack([self.f_pos[idx], self.f_z[idx] / C.HUB_OPENING_HEIGHT])
        frac = np.clip((self.t - self.f_t0[idx]) / np.maximum(self.f_time[idx] - self.f_t0[idx], 1e-6), 0, 1)
        target = np.where(
            self.f_hit[idx, None],
            np.array(self.field.hub_centers)[self.f_owner[idx].clip(0)],
            self.f_land[idx],
        )
        xy = self.f_pos[idx] + (target - self.f_pos[idx]) * frac[:, None]
        return np.column_stack([xy, 4 * frac * (1 - frac)])

    def depot_count(self, alliance: Alliance) -> int:
        box = self.field.depots[alliance]
        g = self.ground_fuel()
        return int(np.count_nonzero((g[:, 0] >= box.x0) & (g[:, 0] <= box.x1) & (g[:, 1] >= box.y0) & (g[:, 1] <= box.y1)))

    def fuel_census(self) -> dict[str, int]:
        return {
            name: int(np.count_nonzero(self.f_state == s))
            for name, s in (("ground", GROUND), ("held", HELD), ("flight", FLIGHT), ("hub", HUB), ("chute", CHUTE))
        }

    def perceived(self, index: int) -> tuple[float, float, float, float, float]:
        """(x, y, heading, vx, vy) of a robot as its own software sees them: the pose estimate
        when sensors are simulated, the truth otherwise."""
        if self.sensors is None:
            r = self.robots[index]
            return r.x, r.y, r.heading, r.vx, r.vy
        return self.sensors.pose(index)

    def can_reach(self, r: Robot, distance: float) -> bool:
        """Whether the robot's shooter can drop FUEL into its HUB from ``distance`` m (high-fidelity
        ballistics); the strategy physics only uses the spec's range."""
        return self.shooters is None or self.shooters.can_reach(r, distance)

    def shot_reach(self, r: Robot) -> tuple[float, float]:
        """Closest and farthest distance from its HUB the robot's shooter can score from."""
        if self.shooters is None or self.shooters.tables[r.index] is None:
            return -math.inf, math.inf
        return self.shooters.tables[r.index].reach()

    def summary(self) -> MatchSummary:
        a, b = self.scores
        winner = None if a.total == b.total else (Alliance.BLUE if a.total > b.total else Alliance.RED)
        rp = self.rp or [ranking_points(self.scores[x], self.scores[x.other], self.cfg.event_level) for x in Alliance]
        return MatchSummary(
            scores=self.scores, rp=rp, winner=winner, first_inactive=self.schedule.first_inactive,
            fouls=list(self.fouls.fouls), robot_stats=[r.stats for r in self.robots],
        )

    # ------------------------------------------------------------------ stepping
    def step(self, commands: list[RobotCommand | None]) -> None:
        if self.done:
            return
        dt = self.cfg.dt
        t0 = self.t
        period = period_at(t0)
        enabled = period not in (Period.PAUSE, Period.POST, Period.DONE)
        cmds = [c if (c is not None and enabled) else None for c in commands]
        if self._pending:  # latency: the motors act on the command from a few steps ago
            cmds = self._delay(cmds, enabled)
        for r, c in zip(self.robots, cmds):
            r.last_cmd = c or RobotCommand()

        if self.drive is not None:
            self._drive_swerve(cmds)
        else:
            self._drive(cmds, dt)
        if self.collider is not None:
            self.contacts = self.collider.resolve()
        else:
            self._collide_static()
            self._collide_robots()
            self._collide_static()
        fuel_step = (self.step_count + 1) % self._fuel_every == 0
        self._climb(cmds, period, dt)
        self._jams(cmds, dt)
        self._intake(cmds, dt)
        if fuel_step:
            self._push_fuel()
        self._outposts(cmds, dt)
        if self.shooters is not None:
            self.shooters.shoot(cmds, dt)
        else:
            self._shoot(cmds, dt)

        self.step_count += 1
        t1 = round(self.step_count * dt, 9)
        if self.shooters is not None:
            self.shooters.fly(t0, dt)
        else:
            self._resolve_flights(t1)
        self._resolve_hubs(t1)
        if fuel_step:
            self._fuel_physics(dt * self._fuel_every)
        if self.sensors is not None:
            self.sensors.update(dt)
        self._referee(period, dt)
        self.t = t1
        self._transitions(t0, t1)

    def run(self, policy, until: float | None = None) -> MatchSummary:
        """Step until done (or ``until``); ``policy(match) -> list[RobotCommand]``."""
        end = C.FINAL_TIME if until is None else until
        while not self.done and self.t < end - 1e-9:
            self.step(policy(self))
        return self.summary()

    # ------------------------------------------------------------------ robots
    def _delay(self, cmds: list[RobotCommand | None], enabled: bool) -> list[RobotCommand | None]:
        out = []
        for q, c in zip(self._pending, cmds):
            q.append(c)
            late = q.popleft()
            out.append(late if enabled else None)
        return out

    def _drive_swerve(self, cmds: list[RobotCommand | None]) -> None:
        """High fidelity: each robot's code asks for chassis speeds (capped like ``_drive``), then
        the swerve modules and the carpet decide what actually happens."""
        target = np.zeros((len(self.robots), 3))
        active = np.zeros(len(self.robots), dtype=bool)
        for i, (r, c) in enumerate(zip(self.robots, cmds)):
            s = r.spec
            if r.climb_state != ClimbState.GROUND or s.max_speed <= 0:
                r.vx = r.vy = r.omega = 0.0
                self.drive.stop(i)
                continue
            active[i] = True
            if c is None:  # disabled: the drive holds zero speed
                continue
            tvx, tvy = c.vx, c.vy
            cap = min(s.max_speed, s.bump_speed) if self.field.on_bump(r.x, r.y) else s.max_speed
            sp = math.hypot(tvx, tvy)
            if sp > cap:
                tvx, tvy = tvx * cap / sp, tvy * cap / sp
            target[i] = (tvx, tvy, max(-s.max_omega, min(s.max_omega, c.omega)))
        if self.sensors is not None:
            heading = self.sensors.est[:, 2]
        else:
            heading = np.array([r.heading for r in self.robots])
        self.drive.step(self.robots, target, heading, active)

    def _drive(self, cmds: list[RobotCommand | None], dt: float) -> None:
        for r, c in zip(self.robots, cmds):
            s = r.spec
            if r.climb_state != ClimbState.GROUND or s.max_speed <= 0:
                r.vx = r.vy = r.omega = 0.0
                continue
            if c is None:
                tvx = tvy = tom = 0.0
            else:
                tvx, tvy = c.vx, c.vy
                cap = min(s.max_speed, s.bump_speed) if self.field.on_bump(r.x, r.y) else s.max_speed
                sp = math.hypot(tvx, tvy)
                if sp > cap:
                    tvx, tvy = tvx * cap / sp, tvy * cap / sp
                tom = max(-s.max_omega, min(s.max_omega, c.omega))
            dvx, dvy = tvx - r.vx, tvy - r.vy
            dv = math.hypot(dvx, dvy)
            lim = s.max_accel * dt
            if dv > lim:
                dvx, dvy = dvx * lim / dv, dvy * lim / dv
            r.vx += dvx
            r.vy += dvy
            r.omega = tom
            r.x += r.vx * dt
            r.y += r.vy * dt
            r.heading = (r.heading + r.omega * dt + math.pi) % (2 * math.pi) - math.pi

    def _collide_static(self) -> None:
        for r in self.robots:
            rad = r.spec.radius
            if r.x < rad:
                r.x, r.vx = rad, max(r.vx, 0.0)
            elif r.x > C.FIELD_LENGTH - rad:
                r.x, r.vx = C.FIELD_LENGTH - rad, min(r.vx, 0.0)
            if r.y < rad:
                r.y, r.vy = rad, max(r.vy, 0.0)
            elif r.y > C.FIELD_WIDTH - rad:
                r.y, r.vy = C.FIELD_WIDTH - rad, min(r.vy, 0.0)
            if not _near_obstacles(r.x, rad):
                continue
            for x0, x1, y0, y1 in self._robot_boxes[not r.spec.can_trench]:
                if r.x + rad <= x0 or r.x - rad >= x1 or r.y + rad <= y0 or r.y - rad >= y1:
                    continue
                qx, qy = min(max(r.x, x0), x1), min(max(r.y, y0), y1)
                dx, dy = r.x - qx, r.y - qy
                d = math.hypot(dx, dy)
                if d >= rad:
                    continue
                if d > 1e-9:
                    nx, ny, push = dx / d, dy / d, rad - d
                else:  # center inside the box: leave along the shallowest side
                    pens = (r.x - x0, x1 - r.x, r.y - y0, y1 - r.y)
                    side = int(np.argmin(pens))
                    nx, ny = ((-1, 0), (1, 0), (0, -1), (0, 1))[side]
                    push = pens[side] + rad
                r.x += nx * push
                r.y += ny * push
                vn = r.vx * nx + r.vy * ny
                if vn < 0:
                    r.vx -= vn * nx
                    r.vy -= vn * ny

    def _collide_robots(self) -> None:
        self.contacts.clear()
        rs = self.robots
        e = self.cfg.robot_restitution
        for i in range(6):
            ri = rs[i]
            for j in range(i + 1, 6):
                rj = rs[j]
                dx, dy = rj.x - ri.x, rj.y - ri.y
                rr = ri.spec.radius + rj.spec.radius
                if abs(dx) >= rr or abs(dy) >= rr:
                    continue
                d2 = dx * dx + dy * dy
                if d2 >= rr * rr:
                    continue
                d = math.sqrt(d2) or 1e-6
                nx, ny = dx / d, dy / d
                overlap = rr - d
                inv_i = 1.0 / ri.spec.mass if ri.mobile else 0.0
                inv_j = 1.0 / rj.spec.mass if rj.mobile else 0.0
                inv = inv_i + inv_j
                self.contacts.add((i, j))
                if inv == 0.0:
                    continue
                ri.x -= nx * overlap * inv_i / inv
                ri.y -= ny * overlap * inv_i / inv
                rj.x += nx * overlap * inv_j / inv
                rj.y += ny * overlap * inv_j / inv
                rvn = (rj.vx - ri.vx) * nx + (rj.vy - ri.vy) * ny
                if rvn < 0:
                    jimp = -(1 + e) * rvn / inv
                    ri.vx -= jimp * inv_i * nx
                    ri.vy -= jimp * inv_i * ny
                    rj.vx += jimp * inv_j * nx
                    rj.vy += jimp * inv_j * ny

    def _climb(self, cmds: list[RobotCommand | None], period: Period, dt: float) -> None:
        for r, c in zip(self.robots, cmds):
            if c is None:  # disabled: mechanisms freeze
                continue
            s = r.spec
            if r.climb_state == ClimbState.GROUND:
                if (c.climb > 0 and s.climb_level > 0 and r.speed < 0.6
                        and self.field.in_climb_zone(r.alliance, r.x, r.y, r.extent_x)):
                    level = 1 if period == Period.AUTO else min(c.climb, s.climb_level)
                    r.climb_state = ClimbState.CLIMBING
                    r.climb_target = level
                    r.climb_timer = s.climb_duration(level)
                    r.vx = r.vy = 0.0
            elif r.climb_state == ClimbState.CLIMBING:
                r.climb_timer -= dt
                if r.climb_timer <= 0:
                    if self.robot_rngs[r.index].random() < s.climb_success:
                        r.climb_state, r.climb_level = ClimbState.CLIMBED, r.climb_target
                    else:
                        r.climb_state, r.climb_level = ClimbState.GROUND, 0
            elif r.climb_state == ClimbState.CLIMBED:
                if c.climb == 0 and math.hypot(c.vx, c.vy) > 0.5:
                    r.climb_state, r.climb_timer = ClimbState.DESCENDING, 1.5
            elif r.climb_state == ClimbState.DESCENDING:
                r.climb_timer -= dt
                if r.climb_timer <= 0:
                    r.climb_state, r.climb_level = ClimbState.GROUND, 0

    # ------------------------------------------------------------------ FUEL handling
    def _jams(self, cmds: list[RobotCommand | None], dt: float) -> None:
        """Mechanisms jam now and then while in use; a jammed robot can drive but not intake or shoot."""
        for r, c in zip(self.robots, cmds):
            if r.jam_timer > 0:
                r.jam_timer = max(0.0, r.jam_timer - dt)
            elif (c is not None and (c.intake or c.shoot) and r.spec.jam_rate > 0
                  and self.robot_rngs[r.index].random() < r.spec.jam_rate / 60.0 * dt):
                r.jam_timer = r.spec.jam_time

    def _intake(self, cmds: list[RobotCommand | None], dt: float) -> None:
        ground = np.flatnonzero(self.f_state == GROUND)
        if ground.size == 0:
            return
        P = self.f_pos[ground]  # copy: picked-up rows are pushed away to avoid double pickup
        for r, c in zip(self.robots, cmds):
            s = r.spec
            if (c is None or not c.intake or r.climb_state != ClimbState.GROUND or r.jam_timer > 0
                    or r.fuel >= s.capacity or s.intake_rate <= 0):
                r.intake_budget = 0.0
                continue
            r.intake_budget = min(r.intake_budget + s.intake_rate * dt, 1.0 + s.intake_rate * dt)
            n = min(int(r.intake_budget), s.capacity - r.fuel)
            if n <= 0:
                continue
            front = r.front
            reach = front + s.intake_reach + C.FUEL_RADIUS
            rx, ry = P[:, 0] - r.x, P[:, 1] - r.y
            near = (np.abs(rx) < reach) & (np.abs(ry) < reach)
            if not near.any():
                continue
            k = np.flatnonzero(near)
            ch, sh = math.cos(r.heading), math.sin(r.heading)
            fwd = rx[k] * ch + ry[k] * sh
            lat = -rx[k] * sh + ry[k] * ch
            ok = (fwd >= front - 0.25) & (fwd <= reach) & (np.abs(lat) <= s.intake_width / 2)
            k, fwd = k[ok], fwd[ok]
            if k.size == 0:
                continue
            if k.size > n:
                k = k[np.argsort(fwd)[:n]]
            idx = ground[k]
            self.f_state[idx] = HELD
            self.f_owner[idx] = r.index
            P[k] = 1e6
            r.fuel += len(idx)
            r.intake_budget -= len(idx)
            r.stats.fuel_collected += len(idx)

    def _push_fuel(self) -> None:
        g = np.flatnonzero(self.f_state == GROUND)
        if g.size == 0:
            return
        if self.collider is not None:
            collision.push_fuel(self, g)
            return
        P = self.f_pos[g]
        changed = False
        for r in self.robots:
            rr = r.spec.radius + C.FUEL_RADIUS
            dx, dy = P[:, 0] - r.x, P[:, 1] - r.y
            m = (np.abs(dx) < rr) & (np.abs(dy) < rr)
            if not m.any():
                continue
            k = np.flatnonzero(m)
            d = np.hypot(dx[k], dy[k])
            inside = d < rr
            k, d = k[inside], d[inside]
            if k.size == 0:
                continue
            d = np.maximum(d, 1e-6)
            nx, ny = dx[k] / d, dy[k] / d
            P[k, 0] = r.x + nx * (rr + 0.004)
            P[k, 1] = r.y + ny * (rr + 0.004)
            push = np.maximum(r.vx * nx + r.vy * ny, 0.0) * 1.15 + 0.05
            gi = g[k]
            self.f_vel[gi, 0] = nx * push + self.rng.normal(0, 0.05, k.size)
            self.f_vel[gi, 1] = ny * push + self.rng.normal(0, 0.05, k.size)
            self.f_settle[gi] = 3
            changed = True
        if changed:
            self.f_pos[g] = P

    def _outposts(self, cmds: list[RobotCommand | None], dt: float) -> None:
        for a in Alliance:
            target = None
            if self.chute_count[a] > 0:
                for r, c in zip(self.robots, cmds):
                    if (r.alliance == a and c is not None and c.intake and r.climb_state == ClimbState.GROUND
                            and r.fuel < r.spec.capacity
                            and self.field.in_outpost_feed(a, r.x, r.y, r.extent_x)):
                        target = r
                        break
            if target is None:
                self.hp_budget[a] = 0.0
                continue
            self.hp_budget[a] += self.cfg.hp_feed_rate * dt
            while self.hp_budget[a] >= 1.0 and self.chute_count[a] > 0 and target.fuel < target.spec.capacity:
                i = np.flatnonzero((self.f_state == CHUTE) & (self.f_owner == a))[0]
                self.f_state[i] = HELD
                self.f_owner[i] = target.index
                target.fuel += 1
                target.stats.fuel_collected += 1
                self.chute_count[a] -= 1
                self.hp_budget[a] -= 1.0

    def _shoot(self, cmds: list[RobotCommand | None], dt: float) -> None:
        for r, c in zip(self.robots, cmds):
            s = r.spec
            r.g407_cooldown = max(0.0, r.g407_cooldown - dt)
            if (c is None or not c.shoot or r.fuel <= 0 or r.climb_state != ClimbState.GROUND
                    or r.jam_timer > 0 or s.shoot_rate <= 0):
                r.shoot_cooldown = max(0.0, r.shoot_cooldown - dt)
                continue
            hx, hy = self.field.hub_centers[r.alliance]
            dx, dy = hx - r.x, hy - r.y
            dist = math.hypot(dx, dy)
            if not s.min_range <= dist <= s.max_range or not self.can_aim(r, dx, dy):
                r.shoot_cooldown = max(0.0, r.shoot_cooldown - dt)
                continue
            r.shoot_cooldown -= dt
            launched = 0
            while r.shoot_cooldown <= 0 and r.fuel > 0:
                self._launch(r, dist, dx, dy)
                r.shoot_cooldown += 1.0 / s.shoot_rate
                launched += 1
            if launched and r.g407_cooldown <= 0 and not self.field.in_alliance_zone(r.alliance, r.x, r.extent_x):
                self._foul(r, "G407 shot from outside own ALLIANCE ZONE", major=True)
                r.g407_cooldown = 1.0

    @staticmethod
    def can_aim(r: Robot, dx: float, dy: float, heading: float | None = None) -> bool:
        if r.spec.turret:
            return True
        heading = r.heading if heading is None else heading
        err = (math.atan2(dy, dx) - heading + math.pi) % (2 * math.pi) - math.pi
        return abs(err) <= math.radians(15)

    def _launch(self, r: Robot, dist: float, dx: float, dy: float) -> None:
        i = np.flatnonzero((self.f_state == HELD) & (self.f_owner == r.index))[0]
        rng = self.robot_rngs[r.index]
        hit = rng.random() < r.spec.hit_probability(dist, r.speed)
        self.f_state[i] = FLIGHT
        self.f_hit[i] = hit
        self.f_owner[i] = r.alliance
        self.f_shooter[i] = r.index
        self.f_pos[i] = (r.x, r.y)
        self.f_t0[i] = self.t
        self.f_time[i] = self.t + 0.45 + 0.11 * dist + 0.08 * rng.random()
        if not hit:  # lands around the HUB: long or short, off to one side
            ux, uy = dx / dist, dy / dist
            along = rng.choice((-1.0, 1.0)) * rng.uniform(0.85, 1.9)
            side = rng.normal(0.0, 0.7)
            hx, hy = self.field.hub_centers[r.alliance]
            self.f_land[i] = (hx + ux * along - uy * side, hy + uy * along + ux * side)
        r.fuel -= 1
        r.stats.fuel_shot += 1

    def _resolve_flights(self, t1: float) -> None:
        idx = np.flatnonzero((self.f_state == FLIGHT) & (self.f_time <= t1))
        if idx.size == 0:
            return
        hits = idx[self.f_hit[idx]]
        if hits.size:
            lo, hi = C.HUB_PROCESS_TIME
            self.f_state[hits] = HUB
            self.f_time[hits] += self.rng.uniform(lo, hi, hits.size)
        misses = idx[~self.f_hit[idx]]
        if misses.size:
            land = self.f_land[misses]
            land[:, 0] = np.clip(land[:, 0], C.FUEL_RADIUS, C.FIELD_LENGTH - C.FUEL_RADIUS)
            land[:, 1] = np.clip(land[:, 1], C.FUEL_RADIUS, C.FIELD_WIDTH - C.FUEL_RADIUS)
            self.f_state[misses] = GROUND
            self.f_pos[misses] = land
            self.f_vel[misses] = self.rng.normal(0, 0.5, (misses.size, 2))
            self.f_settle[misses] = 3

    def _resolve_hubs(self, t1: float) -> None:
        idx = np.flatnonzero((self.f_state == HUB) & (self.f_time <= t1))
        for i in idx:
            a = Alliance(int(self.f_owner[i]))
            ts = float(self.f_time[i])
            shooter = self.robots[int(self.f_shooter[i])]
            own = shooter.alliance == a  # high-fidelity FUEL can drop into the other HUB
            if self.schedule.counts(a, ts):
                auto = fuel_counts_for_auto(ts)
                if auto:
                    self.scores[a].auto_fuel += 1
                else:
                    self.scores[a].teleop_fuel += 1
                if own:
                    shooter.stats.fuel_scored += 1
                self.score_log.append(ScoreEvent(ts, a, shooter.index, auto))
            else:
                self.scores[a].wasted_fuel += 1
                if own:
                    shooter.stats.fuel_wasted += 1
            exits, sign = self.field.hub_exits[a]
            e = exits[self.rng.integers(len(exits))]
            self.f_state[i] = GROUND
            self.f_pos[i] = e + self.rng.normal(0, 0.03, 2)
            self.f_vel[i] = (sign * self.rng.uniform(*C.HUB_EXIT_SPEED), self.rng.uniform(-1.5, 1.5))
            self.f_settle[i] = 3

    def _fuel_physics(self, dt: float) -> None:
        g = np.flatnonzero(self.f_state == GROUND)
        if g.size == 0:
            return
        V = self.f_vel[g]
        moving = (np.abs(V[:, 0]) + np.abs(V[:, 1]) > 0.03) | (self.f_settle[g] > 0)
        a = g[moving]
        if a.size == 0:
            return
        P, V = self.f_pos[a], self.f_vel[a]
        spd = np.hypot(V[:, 0], V[:, 1])
        V *= (np.maximum(0.0, 1.0 - self.cfg.fuel_friction * dt / np.maximum(spd, 1e-9)))[:, None]
        P += V * dt
        rad, e = C.FUEL_RADIUS, self.cfg.fuel_restitution
        for axis, hi in ((0, C.FIELD_LENGTH - rad), (1, C.FIELD_WIDTH - rad)):
            lo_m, hi_m = P[:, axis] < rad, P[:, axis] > hi
            P[lo_m, axis], V[lo_m, axis] = rad, np.abs(V[lo_m, axis]) * e
            P[hi_m, axis], V[hi_m, axis] = hi, -np.abs(V[hi_m, axis]) * e
        x = P[:, 0]
        tower0, band0, band1, tower1 = _OBSTACLE_X
        near = ((x < tower0 + rad) | (x > tower1 - rad) | ((x > band0[0] - rad) & (x < band0[1] + rad))
                | ((x > band1[0] - rad) & (x < band1[1] + rad)))
        if near.any():
            nk = np.flatnonzero(near)
            Pn, Vn = P[nk], V[nk]
            for x0, x1, y0, y1 in self.field.fuel_obstacle_array:
                cx, cy = np.clip(Pn[:, 0], x0, x1), np.clip(Pn[:, 1], y0, y1)
                dx, dy = Pn[:, 0] - cx, Pn[:, 1] - cy
                d2 = dx * dx + dy * dy
                m = d2 < rad * rad
                if not m.any():
                    continue
                k = np.flatnonzero(m)
                d = np.sqrt(d2[k])
                inside = d < 1e-9
                nx = np.where(inside, np.where(Pn[k, 0] < (x0 + x1) / 2, -1.0, 1.0), dx[k] / np.maximum(d, 1e-9))
                ny = np.where(inside, 0.0, dy[k] / np.maximum(d, 1e-9))
                depth = np.where(inside, np.minimum(Pn[k, 0] - x0, x1 - Pn[k, 0]) + rad, rad - d)
                Pn[k, 0] += nx * depth
                Pn[k, 1] += ny * depth
                vn = Vn[k, 0] * nx + Vn[k, 1] * ny
                refl = np.where(vn < 0, -(1 + e) * vn, 0.0)
                Vn[k, 0] += refl * nx
                Vn[k, 1] += refl * ny
            P[nk], V[nk] = Pn, Vn
        self.f_pos[a], self.f_vel[a] = P, V
        self._separate(a, g)
        self.f_settle[a] = np.maximum(self.f_settle[a] - 1, 0)

    @staticmethod
    def _cells(P: np.ndarray) -> np.ndarray:
        cx = np.clip((P[:, 0] / _CELL).astype(np.int64), 0, _NX - 1) + 1
        cy = np.clip((P[:, 1] / _CELL).astype(np.int64), 0, _NY - 1) + 1
        return cx * _PNY + cy

    def _separate(self, active: np.ndarray, ground: np.ndarray) -> None:
        """Push overlapping FUEL apart (only around balls that moved), via a uniform grid."""
        cid = self._cells(self.f_pos[ground])
        order = np.argsort(cid, kind="stable")
        sc = cid[order]
        rank = np.arange(sc.size) - np.searchsorted(sc, sc, side="left")
        keep = rank < _CELL_CAP
        table = self._cell_table
        table.fill(-1)
        table[sc[keep], rank[keep]] = ground[order][keep]

        Pa = self.f_pos[active]
        cand = table[self._cells(Pa)[:, None] + _NEIGHBOR_OFFSETS[None, :]].reshape(len(active), -1)
        valid = (cand >= 0) & (cand != active[:, None])
        if not valid.any():
            return
        diff = self.f_pos[np.where(valid, cand, 0)] - Pa[:, None, :]
        dist = np.hypot(diff[..., 0], diff[..., 1])
        over = valid & (dist < _CELL - 1e-4)
        if not over.any():
            return
        ai, kj = np.nonzero(over)
        i_idx, j_idx = active[ai], cand[ai, kj]
        dd = dist[ai, kj]
        n = diff[ai, kj] / np.maximum(dd, 1e-9)[:, None]
        same = dd < 1e-9  # exactly on top of each other: push apart in a random direction
        if same.any():
            ang = self.rng.uniform(0.0, 2 * math.pi, int(same.sum()))
            n[same] = np.column_stack([np.cos(ang), np.sin(ang)])
        dd = np.maximum(dd, 1e-6)
        corr = n * ((_CELL - dd) * 0.25)[:, None]
        np.add.at(self.f_pos, i_idx, -corr)
        np.add.at(self.f_pos, j_idx, corr)
        self.f_settle[j_idx] = np.maximum(self.f_settle[j_idx], 2)

    # ------------------------------------------------------------------ referee
    def call_foul(self, robot_index: int, rule: str, major: bool) -> None:
        """Assess a foul the physics doesn't model (used for scripted bots' incidental fouls)."""
        self._foul(self.robots[robot_index], rule, major)

    def _foul(self, r: Robot, rule: str, major: bool) -> None:
        self.fouls.call(self.scores, self.t, r.alliance, r.index, rule, major)
        if major:
            r.stats.fouls_major += 1
        else:
            r.stats.fouls_minor += 1

    def _referee(self, period: Period, dt: float) -> None:
        rs = self.robots
        t = self.t
        if period == Period.AUTO:  # G403: stay on your side of the CENTER LINE in AUTO
            for r in rs:
                if not r.crossed_center_foul and self.field.fully_across_center(r.alliance, r.x, r.extent_x):
                    r.crossed_center_foul = True
                    self._foul(r, "G403 crossed the CENTER LINE in AUTO", major=True)
            for i, j in self.contacts:
                for off, other in ((rs[i], rs[j]), (rs[j], rs[i])):
                    if (off.alliance != other.alliance and (off.index, other.index) not in self._g403_contacts
                            and self.field.fully_across_center(off.alliance, off.x, off.extent_x)):
                        self._g403_contacts.add((off.index, other.index))
                        self._foul(off, "G403 contact across the CENTER LINE in AUTO", major=True)

        # G418 pinning: MINOR after 3 s, then a MAJOR for every further 3 s. A pin is an
        # opponent pushing into a robot that is trying to get away and can't; two robots
        # shoving into each other is not a pin.
        pinned_now = set()
        for i, j in self.contacts:
            for p, v in ((rs[i], rs[j]), (rs[j], rs[i])):
                if p.alliance == v.alliance or not v.mobile:
                    continue
                dx, dy = v.x - p.x, v.y - p.y
                d = math.hypot(dx, dy) or 1e-6
                push = (p.last_cmd.vx * dx + p.last_cmd.vy * dy) / d
                want = math.hypot(v.last_cmd.vx, v.last_cmd.vy)
                escaping = want > 0.8 and (v.last_cmd.vx * dx + v.last_cmd.vy * dy) / (want * d) > -0.3
                if push > 0.5 and escaping and v.speed < 0.35:
                    pinned_now.add((p.index, v.index))
                    self.pin_timer[p.index, v.index] += dt
                    self.pin_last[p.index, v.index] = t
                    n_calls = int(self.pin_timer[p.index, v.index] // C.PIN_LIMIT)
                    while self.pin_fouls[p.index, v.index] < n_calls:
                        major = self.pin_fouls[p.index, v.index] > 0
                        self._foul(p, "G418 pinning", major=major)
                        self.pin_fouls[p.index, v.index] += 1
        for pi, vi in np.argwhere(self.pin_timer > 0):
            if (pi, vi) in pinned_now:
                continue
            p, v = rs[pi], rs[vi]
            apart = math.hypot(p.x - v.x, p.y - v.y) > C.PIN_RELEASE_DISTANCE
            idle = t - self.pin_last[pi, vi]
            if (apart and idle >= C.PIN_LIMIT) or idle >= 5.0:
                self.pin_timer[pi, vi] = 0.0
                self.pin_fouls[pi, vi] = 0

        # G420: in END GAME, no contact with an opponent that is touching its TOWER
        if period == Period.ENDGAME:
            for i, j in self.contacts:
                for off, vic in ((rs[i], rs[j]), (rs[j], rs[i])):
                    if off.alliance == vic.alliance or (off.index, vic.index) in self._g420_contacts:
                        continue
                    on_tower = vic.climb_state in (ClimbState.CLIMBING, ClimbState.CLIMBED)
                    if on_tower or self.field.in_climb_zone(vic.alliance, vic.x, vic.y, vic.extent_x):
                        self._g420_contacts.add((off.index, vic.index))
                        self._foul(off, "G420 contact with a robot at its TOWER in END GAME", major=True)
                        if on_tower:
                            vic.awarded_level3 = True
            for pair in list(self._g420_contacts):
                a, b = sorted(pair)
                if (a, b) not in self.contacts:
                    self._g420_contacts.discard(pair)

    # ------------------------------------------------------------------ period changes
    def _transitions(self, t0: float, t1: float) -> None:
        if t0 < C.AUTO_LEN <= t1:
            for a in Alliance:
                on = [r for r in self.robots if r.alliance == a and r.climb_state == ClimbState.CLIMBED]
                for r in on[: C.AUTO_TOWER_MAX_ROBOTS]:
                    self.scores[a].auto_tower += C.AUTO_TOWER_L1_POINTS
                    r.auto_tower = True
                    r.stats.tower_points += C.AUTO_TOWER_L1_POINTS
        if t0 < C.TELEOP_START <= t1:
            self.schedule.resolve_auto(self.scores[Alliance.BLUE].auto_fuel, self.scores[Alliance.RED].auto_fuel,
                                       float(self.coin_rng.random()))
        if t0 < C.FINAL_TIME <= t1:
            for r in self.robots:
                if r.awarded_level3:
                    pts = C.TELEOP_TOWER_POINTS[3]
                elif r.climb_state == ClimbState.CLIMBED:
                    pts = C.TELEOP_TOWER_POINTS[r.climb_level]
                else:
                    pts = 0
                self.scores[r.alliance].teleop_tower += pts
                r.stats.tower_points += pts
            self.rp = [ranking_points(self.scores[a], self.scores[a.other], self.cfg.event_level) for a in Alliance]
            self.done = True


def _near_obstacles(x: float, rad: float) -> bool:
    """Cheap x-range test: HUB lines and TOWERs are the only static obstacles."""
    band0, band1 = BAND_X
    return (x < C.TOWER_DEPTH + rad or x > C.FIELD_LENGTH - C.TOWER_DEPTH - rad
            or band0[0] - rad < x < band0[1] + rad or band1[0] - rad < x < band1[1] + rad)


def make_match(blue: list[str], red: list[str], seed: int | None = None, config: MatchConfig | None = None) -> Match:
    """Build a match from tier names, e.g. ``make_match(["elite", "low", "low"], ["strong", "mid", "low"])``."""
    return Match([TIERS[n] for n in blue] + [TIERS[n] for n in red], config=config, seed=seed)
