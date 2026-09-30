"""FUEL ballistics and shooter mechanisms (high fidelity).

FUEL leaves a shooter at the exit speed, hood elevation and aim direction the mechanism is at
(plus shot-to-shot spread), on top of the robot's own velocity. In flight it falls under gravity
and slows with air drag (``constants.FUEL_DRAG``). It scores by dropping through the hexagonal
opening at the top of a HUB, 72 in up; it bounces off the HUB's rim and sides, the field walls
and the carpet, and then rolls. The mechanisms lag: the flywheel spins up with a time constant
and loses a little speed with every shot, the hood and turret move at limited rates.

A robot's *aim software* (used by the scripted bots and the ``macro`` and ``continuous`` action
modes) works like a team's shooter code. A shot table, computed from this same flight model,
maps distance to the most forgiving hood angle and the exit speed for it. Robots whose shooter
spec says ``lead`` aim at a virtual target that cancels their own motion. It fires only once
flywheel, hood and aim are on target, all judged from where the robot *believes* it is. With
``RobotCommand.shot_speed`` set (the ``continuous_aim`` action mode) the driver sets exit speed,
hood and turret directly and every shot fires as commanded.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

from . import constants as C
from .constants import FLIGHT, GROUND, HELD, HUB, Alliance
from .field import FIELD
from .robot import ClimbState, Robot, ShooterSpec

H = C.HUB_OPENING_HEIGHT
# FUEL whose center drops through this hexagon goes in (a ball clipping the rim counts as in when
# its center is inside the rim by at least half a radius)
APOTHEM = C.HUB_OPENING_ACROSS / 2 - C.FUEL_RADIUS / 2
_HEX = ((1.0, 0.0), (0.5, math.sqrt(3) / 2), (-0.5, math.sqrt(3) / 2))  # normals of the three pairs of flats
EDGE = C.HUB_HALF + C.FUEL_RADIUS  # FUEL this close to a HUB's center (face-on) touches its side
RIM_BOUNCE = 0.35
SIDE_BOUNCE = 0.3
WALL_BOUNCE = 0.35
CARPET_BOUNCE = 0.45
LAND_SPEED = 1.5  # m/s: FUEL hitting the carpet slower than this stops bouncing and rolls
# The aim software starts a volley once the flywheel, hood and aim are on target, then keeps
# feeding FUEL while they stay roughly there (each shot takes some flywheel speed out).
SPEED_TOLERANCE = (0.02, 0.06)  # flywheel within this fraction of the target speed: (start, keep firing)
HOOD_TOLERANCE = (0.015, 0.05)  # rad
AIM_SLACK = 2.0  # keep firing while the aim error is within this many times the tier's aim tolerance


def in_opening(rx, ry):
    """Mask: FUEL crossing the opening's height at (rx, ry) from a HUB's center drops in."""
    ok = np.ones(np.shape(rx), dtype=bool)
    for nx, ny in _HEX:
        ok &= np.abs(rx * nx + ry * ny) <= APOTHEM
    return ok


def flight_step(pos, z, vel, vz, dt: float) -> None:
    """Advance FUEL in the air by one step (semi-implicit Euler: gravity and quadratic drag).
    Updates in place: ``pos``/``vel`` (k, 2) horizontal, ``z``/``vz`` (k,) vertical."""
    speed = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2 + vz ** 2)
    keep = 1.0 - C.FUEL_DRAG * speed * dt
    vel *= keep[:, None]
    vz *= keep
    vz -= C.GRAVITY * dt
    pos += vel * dt
    z += vz * dt


# --------------------------------------------------------------------------- aim solutions
D_STEP = 0.05
DISTANCES = np.arange(0.5, 10.0 + 1e-9, D_STEP)  # m from the shooter to the HUB center
_D0, _K = float(DISTANCES[0]), len(DISTANCES)


@dataclass
class ShotTable:
    """Aim solutions for one shooter: for each distance, the most forgiving hood angle and the
    exit speed that drops FUEL through the middle of the HUB opening."""

    hoods: np.ndarray  # (J,) rad
    speed: np.ndarray  # (J, K) m/s for DISTANCES; nan where that hood can't make the distance
    time: np.ndarray  # (J, K) s from launch to the opening
    spread: np.ndarray  # (J, K) m, expected miss distance along the shot from the shooter's spreads
    best: np.ndarray  # (K,) index of the most forgiving hood, -1 where no hood can make it

    def __post_init__(self) -> None:  # plain lists: much faster than numpy for single lookups
        self._best = [int(j) for j in self.best]
        self._hoods = [float(h) for h in self.hoods]
        self._speed = self.speed.tolist()
        self._time = self.time.tolist()

    def solve(self, d: float) -> tuple[float, float, float] | None:
        """(hood, exit speed, flight time) for a stationary shot from ``d`` m, or None."""
        x = (d - _D0) / D_STEP
        if not 0.0 <= x <= _K - 1:
            return None
        j = self._best[int(x + 0.5)]
        if j < 0:
            return None
        k0 = min(int(x), _K - 2)
        f = x - k0
        speed, time = self._speed[j], self._time[j]
        v0, v1 = speed[k0], speed[k0 + 1]
        if v0 != v0 or v1 != v1:  # nan: at the edge of this hood's reach, use the nearest point
            k = k0 + int(f + 0.5)
            if speed[k] != speed[k]:
                return None
            return self._hoods[j], speed[k], time[k]
        t0, t1 = time[k0], time[k0 + 1]
        return self._hoods[j], v0 + (v1 - v0) * f, t0 + (t1 - t0) * f

    def reach(self) -> tuple[float, float]:
        """Shortest and longest distance this shooter can score from."""
        ok = np.flatnonzero(self.best >= 0)
        return (float(DISTANCES[ok[0]]), float(DISTANCES[ok[-1]])) if ok.size else (math.nan, math.nan)


@functools.lru_cache(maxsize=None)
def shot_table(sh: ShooterSpec, dt: float) -> ShotTable:
    """Fly every (hood, exit speed) pair with the match's own flight model and step, then invert:
    for each distance and hood, the speed that crosses the opening's height (coming down) right
    over the HUB center, clearing the HUB's side on the way. Among the hoods that can, the best
    one is the least sensitive to the shooter's speed and angle spreads."""
    lo, hi = sh.hood_range
    n = int(round((hi - lo) / 0.025)) + 1
    hoods = np.linspace(lo, hi, n) if n > 1 else np.array([lo])
    step = 0.02 if n == 1 else float(hoods[1] - hoods[0])
    flown = np.concatenate([[hoods[0] - step], hoods, [hoods[-1] + step]])  # neighbors: angle sensitivity
    speeds = np.arange(2.5, 16.0, 0.02)
    th, v = np.meshgrid(flown, speeds, indexing="ij")
    x = np.zeros_like(th)
    z = np.full_like(th, sh.height)
    vx, vz = v * np.cos(th), v * np.sin(th)
    rise = np.full_like(th, np.nan)  # where the ball first climbs past the rim (height H + radius)
    cross = np.full_like(th, np.nan)  # where it comes back down through height H
    tcross = np.full_like(th, np.nan)
    alive = np.ones_like(th, dtype=bool)
    top = H + C.FUEL_RADIUS
    for k in range(int(round(5.0 / dt))):
        speed = np.sqrt(vx * vx + vz * vz)
        keep = 1.0 - C.FUEL_DRAG * speed * dt
        vx *= keep
        vz = vz * keep - C.GRAVITY * dt
        x0, z0 = x, z
        x, z = x + vx * dt, z + vz * dt
        up = np.isnan(rise) & (z0 < top) & (z >= top)
        rise = np.where(up, x0 + (x - x0) * (top - z0) / np.where(up, z - z0, 1.0), rise)
        down = alive & (z0 > H) & (z <= H)
        frac = (z0 - H) / np.where(down, z0 - z, 1.0)
        cross = np.where(down, x0 + (x - x0) * frac, cross)
        tcross = np.where(down, (k + frac) * dt, tcross)
        alive &= ~down & (z > 0.0)
        if not alive.any():
            break
    J, K = len(flown), len(DISTANCES)
    spd = np.full((J, K), np.nan)
    tim = np.full((J, K), np.nan)
    clear = np.zeros((J, K), dtype=bool)
    for j in range(J):
        ok = ~np.isnan(cross[j]) & ~np.isnan(rise[j])
        if ok.sum() < 2:
            continue
        xs, vs = cross[j][ok], speeds[ok]
        order = np.argsort(xs)  # crossing distance grows with speed; sort to be safe
        xs, vs = xs[order], vs[order]
        inside = (DISTANCES >= xs[0]) & (DISTANCES <= xs[-1])
        spd[j, inside] = np.interp(DISTANCES[inside], xs, vs)
        tim[j, inside] = np.interp(DISTANCES[inside], xs, tcross[j][ok][order])
        up_at = np.interp(DISTANCES[inside], xs, rise[j][ok][order])
        clear[j, inside] = DISTANCES[inside] - up_at >= EDGE
    # sensitivity of the crossing point to the shooter's spreads, per hood and distance
    dv_dd = np.gradient(spd, D_STEP, axis=1)
    dx_dv = 1.0 / np.where(np.abs(dv_dd) > 1e-6, dv_dd, np.nan)
    spread = np.full((J, K), np.nan)
    for j in range(1, J - 1):
        near = []
        for jj in (j - 1, j + 1):  # crossing distance of the neighboring hoods at this hood's speeds
            ok = ~np.isnan(cross[jj])
            near.append(np.interp(spd[j], speeds[ok], cross[jj][ok], left=np.nan, right=np.nan)
                        if ok.sum() >= 2 else np.full(K, np.nan))
        dx_dth = (near[1] - near[0]) / (flown[j + 1] - flown[j - 1])
        spread[j] = np.hypot(dx_dv[j] * spd[j] * sh.speed_sd, dx_dth * sh.angle_sd)
    usable = clear & (spd >= sh.speed_range[0]) & (spd <= sh.speed_range[1]) & ~np.isnan(spread)
    usable[0] = usable[-1] = False  # the neighbor hoods are outside the hood's range
    score = np.where(usable, spread, np.inf)
    best = np.where(np.isfinite(score).any(axis=0), np.argmin(score, axis=0) - 1, -1)
    keep = slice(1, J - 1)
    return ShotTable(hoods=hoods, speed=np.where(usable, spd, np.nan)[keep], time=tim[keep],
                     spread=np.where(usable, spread, np.nan)[keep], best=best)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Shooters:
    """Shooter mechanisms of all robots in a match, and the FUEL they put in the air."""

    def __init__(self, match) -> None:
        self.match = match
        self.tables = [shot_table(r.spec.shooter, match.cfg.dt) if r.spec.shoot_rate > 0 else None
                       for r in match.robots]
        self.targets = np.zeros((len(match.robots), 3))  # flywheel speed, hood, turret the robot is going for
        self.volley = [False] * len(match.robots)  # firing: the looser keep-firing tolerances apply
        for r in match.robots:
            r.hood = r.spec.shooter.hood_range[0]
            self.targets[r.index, 1] = r.hood

    def can_reach(self, r: Robot, distance: float) -> bool:
        table = self.tables[r.index]
        return table is not None and table.solve(distance) is not None

    def aim(self, r: Robot) -> tuple[float, float, float, float] | None:
        """The aim software: (exit speed, hood, field yaw to shoot along, believed heading) to hit
        the robot's own HUB from where it believes it is, or None if it can't from here."""
        x, y, h, vx, vy = self.match.perceived(r.index)
        hx, hy = FIELD.hub_centers[r.alliance]
        dx, dy = hx - x, hy - y
        table = self.tables[r.index]
        if r.spec.shooter.lead:  # aim where the HUB will be relative to the robot when the shot arrives
            for _ in range(3):
                sol = table.solve(math.hypot(dx, dy))
                if sol is None:
                    break
                dx, dy = hx - x - vx * sol[2], hy - y - vy * sol[2]
        sol = table.solve(math.hypot(dx, dy))
        if sol is None:
            return None
        return sol[1], sol[0], math.atan2(dy, dx), h

    # ---------------------------------------------------------------- each step
    def shoot(self, cmds, dt: float) -> None:
        m = self.match
        for r, c in zip(m.robots, cmds):
            s, sh = r.spec, r.spec.shooter
            r.g407_cooldown = max(0.0, r.g407_cooldown - dt)
            if s.shoot_rate <= 0:
                continue
            tgt = self.targets[r.index]
            aim_err = 0.0
            manual = c is not None and c.shot_speed is not None
            if manual:  # the driver sets the shooter
                tgt[0] = min(max(c.shot_speed, sh.speed_range[0]), sh.speed_range[1])
                if c.hood is not None:
                    tgt[1] = min(max(c.hood, sh.hood_range[0]), sh.hood_range[1])
                tgt[2] = max(-sh.turret_range, min(sh.turret_range, c.turret or 0.0)) if s.turret else 0.0
            else:
                sol = self.aim(r) if r.fuel > 0 else None
                if sol is not None:
                    speed, hood, yaw, h = sol
                    tgt[0], tgt[1] = speed, hood
                    if s.turret:
                        tgt[2] = max(-sh.turret_range, min(sh.turret_range, _wrap(yaw - h)))
            # the mechanisms move toward their targets
            r.flywheel += (tgt[0] - r.flywheel) * (1.0 - math.exp(-dt / sh.spinup))
            r.hood += max(-sh.hood_rate * dt, min(sh.hood_rate * dt, tgt[1] - r.hood))
            if s.turret:
                r.turret += max(-sh.turret_rate * dt, min(sh.turret_rate * dt, tgt[2] - r.turret))
            if manual:
                ready = True
            else:
                ready = sol is not None
                if ready:
                    k = int(self.volley[r.index])
                    aim_err = _wrap(yaw - (h + r.turret))
                    ready = (abs(r.flywheel - tgt[0]) <= SPEED_TOLERANCE[k] * tgt[0]
                             and abs(r.hood - tgt[1]) <= HOOD_TOLERANCE[k]
                             and abs(aim_err) <= sh.aim_tolerance * (AIM_SLACK if k else 1.0))
            r.shot_ready = ready
            if (c is None or not c.shoot or not ready or r.fuel <= 0 or r.climb_state != ClimbState.GROUND
                    or r.jam_timer > 0):
                self.volley[r.index] = False
                r.shoot_cooldown = max(0.0, r.shoot_cooldown - dt)
                continue
            self.volley[r.index] = True
            r.shoot_cooldown -= dt
            launched = 0
            while r.shoot_cooldown <= 0 and r.fuel > 0:
                self._launch(r)
                r.shoot_cooldown += 1.0 / s.shoot_rate
                launched += 1
            if launched and r.g407_cooldown <= 0 and not FIELD.in_alliance_zone(r.alliance, r.x, r.extent_x):
                m._foul(r, "G407 shot from outside own ALLIANCE ZONE", major=True)
                r.g407_cooldown = 1.0

    def _launch(self, r: Robot) -> None:
        m = self.match
        sh = r.spec.shooter
        rng = m.robot_rngs[r.index]
        i = np.flatnonzero((m.f_state == HELD) & (m.f_owner == r.index))[0]
        speed = r.flywheel * (1.0 + rng.normal(0.0, sh.speed_sd))
        elev = r.hood + rng.normal(0.0, sh.angle_sd)
        yaw = r.heading + r.turret + rng.normal(0.0, sh.angle_sd)
        horiz = speed * math.cos(elev)
        m.f_state[i] = FLIGHT
        m.f_owner[i] = r.alliance
        m.f_shooter[i] = r.index
        m.f_pos[i] = (r.x, r.y)
        m.f_vel[i] = (horiz * math.cos(yaw) + r.vx, horiz * math.sin(yaw) + r.vy)
        m.f_z[i] = sh.height
        m.f_vz[i] = speed * math.sin(elev)
        m.f_t0[i] = m.t
        r.flywheel *= 1.0 - sh.recovery
        r.fuel -= 1
        r.stats.fuel_shot += 1

    def fly(self, t0: float, dt: float) -> None:
        """Move FUEL in the air one step: into a HUB, off its rim or sides, off the walls, onto the carpet."""
        m = self.match
        idx = np.flatnonzero(m.f_state == FLIGHT)
        if idx.size == 0:
            return
        P, Z, V, VZ = m.f_pos[idx], m.f_z[idx], m.f_vel[idx], m.f_vz[idx]  # copies
        P0, Z0 = P.copy(), Z.copy()
        flight_step(P, Z, V, VZ, dt)
        rb = C.FUEL_RADIUS
        done = np.zeros(idx.size, dtype=bool)
        down = (Z0 > H) & (Z <= H)
        if down.any():  # coming down through the height of the HUB tops
            frac = np.where(down, (Z0 - H) / np.where(down, Z0 - Z, 1.0), 0.0)
            cx = P0[:, 0] + (P[:, 0] - P0[:, 0]) * frac
            cy = P0[:, 1] + (P[:, 1] - P0[:, 1]) * frac
            for a in Alliance:
                hx, hy = FIELD.hub_centers[a]
                rx, ry = cx - hx, cy - hy
                over = down & (np.abs(rx) <= C.HUB_HALF) & (np.abs(ry) <= C.HUB_HALF)
                if not over.any():
                    continue
                inside = over & in_opening(rx, ry)
                if inside.any():
                    k = np.flatnonzero(inside)
                    lo, hi = C.HUB_PROCESS_TIME
                    m.f_state[idx[k]] = HUB
                    m.f_owner[idx[k]] = a
                    m.f_time[idx[k]] = t0 + frac[k] * dt + m.rng.uniform(lo, hi, k.size)
                    done[k] = True
                rim = over & ~inside
                if rim.any():  # hits the top edge around the opening: bounces up and off, outward
                    k = np.flatnonzero(rim)
                    d = np.hypot(rx[k], ry[k]) + 1e-9
                    Z[k] = 2 * H - Z[k]
                    VZ[k] = -VZ[k] * RIM_BOUNCE
                    V[k, 0] = V[k, 0] * 0.5 + rx[k] / d
                    V[k, 1] = V[k, 1] * 0.5 + ry[k] / d
        for a in Alliance:  # below the top: the HUB's sides
            hx, hy = FIELD.hub_centers[a]
            rx, ry = P[:, 0] - hx, P[:, 1] - hy
            hit = ~done & (Z < H) & (np.abs(rx) < EDGE) & (np.abs(ry) < EDGE)
            if not hit.any():
                continue
            k = np.flatnonzero(hit)
            ox, oy = P0[k, 0] - hx, P0[k, 1] - hy
            xface = (EDGE - np.abs(rx[k])) < (EDGE - np.abs(ry[k]))
            sx, sy = np.where(ox >= 0, 1.0, -1.0), np.where(oy >= 0, 1.0, -1.0)
            P[k, 0] = np.where(xface, hx + sx * EDGE, P[k, 0])
            P[k, 1] = np.where(xface, P[k, 1], hy + sy * EDGE)
            V[k, 0] = np.where(xface, sx * np.abs(V[k, 0]) * SIDE_BOUNCE, V[k, 0])
            V[k, 1] = np.where(xface, V[k, 1], sy * np.abs(V[k, 1]) * SIDE_BOUNCE)
        for axis, top in ((0, C.FIELD_LENGTH - rb), (1, C.FIELD_WIDTH - rb)):  # field walls
            lo_m, hi_m = P[:, axis] < rb, P[:, axis] > top
            P[lo_m, axis], V[lo_m, axis] = rb, np.abs(V[lo_m, axis]) * WALL_BOUNCE
            P[hi_m, axis], V[hi_m, axis] = top, -np.abs(V[hi_m, axis]) * WALL_BOUNCE
        low = ~done & (Z <= rb)
        bounce = low & (VZ < -LAND_SPEED)
        Z[bounce] = rb
        VZ[bounce] *= -CARPET_BOUNCE
        V[bounce] *= 0.8
        land = low & ~bounce
        if land.any():  # rolls from here on
            k = np.flatnonzero(land)
            gi = idx[k]
            m.f_state[gi] = GROUND
            m.f_pos[gi] = P[k]
            m.f_vel[gi] = V[k] * 0.8
            m.f_z[gi] = 0.0
            m.f_vz[gi] = 0.0
            m.f_settle[gi] = 3
            done[k] = True
        fly = ~done
        m.f_pos[idx[fly]] = P[fly]
        m.f_vel[idx[fly]] = V[fly]
        m.f_z[idx[fly]] = Z[fly]
        m.f_vz[idx[fly]] = VZ[fly]
