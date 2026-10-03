"""Observation vector for a robot, in its alliance's frame of reference.

Every robot sees the field as if it were on the blue alliance (red observations are
rotated 180 degrees about the field center), so one policy can play either color.
All values are roughly within [-1, 1]. ``OBS_LAYOUT`` documents the slices.

With the high-fidelity physics the robot's own pose and velocity come from its pose
estimate, other robots and FUEL are seen with perception noise, and 8 more values describe
its shooter (docs/03-driving-and-aiming.md).
"""

from __future__ import annotations

import math

import numpy as np

from . import constants as C
from .constants import Alliance
from .controller import MacroController, fuel_density
from .field import FIELD
from .robot import ClimbState, Robot
from .rules import Period
from .sensors import FUEL_SD, OTHER_SD, OTHER_VEL_SD

_PERIODS = (Period.AUTO, Period.PAUSE, Period.TRANSITION, Period.SHIFT1, Period.SHIFT2, Period.SHIFT3,
            Period.SHIFT4, Period.ENDGAME)
GRID = (12, 6)  # FUEL density cells over the field (~1.4 m)
N_NEAREST = 5

OBS_LAYOUT = {
    "match": (0, 17),  # clock, period one-hot, HUB status, time to toggle, AUTO winner, scores (squashed), FUEL counts
    "self": (17, 36),  # pose, velocity, load, legality, climb, jam, HUB distance, robot capabilities
    "robots": (36, 76),  # 2 teammates then 3 opponents, nearest first: 8 values each
    "fuel_grid": (76, 148),  # 12 x 6 FUEL counts / 15
    "nearest_fuel": (148, 158),  # 5 nearest ground FUEL, relative position / 4 m
    "stock": (158, 162),  # own/opp DEPOT FUEL, own/opp OUTPOST chute FUEL (/24)
    "shooter": (162, 170),  # high fidelity only: flywheel, hood, turret (cos, sin), speed and hood ranges
}
OBS_SIZE = 162
OBS_SIZE_HIFI = 170
_HALF_PI = math.pi / 2


def to_frame(alliance: Alliance, x, y):
    """Field coordinates -> this alliance's frame (blue unchanged, red rotated 180 degrees)."""
    if alliance == Alliance.BLUE:
        return x, y
    return C.FIELD_LENGTH - x, C.FIELD_WIDTH - y


def vel_to_frame(alliance: Alliance, vx, vy):
    return (vx, vy) if alliance == Alliance.BLUE else (-vx, -vy)


def heading_to_frame(alliance: Alliance, heading: float) -> float:
    return heading if alliance == Alliance.BLUE else heading + math.pi


def observe(match, ctl: MacroController, index: int) -> np.ndarray:
    r: Robot = match.robots[index]
    a = r.alliance
    hifi = match.hifi is not None
    noise = match.hifi.sensor_noise if hifi else 0.0
    rng = match.obs_rng
    o = np.zeros(OBS_SIZE_HIFI if hifi else OBS_SIZE, dtype=np.float32)
    t = match.t
    period = match.period

    # --- match ---
    o[0] = t / C.FINAL_TIME
    if period in _PERIODS:
        o[1 + _PERIODS.index(period)] = 1.0
    o[9] = float(match.hub_active(a))
    o[10] = float(match.hub_active(a.other))
    ttg = match.time_to_toggle(a)
    o[11] = min(1.2, ttg / C.SHIFT_LEN) if ttg is not None else 1.2
    first = match.schedule.first_inactive
    o[12] = 0.5 if first is None else float(first == a)  # 1 = we won AUTO (our HUB is off in SHIFT 1)
    own, opp = match.scores[a], match.scores[a.other]
    o[13] = _squash(own.total, 200.0)  # 0 -> 0, 200 -> 0.5, stays below 1 for any score
    o[14] = _squash(opp.total, 200.0)
    o[15] = own.fuel / 360.0 if own.fuel < 360 else 1.0 + _squash(own.fuel - 360, 360.0)
    o[16] = opp.fuel / 360.0 if opp.fuel < 360 else 1.0 + _squash(opp.fuel - 360, 360.0)

    # --- self, as the robot's own sensors see it ---
    s = r.spec
    px, py, ph, pvx, pvy = match.perceived(index)
    x, y = to_frame(a, px, py)
    h = heading_to_frame(a, ph)
    vx, vy = vel_to_frame(a, pvx, pvy)
    hx, hy = FIELD.hub_centers[a]
    o[17] = x / C.FIELD_LENGTH
    o[18] = y / C.FIELD_WIDTH
    o[19] = math.cos(h)
    o[20] = math.sin(h)
    o[21] = vx / 5.0
    o[22] = vy / 5.0
    o[23] = r.fuel / 50.0
    o[24] = r.fuel / max(1, s.capacity)
    o[25] = float(FIELD.in_alliance_zone(a, px, r.extent_x))
    o[26] = float(ctl.can_shoot_here(r, px, py) and match.can_aim(r, hx - px, hy - py, ph))
    o[27] = {ClimbState.GROUND: 0.0, ClimbState.CLIMBING: 0.5, ClimbState.CLIMBED: 1.0,
             ClimbState.DESCENDING: 0.5}[r.climb_state]
    o[28] = float(r.jam_timer > 0)
    o[29] = math.hypot(hx - px, hy - py) / 8.0
    o[30] = s.capacity / 50.0
    o[31] = float(s.can_trench)
    o[32] = float(s.turret)
    o[33] = s.climb_level / 3.0
    o[34] = s.shoot_rate / 8.0
    o[35] = s.max_speed / 5.0

    # --- other robots: teammates first, then opponents, each nearest first ---
    others = sorted((q for q in match.robots if q is not r), key=lambda q: (q.alliance != a, math.hypot(q.x - r.x, q.y - r.y)))
    k = 36
    for q in others:
        qx, qy = to_frame(a, q.x, q.y)
        qvx, qvy = vel_to_frame(a, q.vx, q.vy)
        if noise > 0:
            ex, ey, evx, evy = rng.normal(0.0, 1.0, 4)
            qx, qy = qx + ex * OTHER_SD * noise, qy + ey * OTHER_SD * noise
            qvx, qvy = qvx + evx * OTHER_VEL_SD * noise, qvy + evy * OTHER_VEL_SD * noise
        o[k:k + 8] = (
            (qx - x) / C.FIELD_LENGTH, (qy - y) / C.FIELD_WIDTH, qvx / 5.0, qvy / 5.0, q.fuel / 50.0,
            float(q.climb_state != ClimbState.GROUND), float(q.mobile), math.hypot(qx - x, qy - y) / 10.0,
        )
        k += 8

    # --- FUEL on the floor ---
    g = match.ground_fuel()
    if g.size:
        gx, gy = to_frame(a, g[:, 0], g[:, 1])
        pts = np.column_stack([gx, gy])
        grid = fuel_density(pts, GRID)
        o[76:148] = np.minimum(grid.ravel() / 15.0, 1.0)
        if noise > 0:  # where object detection puts each ball
            gx = gx + rng.normal(0.0, FUEL_SD * noise, gx.size)
            gy = gy + rng.normal(0.0, FUEL_SD * noise, gy.size)
        d = np.hypot(gx - x, gy - y)
        n = min(N_NEAREST, d.size)
        near = np.argpartition(d, n - 1)[:n] if d.size > n else np.arange(d.size)
        near = near[np.argsort(d[near])]
        rel = np.clip(np.column_stack([gx[near] - x, gy[near] - y]) / 4.0, -1.0, 1.0)
        o[148:148 + 2 * n] = rel.ravel()

    # --- stock waiting at the walls ---
    o[158] = match.depot_count(a) / 24.0
    o[159] = match.depot_count(a.other) / 24.0
    o[160] = match.chute_count[a] / 24.0
    o[161] = match.chute_count[a.other] / 24.0

    # --- shooter (high fidelity): state of the mechanism and what it can do ---
    if hifi:
        sh = s.shooter
        o[162] = r.flywheel / 15.0
        o[163] = r.hood / _HALF_PI
        o[164] = math.cos(r.turret)
        o[165] = math.sin(r.turret)
        o[166] = sh.speed_range[0] / 15.0
        o[167] = sh.speed_range[1] / 15.0
        o[168] = sh.hood_range[0] / _HALF_PI
        o[169] = sh.hood_range[1] / _HALF_PI
    return np.clip(o, -2.0, 2.0, out=o)  # the declared observation space is [-2, 2]


def _squash(v: float, scale: float) -> float:
    return v / (v + scale) if v > 0 else 0.0
