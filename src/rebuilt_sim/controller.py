"""Macro actions -> drive commands: navigation, FUEL targeting, shooting spots, defense, climbing.

Scripted bots and the RL environment's ``macro`` action mode both go through
``MacroController`` so a learned policy picks *what* to do and this layer does the driving.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from . import constants as C
from .collision import aabb_box, box_contact, half_extents, robot_box
from .constants import Alliance
from .field import BAND_X, BUMP_PORTAL_Y, FIELD, TRENCH_PORTAL_Y
from .robot import ClimbState, Robot, RobotCommand
from .rules import Period


class Macro(IntEnum):
    IDLE = 0
    COLLECT = 1  # drive to the best FUEL cluster with the intake on
    COLLECT_DEPOT = 2
    COLLECT_OUTPOST = 3  # human player feeds through the chute
    SHOOT = 4  # go to a legal shooting spot and fire
    STAGE = 5  # wait at a shooting spot without firing (HUB about to turn on)
    DEFEND = 6  # push the most dangerous opponent
    CLIMB = 7


N_MACROS = len(Macro)
_GRID = (34, 17)  # FUEL density cells (~0.49 m) used for targeting
_BUMP_COST = 2.0  # extra meters charged for going over a BUMP instead of under a TRENCH


def region(x: float) -> int:
    """0 blue zone, 1 blue HUB line, 2 neutral, 3 red HUB line, 4 red zone."""
    if x < BAND_X[0][0]:
        return 0
    if x <= BAND_X[0][1]:
        return 1
    if x < BAND_X[1][0]:
        return 2
    if x <= BAND_X[1][1]:
        return 3
    return 4


def _portals(can_trench: bool) -> list[tuple[float, bool]]:
    ports = [(y, True) for y in BUMP_PORTAL_Y]
    if can_trench:
        ports += [(y, False) for y in TRENCH_PORTAL_Y]
    return ports


def _segment_hits_box(x0: float, y0: float, x1: float, y1: float,
                      bx0: float, bx1: float, by0: float, by1: float) -> bool:
    """Slab test: does the segment (x0, y0)-(x1, y1) pass through the box?"""
    t0, t1 = 0.0, 1.0
    for p, d, lo, hi in ((x0, x1 - x0, bx0, bx1), (y0, y1 - y0, by0, by1)):
        if abs(d) < 1e-12:
            if p < lo or p > hi:
                return False
            continue
        ta, tb = (lo - p) / d, (hi - p) / d
        if ta > tb:
            ta, tb = tb, ta
        t0, t1 = max(t0, ta), min(t1, tb)
        if t0 > t1:
            return False
    return True


def _around_tower(r: Robot, gx: float, gy: float) -> tuple[float, float] | None:
    """Waypoint around a TOWER that sits between the robot and its goal (e.g. DEPOT <-> OUTPOST).

    Only triggers when the robot would actually run into the TOWER's front (its UPRIGHTS and
    RUNGS), so goals right against it (the climbing positions) are still reached directly, and a
    robot hugging the alliance wall passes behind the UPRIGHTS.
    """
    rad = r.spec.radius
    m = rad - 0.05
    too_tall = FIELD.clearance(r.spec) >= 3  # can't pass under the supports behind the UPRIGHTS
    for a in Alliance:
        t = (FIELD.towers if too_tall else FIELD.tower_fronts)[a]
        box = (t.x0 - m, t.x1 + m, t.y0 - m, t.y1 + m)
        if box[0] <= gx <= box[1] and box[2] <= gy <= box[3]:
            continue  # the goal is at the TOWER itself: no way around leads there
        if not _segment_hits_box(r.x, r.y, gx, gy, *box):
            continue
        front = t.x1 + rad + 0.25 if a == Alliance.BLUE else t.x0 - rad - 0.25
        below, above = t.y0 - rad - 0.25, t.y1 + rad + 0.25
        near, far = (below, above) if r.y < (t.y0 + t.y1) / 2 else (above, below)
        if not _segment_hits_box(r.x, r.y, front, far, t.x0 - m, t.x1 + m, t.y0 - m, t.y1 + m):
            return front, far
        return front, near
    return None


def _lane_blocked(py: float, xa: float, xb: float, rad: float, parked) -> bool:
    """True if a robot that can't move sits in the crossing lane at height ``py``."""
    lo, hi = min(xa, xb), max(xa, xb)
    for ox, oy, orad in parked:
        cx = min(max(ox, lo), hi)
        if math.hypot(ox - cx, oy - py) < orad + rad:
            return True
    return False


def next_waypoint(r: Robot, gx: float, gy: float, parked=()) -> tuple[float, float, bool]:
    """Next point to drive at on the way to (gx, gy); the bool is True when it is the goal itself.

    ``parked`` lists (x, y, radius) of robots that can't move (dead or on a TOWER); crossing
    lanes they block are avoided.
    """
    rad = r.spec.radius
    rr, gr = region(r.x), region(gx)
    if rr in (1, 3):  # inside a crossing: keep going through it toward the goal
        band = BAND_X[0 if rr == 1 else 1]
        ys = [y for y, _ in _portals(r.spec.can_trench)]
        py = min(ys, key=lambda y: abs(y - r.y))
        ex = band[1] + rad + 0.15 if gx > r.x else band[0] - rad - 0.15
        return ex, py, False
    lo, hi = min(rr, gr), max(rr, gr)
    crossings = [b for b in (1, 3) if lo < b < hi]
    if not crossings:
        detour = _around_tower(r, gx, gy)
        if detour is not None:
            return detour[0], detour[1], False
        return gx, gy, True
    b = crossings[0] if gr > rr else crossings[-1]
    band = BAND_X[0 if b == 1 else 1]
    going_up = gr > rr
    entry_x = band[0] - rad - 0.15 if going_up else band[1] + rad + 0.15
    exit_x = band[1] + rad + 0.15 if going_up else band[0] - rad - 0.15
    best = None
    for py, is_bump in _portals(r.spec.can_trench):
        cost = math.hypot(entry_x - r.x, py - r.y) + math.hypot(gx - exit_x, gy - py) + (_BUMP_COST if is_bump else 0.0)
        if _lane_blocked(py, entry_x, exit_x, rad, parked):
            cost += 100.0
        if best is None or cost < best[0]:
            best = (cost, py)
    py = best[1]
    # Lined up once level with the lane and anywhere past the entry point: go straight through.
    past_entry = r.x >= entry_x - 0.45 if going_up else r.x <= entry_x + 0.45
    if abs(r.y - py) < 0.35 and past_entry:
        return exit_x, py, False
    return entry_x, py, False


def path_length(r: Robot, gx: float, gy: float) -> float:
    """Rough driving distance to a goal including detours through crossings."""
    rr, gr = region(r.x), region(gx)
    n_cross = sum(1 for b in (1, 3) if min(rr, gr) < b < max(rr, gr))
    return math.hypot(gx - r.x, gy - r.y) + 1.5 * n_cross


def _angle_err(target: float, heading: float) -> float:
    return (target - heading + math.pi) % (2 * math.pi) - math.pi


def fuel_density(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Count FUEL per cell on an (nx, ny) grid laid over the whole field."""
    nx, ny = shape
    if points.size == 0:
        return np.zeros(shape)
    ix = np.clip((points[:, 0] * (nx / C.FIELD_LENGTH)).astype(np.int64), 0, nx - 1)
    iy = np.clip((points[:, 1] * (ny / C.FIELD_WIDTH)).astype(np.int64), 0, ny - 1)
    return np.bincount(ix * ny + iy, minlength=nx * ny).reshape(nx, ny).astype(np.float64)


# cell centers and HUB-line regions of the targeting grid, computed once
_CX = (np.arange(_GRID[0]) + 0.5) * C.FIELD_LENGTH / _GRID[0]
_CY = (np.arange(_GRID[1]) + 0.5) * C.FIELD_WIDTH / _GRID[1]
_GX, _GY = np.meshgrid(_CX, _CY, indexing="ij")
_GREG = np.digitize(_GX, [BAND_X[0][0], BAND_X[0][1], BAND_X[1][0], BAND_X[1][1]])
_TRENCH_BOXES = [pair[a] for pair in FIELD.trench_openings for a in Alliance]


def _in_trench(x, y, margin: float = 0.1):
    """Mask of points under a TRENCH arm (unreachable for robots too tall to drive under it)."""
    inside = np.zeros(np.shape(x), dtype=bool)
    for b in _TRENCH_BOXES:
        inside |= (x > b.x0 - margin) & (x < b.x1 + margin) & (y > b.y0 - margin) & (y < b.y1 + margin)
    return inside


_GTRENCH = _in_trench(_GX, _GY, margin=0.0)


def _in_tower(x, y):
    """Mask of points inside a TOWER's footprint: FUEL rolls in under the RUNGS, but the UPRIGHTS
    are closer together than a robot is wide, so the bots leave it there."""
    inside = np.zeros(np.shape(x), dtype=bool)
    for b in FIELD.towers:
        inside |= (x > b.x0) & (x < b.x1) & (y > b.y0) & (y < b.y1)
    return inside


_GTOWER = _in_tower(_GX, _GY)


@dataclass
class RobotMemory:
    macro: Macro = Macro.IDLE
    since: float = 0.0
    target: tuple[float, float] | None = None
    target_t: float = -1.0
    backoff_until: float = -1.0
    contact_time: float = 0.0
    sweep_phase: float = 0.0
    stall: float = 0.0  # seconds spent pushing without moving
    unstick_until: float = -1.0
    unstick_vel: tuple[float, float] = (0.0, 0.0)
    unstick_side: float = 1.0
    avoid: tuple[float, float, float] | None = None  # (x, y, until): a FUEL target that proved unreachable
    notes: dict = field(default_factory=dict)


class MacroController:
    """Turns one macro per robot into a ``RobotCommand`` every call."""

    def __init__(self, match) -> None:
        self.match = match
        self.mem = [RobotMemory() for _ in match.robots]
        self._grid_step = -1
        self._grid = None
        self._ground = None

    # ---------------------------------------------------------------- shared caches
    def _fuel_grid(self) -> tuple[np.ndarray, np.ndarray]:
        if self._grid_step != self.match.step_count:
            g = self.match.ground_fuel()
            self._ground = g
            self._grid = fuel_density(g, _GRID)
            self._grid_step = self.match.step_count
        return self._grid, self._ground

    # ---------------------------------------------------------------- public
    def command(self, r: Robot, macro: Macro) -> RobotCommand:
        mem = self.mem[r.index]
        if macro != mem.macro:
            mem.macro, mem.since, mem.target = macro, self.match.t, None
        if r.spec.max_speed <= 0:
            return RobotCommand()
        if r.climb_state in (ClimbState.CLIMBING,):
            return RobotCommand(climb=r.climb_target)
        if r.climb_state == ClimbState.CLIMBED and macro == Macro.CLIMB:
            return RobotCommand(climb=r.climb_level)
        handler = {
            Macro.IDLE: self._idle,
            Macro.COLLECT: self._collect,
            Macro.COLLECT_DEPOT: self._collect_depot,
            Macro.COLLECT_OUTPOST: self._collect_outpost,
            Macro.SHOOT: self._shoot,
            Macro.STAGE: self._stage,
            Macro.DEFEND: self._defend,
            Macro.CLIMB: self._climb,
        }[macro]
        return self._unstick(r, mem, macro, handler(r, mem))

    def _unstick(self, r: Robot, mem: RobotMemory, macro: Macro, cmd: RobotCommand) -> RobotCommand:
        """Safety net: a robot that pushes without moving for a second backs off sideways and re-plans."""
        m = self.match
        t = m.t
        if t < mem.unstick_until:
            vx, vy = mem.unstick_vel
            return RobotCommand(vx, vy, cmd.omega, intake=cmd.intake, shoot=cmd.shoot)
        want = math.hypot(cmd.vx, cmd.vy)
        if macro != Macro.DEFEND and want > 1.0 and r.speed < 0.15 and r.climb_state == ClimbState.GROUND:
            mem.stall += m.cfg.dt
        else:
            mem.stall = 0.0
        if mem.stall < 1.0:
            return cmd
        mem.stall = 0.0
        mem.unstick_side = -mem.unstick_side  # alternate sides on repeated stalls
        ux, uy = cmd.vx / want, cmd.vy / want
        k = r.spec.max_speed * 0.6
        mem.unstick_vel = ((-0.5 * ux - uy * mem.unstick_side) * k, (-0.5 * uy + ux * mem.unstick_side) * k)
        mem.unstick_until = t + 0.6
        if macro == Macro.COLLECT and mem.target is not None:
            mem.avoid = (mem.target[0], mem.target[1], t + 4.0)
        mem.target = None
        vx, vy = mem.unstick_vel
        return RobotCommand(vx, vy, cmd.omega, intake=cmd.intake, shoot=cmd.shoot)

    def goto(self, r: Robot, gx: float, gy: float, heading: float | None = None) -> RobotCommand:
        """Drive to a field point with path planning (crossings, TOWERs, parked robots)."""
        vx, vy, om, _ = self._drive(r, gx, gy, heading=heading, face_travel=heading is None)
        return RobotCommand(vx, vy, om)

    # ---------------------------------------------------------------- helpers
    def _drive(self, r: Robot, gx: float, gy: float, heading: float | None = None, face_travel: bool = False,
               arrive: float = 0.08, speed_scale: float = 1.0) -> tuple[float, float, float, bool]:
        """Velocity and turn rate toward (gx, gy); returns (vx, vy, omega, arrived)."""
        if self.match.period == Period.AUTO:  # G403: never plan past the CENTER LINE in AUTO
            gx = min(gx, C.CENTER_X - 0.05) if r.alliance == Alliance.BLUE else max(gx, C.CENTER_X + 0.05)
        parked = [(o.x, o.y, o.spec.radius) for o in self.match.robots if o is not r and not o.mobile]
        wx, wy, final = next_waypoint(r, gx, gy, parked)
        dx, dy = wx - r.x, wy - r.y
        d = math.hypot(dx, dy)
        vmax = r.spec.max_speed * speed_scale
        arrived = final and d < arrive
        if arrived or d < 1e-6:
            vx = vy = 0.0
        else:
            v = min(vmax, math.sqrt(2 * r.spec.max_accel * d) * 0.85) if final else vmax
            vx, vy = dx / d * v, dy / d * v
        vx, vy = self._avoid(r, vx, vy)
        if r.rect:
            vx, vy = self._slide(r, vx, vy)
        if face_travel and (abs(vx) + abs(vy)) > 0.3:
            heading = math.atan2(vy, vx)
        omega = 0.0 if heading is None else max(-r.spec.max_omega, min(r.spec.max_omega, 5.0 * _angle_err(heading, r.heading)))
        return vx, vy, omega, arrived

    def _avoid(self, r: Robot, vx: float, vy: float) -> tuple[float, float]:
        """Steer around teammates and dead or climbing robots so an alliance doesn't jam itself."""
        for o in self.match.robots:
            if o is r or (o.alliance != r.alliance and o.mobile):
                continue
            dx, dy = r.x - o.x, r.y - o.y
            d = math.hypot(dx, dy)
            clear = r.reach + o.reach + 0.25
            if 1e-6 < d < clear:
                k = (clear - d) / clear * r.spec.max_speed
                vx += dx / d * k
                vy += dy / d * k
        return vx, vy

    def _slide(self, r: Robot, vx: float, vy: float) -> tuple[float, float]:
        """Steer along a wall, field element, teammate or parked robot the bumpers are touching instead
        of into it. A circle slides off by itself; a swerve robot's wheels grip sideways, so it would
        stick (high fidelity). Opponents still get pushed: that is defense."""
        c, s = math.cos(r.heading), math.sin(r.heading)
        box = (r.x, r.y, c, s, r.spec.length / 2 + 0.03, r.spec.width / 2 + 0.03)
        touching = []
        ext_x, ext_y = half_extents(box)
        if r.x - ext_x < 0:
            touching.append((1.0, 0.0))
        if r.x + ext_x > C.FIELD_LENGTH:
            touching.append((-1.0, 0.0))
        if r.y - ext_y < 0:
            touching.append((0.0, 1.0))
        if r.y + ext_y > C.FIELD_WIDTH:
            touching.append((0.0, -1.0))
        reach = r.reach + 0.03
        for x0, x1, y0, y1 in self.match._boxes_of[r.index]:
            if r.x + ext_x <= x0 or r.x - ext_x >= x1 or r.y + ext_y <= y0 or r.y - ext_y >= y1:
                continue
            hit = box_contact(aabb_box(x0, x1, y0, y1), box)
            if hit is not None:
                touching.append((hit[0], hit[1]))
        for o in self.match.robots:
            if o is r or (o.alliance != r.alliance and o.mobile):
                continue
            if abs(o.x - r.x) < reach + o.reach and abs(o.y - r.y) < reach + o.reach:
                hit = box_contact(robot_box(o), box)
                if hit is not None:
                    touching.append((hit[0], hit[1]))
        for nx, ny in touching:
            vn = vx * nx + vy * ny
            if vn < 0:
                vx -= vn * nx
                vy -= vn * ny
        return vx, vy

    def _own_side_limit(self, r: Robot) -> float:
        """During AUTO stay on this side of the CENTER LINE (G403)."""
        return C.CENTER_X - r.spec.radius * 0.2 if r.alliance == Alliance.BLUE else C.CENTER_X + r.spec.radius * 0.2

    def shoot_spot(self, r: Robot, d_pref: float | None = None) -> tuple[float, float]:
        """Nearest legal spot (bumpers overlapping own ALLIANCE ZONE) at a good range from the HUB."""
        s = r.spec
        hx, hy = FIELD.hub_centers[r.alliance]
        d = d_pref if d_pref is not None else min(max(s.sweet_range, s.min_range + 0.35), s.max_range - 0.4)
        lo, hi = self.match.shot_reach(r)  # high fidelity: where the shooter can actually score from
        d = min(max(d, lo + 0.15), hi - 0.2)
        sign = -1.0 if r.alliance == Alliance.BLUE else 1.0  # direction from HUB toward own wall
        zone_edge = C.ALLIANCE_ZONE_DEPTH if r.alliance == Alliance.BLUE else C.FIELD_LENGTH - C.ALLIANCE_ZONE_DEPTH
        angles = np.linspace(-1.35, 1.35, 19)
        xs = hx + sign * d * np.cos(angles)
        ys = hy + d * np.sin(angles)
        legal = (xs - zone_edge) * sign >= -0.5 * s.radius  # center no farther than half a radius past the line
        xs, ys = xs[legal], ys[legal]
        ys = np.clip(ys, s.radius + 0.05, C.FIELD_WIDTH - s.radius - 0.05)
        cost = np.hypot(xs - r.x, ys - r.y)
        for o in self.match.robots:
            if o is not r and o.alliance == r.alliance:
                tm = self.mem[o.index].target
                if tm is not None:
                    cost += 2.0 * (np.hypot(xs - tm[0], ys - tm[1]) < 1.0)
        k = int(np.argmin(cost))
        return float(xs[k]), float(ys[k])

    def can_shoot_here(self, r: Robot, x: float | None = None, y: float | None = None) -> bool:
        """In range of its HUB and legal to shoot, judged at (x, y) (default: where the robot is)."""
        x = r.x if x is None else x
        y = r.y if y is None else y
        hx, hy = FIELD.hub_centers[r.alliance]
        dist = math.hypot(hx - x, hy - y)
        x = x + r.vx * self.match.lookahead  # with command latency, judge where the shot will leave from
        return (r.spec.min_range <= dist <= r.spec.max_range
                and FIELD.in_alliance_zone(r.alliance, x, r.spec.radius * 0.5) and self.match.can_reach(r, dist))

    # ---------------------------------------------------------------- macros
    def _idle(self, r: Robot, mem: RobotMemory) -> RobotCommand:
        return RobotCommand()

    def _collect(self, r: Robot, mem: RobotMemory) -> RobotCommand:
        grid, ground = self._fuel_grid()
        t = self.match.t
        if mem.target is None or t - mem.target_t > 0.6:
            mem.target = self._pick_cluster(r, grid)
            mem.target_t = t
        if mem.target is None:
            return RobotCommand(intake=True)
        tx, ty = mem.target
        # close to the cluster: chase the nearest ball in reach
        if ground.size and math.hypot(tx - r.x, ty - r.y) < 1.0:
            d = np.hypot(ground[:, 0] - r.x, ground[:, 1] - r.y)
            if self.match.period == Period.AUTO:
                limit = self._own_side_limit(r)
                ok = ground[:, 0] <= limit if r.alliance == Alliance.BLUE else ground[:, 0] >= limit
                d = np.where(ok, d, np.inf)
            if not r.spec.can_trench:  # can't reach FUEL under a TRENCH arm
                d = np.where(_in_trench(ground[:, 0], ground[:, 1]), np.inf, d)
            d = np.where(_in_tower(ground[:, 0], ground[:, 1]), np.inf, d)
            if mem.avoid is not None and t < mem.avoid[2]:
                d = np.where(np.hypot(ground[:, 0] - mem.avoid[0], ground[:, 1] - mem.avoid[1]) < 0.6, np.inf, d)
            k = int(np.argmin(d))
            if d[k] < 1.6:
                tx, ty = float(ground[k, 0]), float(ground[k, 1])
        vx, vy, om, _ = self._drive(r, tx, ty, face_travel=True, arrive=0.02)
        if math.hypot(tx - r.x, ty - r.y) < 0.6:  # turn the intake onto the target
            om = max(-r.spec.max_omega, min(r.spec.max_omega, 6.0 * _angle_err(math.atan2(ty - r.y, tx - r.x), r.heading)))
        return RobotCommand(vx, vy, om, intake=True, shoot=self._opportunistic_shot(r))

    def _pick_cluster(self, r: Robot, grid: np.ndarray) -> tuple[float, float] | None:
        cx, cy, X, Y, reg = _CX, _CY, _GX, _GY, _GREG
        dist = np.hypot(X - r.x, Y - r.y)
        # charge for HUB-line crossings between robot and cell
        rr = region(r.x)
        crossings = ((np.minimum(reg, rr) < 1) & (np.maximum(reg, rr) > 1)).astype(float) + \
                    ((np.minimum(reg, rr) < 3) & (np.maximum(reg, rr) > 3)).astype(float)
        score = grid ** 0.8 / (1.0 + (dist + 1.5 * crossings) / 2.5)
        if self.match.period == Period.AUTO:
            limit = self._own_side_limit(r)
            score[(X > limit) if r.alliance == Alliance.BLUE else (X < limit)] = 0.0
        score[(reg == 1) | (reg == 3)] *= 0.3  # FUEL stuck against the HUB line is awkward to reach
        if not r.spec.can_trench:
            score[_GTRENCH] = 0.0
        score[_GTOWER] = 0.0
        mem = self.mem[r.index]
        if mem.avoid is not None and self.match.t < mem.avoid[2]:
            score[np.hypot(X - mem.avoid[0], Y - mem.avoid[1]) < 1.0] = 0.0
        for o in self.match.robots:  # spread teammates over different clusters
            if o is not r and o.alliance == r.alliance:
                tm = self.mem[o.index].target
                if tm is not None and self.mem[o.index].macro == Macro.COLLECT:
                    score *= 1.0 - 0.7 * (np.hypot(X - tm[0], Y - tm[1]) < 1.5)
        if score.max() <= 0:
            return None
        i, j = np.unravel_index(int(np.argmax(score)), score.shape)
        return float(cx[i]), float(cy[j])

    def _opportunistic_shot(self, r: Robot) -> bool:
        """Turret robots fire on the move whenever the HUB counts and the shot is legal."""
        return (r.spec.turret and r.fuel > 0 and self.match.hub_active(r.alliance) and self.can_shoot_here(r))

    def _collect_depot(self, r: Robot, mem: RobotMemory) -> RobotCommand:
        box = FIELD.depots[r.alliance]
        cx, cy = box.center
        sign = 1.0 if r.alliance == Alliance.BLUE else -1.0
        wall_heading = math.pi if r.alliance == Alliance.BLUE else 0.0
        mem.sweep_phase += self.match.cfg.dt
        sweep = 0.3 * math.sin(mem.sweep_phase * 2.0)
        gx = (box.x1 if r.alliance == Alliance.BLUE else box.x0) + sign * (r.spec.radius - 0.1)
        # FUEL the intake can reach from there: in the DEPOT, no deeper than this x
        deepest = gx - sign * (r.front + r.spec.intake_reach + C.FUEL_RADIUS)
        g = self.match.ground_fuel()
        reachable = ((g[:, 0] >= box.x0) & (g[:, 0] <= box.x1) & (g[:, 1] >= box.y0) & (g[:, 1] <= box.y1)
                     & (sign * (g[:, 0] - deepest) >= 0.0))
        # A flat bumper pins FUEL against the wall, and FUEL pushed back against it is out of reach:
        # then bring the intake right up to the wall.
        if r.rect or not reachable.any():
            gx = (0.0 if r.alliance == Alliance.BLUE else C.FIELD_LENGTH) + sign * (r.front + 0.06)
        vx, vy, om, _ = self._drive(r, gx, cy + sweep, heading=wall_heading, arrive=0.05)
        return RobotCommand(vx, vy, om, intake=True)

    def _collect_outpost(self, r: Robot, mem: RobotMemory) -> RobotCommand:
        gx, gy = FIELD.outpost_spot(r.alliance, r.spec.radius)
        wall_heading = math.pi if r.alliance == Alliance.BLUE else 0.0
        vx, vy, om, _ = self._drive(r, gx, gy, heading=wall_heading, arrive=0.05)
        return RobotCommand(vx, vy, om, intake=True)

    def _aim(self, r: Robot) -> float:
        hx, hy = FIELD.hub_centers[r.alliance]
        return math.atan2(hy - r.y, hx - r.x)

    def _shoot(self, r: Robot, mem: RobotMemory, fire: bool = True) -> RobotCommand:
        s = r.spec
        hx, hy = FIELD.hub_centers[r.alliance]
        dist = math.hypot(hx - r.x, hy - r.y)
        legal_here = self.can_shoot_here(r)
        aim_om = 0.0 if s.turret else max(-s.max_omega, min(s.max_omega, 6.0 * _angle_err(self._aim(r), r.heading)))
        if legal_here and dist <= s.sweet_range + 0.6:  # good enough: stop and fire
            mem.target = None
            return RobotCommand(0.0, 0.0, aim_om, shoot=fire)
        if mem.target is None or self.match.t - mem.target_t > 1.0:
            mem.target = self.shoot_spot(r)
            mem.target_t = self.match.t
        heading = None if s.turret else self._aim(r)
        vx, vy, om, _ = self._drive(r, *mem.target, heading=heading, face_travel=s.turret, arrive=0.1)
        # fire on the way in once the shot is legal (a moving shot is less accurate)
        return RobotCommand(vx, vy, om, intake=r.fuel < s.capacity, shoot=fire and legal_here)

    def _stage(self, r: Robot, mem: RobotMemory) -> RobotCommand:
        return self._shoot(r, mem, fire=False)

    def _defend(self, r: Robot, mem: RobotMemory) -> RobotCommand:
        m = self.match
        t = m.t
        if m.period == Period.AUTO:  # no defense across the CENTER LINE in AUTO (G403)
            return self._collect(r, mem)
        if t < mem.backoff_until:  # G418: get 72 in clear for 3 s so the pin count resets
            opp = m.robots[mem.notes.get("victim", r.index)]
            dx, dy = r.x - opp.x, r.y - opp.y
            d = math.hypot(dx, dy) or 1.0
            if d > C.PIN_RELEASE_DISTANCE + 0.4:
                return RobotCommand()
            return RobotCommand(dx / d * r.spec.max_speed, dy / d * r.spec.max_speed)
        victim = self._pick_victim(r)
        if victim is None:
            return self._collect(r, mem)
        mem.notes["victim"] = victim.index
        # get between the victim and its HUB, then push
        hx, hy = FIELD.hub_centers[victim.alliance]
        vx_, vy_ = hx - victim.x, hy - victim.y
        dv = math.hypot(vx_, vy_) or 1.0
        gx = victim.x + vx_ / dv * 0.6
        gy = victim.y + vy_ / dv * 0.6
        pair = (min(r.index, victim.index), max(r.index, victim.index))
        if pair in m.contacts and victim.speed < 0.4:
            mem.contact_time += m.cfg.dt
        else:
            mem.contact_time = max(0.0, mem.contact_time - m.cfg.dt)
        if mem.contact_time > 2.0:
            mem.backoff_until = t + C.PIN_LIMIT + 1.0
            mem.contact_time = 0.0
        vx, vy, om, _ = self._drive(r, gx, gy, face_travel=True, arrive=0.0)
        return RobotCommand(vx, vy, om)

    def _pick_victim(self, r: Robot) -> Robot | None:
        m = self.match
        best, best_score = None, -1e9
        for o in m.robots:
            if o.alliance == r.alliance or not o.mobile:
                continue
            if m.period == Period.ENDGAME and FIELD.in_climb_zone(o.alliance, o.x, o.y, o.extent_x):
                continue  # G420
            score = o.fuel * (2.0 if m.hub_active(o.alliance) else 1.0) - math.hypot(o.x - r.x, o.y - r.y) * 2.0
            if score > best_score:
                best, best_score = o, score
        return best

    def _climb(self, r: Robot, mem: RobotMemory) -> RobotCommand:
        if r.spec.climb_level <= 0:
            return RobotCommand()
        if mem.target is None:  # nearest climbing position no teammate holds or is heading to
            taken = []
            for o in self.match.robots:
                if o is r or o.alliance != r.alliance:
                    continue
                if o.climb_state != ClimbState.GROUND:
                    taken.append((o.x, o.y))
                elif self.mem[o.index].macro == Macro.CLIMB and self.mem[o.index].target is not None:
                    taken.append(self.mem[o.index].target)
            spots = FIELD.climb_spots(r.alliance, r.spec.radius)
            free = [s for s in spots if all(math.hypot(s[0] - tx, s[1] - ty) > 0.5 for tx, ty in taken)] or spots
            mem.target = min(free, key=lambda s: math.hypot(s[0] - r.x, s[1] - r.y))
        gx, gy = mem.target
        heading = math.pi if r.alliance == Alliance.BLUE else 0.0
        vx, vy, om, _ = self._drive(r, gx, gy, heading=heading, arrive=0.05)
        in_zone = FIELD.in_climb_zone(r.alliance, r.x, r.y, r.extent_x)
        return RobotCommand(vx, vy, om, climb=r.spec.climb_level if in_zone else 0)
