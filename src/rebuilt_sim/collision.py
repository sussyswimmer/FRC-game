"""Rectangular bumper collisions (high fidelity).

Robots are their bumper footprints: rectangles that turn with the robot. Contacts come from
the separating axis test; the contact point is found by clipping the touching edges (so two
robots meeting bumper-to-bumper push through the middle of the overlap, while a corner hit
pushes at the corner and spins the robot). Each contact gets an impulse that stops the robots
moving into each other (with a little bounce) plus bumper friction, applied at the contact
point so it changes both velocity and spin. Leftover overlap is then pushed apart, split by
mass. Robots that can't move (climbing, broken) act as fixed obstacles.
"""

from __future__ import annotations

import math

import numpy as np

from . import constants as C
from .drivetrain import moment_of_inertia

BUMPER_FRICTION = 0.3  # bumper fabric on bumper fabric, with the robots' own motion shaking them loose
WALL_FRICTION = 0.3  # bumpers sliding along the field wall or a field element
WALL_RESTITUTION = 0.1
SLOP = 0.002  # m of overlap left for the next step, so resting contacts don't jitter
PUSH = 0.8  # fraction of the overlap removed per pass
ITERATIONS = 2


def robot_box(r) -> tuple[float, float, float, float, float, float]:
    """(cx, cy, cos, sin, half length, half width) of a robot's bumper footprint."""
    return r.x, r.y, math.cos(r.heading), math.sin(r.heading), r.spec.length / 2, r.spec.width / 2


def half_extents(box) -> tuple[float, float]:
    """Half size of an oriented box's axis-aligned bounding box."""
    _, _, c, s, hl, hw = box
    return abs(hl * c) + abs(hw * s), abs(hl * s) + abs(hw * c)


def aabb_box(x0: float, x1: float, y0: float, y1: float) -> tuple[float, float, float, float, float, float]:
    return (x0 + x1) / 2, (y0 + y1) / 2, 1.0, 0.0, (x1 - x0) / 2, (y1 - y0) / 2


def corners(box) -> list[tuple[float, float]]:
    cx, cy, c, s, hl, hw = box
    ux, uy, vx, vy = c * hl, s * hl, -s * hw, c * hw
    return [(cx + ux + vx, cy + uy + vy), (cx + ux - vx, cy + uy - vy),
            (cx - ux - vx, cy - uy - vy), (cx - ux + vx, cy - uy + vy)]


def box_contact(a, b):
    """Overlap of two oriented boxes: (nx, ny, depth, px, py) with the normal pointing from a to
    b and (px, py) the contact point, or None if they don't touch."""
    ax, ay, ac, as_, ahl, ahw = a
    bx, by, bc, bs, bhl, bhw = b
    dx, dy = bx - ax, by - ay
    best = None
    for owner, lx, ly in ((0, ac, as_), (0, -as_, ac), (1, bc, bs), (1, -bs, bc)):
        ra = ahl * abs(ac * lx + as_ * ly) + ahw * abs(-as_ * lx + ac * ly)
        rb = bhl * abs(bc * lx + bs * ly) + bhw * abs(-bs * lx + bc * ly)
        d = dx * lx + dy * ly
        overlap = ra + rb - abs(d)
        if overlap <= 0.0:
            return None
        if best is None or overlap < best[0] - 1e-9:
            sign = 1.0 if d >= 0 else -1.0
            best = (overlap, lx * sign, ly * sign, owner)
    depth, nx, ny, owner = best
    # reference face: the face of the box that owns the axis, facing the other (incident) box
    ref, inc, rnx, rny = (a, b, nx, ny) if owner == 0 else (b, a, -nx, -ny)
    rx, ry, rc, rs, rhl, rhw = ref
    if abs(rc * rnx + rs * rny) > 0.5:  # the face is at the end of the reference box's length
        r_off, r_half = rhl, rhw
    else:
        r_off, r_half = rhw, rhl
    fx, fy = rx + rnx * r_off, ry + rny * r_off
    tx, ty = -rny, rnx
    # incident edge: the incident box's face most opposed to the reference normal
    ix, iy, ic, is_, ihl, ihw = inc
    du, dv = ic * rnx + is_ * rny, -is_ * rnx + ic * rny
    if abs(du) >= abs(dv):
        sgn = -1.0 if du > 0 else 1.0
        ex, ey = ix + sgn * ic * ihl, iy + sgn * is_ * ihl
        sx, sy = -is_ * ihw, ic * ihw
    else:
        sgn = -1.0 if dv > 0 else 1.0
        ex, ey = ix - sgn * is_ * ihw, iy + sgn * ic * ihw
        sx, sy = ic * ihl, is_ * ihl
    # clip the edge (ex, ey) +- (sx, sy) to the reference face's extent along its tangent
    t0 = (ex - sx - fx) * tx + (ey - sy - fy) * ty
    t1 = (ex + sx - fx) * tx + (ey + sy - fy) * ty
    if abs(t1 - t0) > 1e-12:
        wa, wb = (-r_half - t0) / (t1 - t0), (r_half - t0) / (t1 - t0)
        lo, hi = max(0.0, min(wa, wb)), min(1.0, max(wa, wb))
    else:
        lo, hi = (0.0, 1.0) if abs(t0) <= r_half else (1.0, 0.0)
    pts = []
    if lo <= hi:
        for w in ((lo, hi) if hi > lo else (lo,)):
            qx, qy = ex - sx + 2 * sx * w, ey - sy + 2 * sy * w
            if (fx - qx) * rnx + (fy - qy) * rny > -1e-6:  # behind the reference face: touching
                pts.append((qx, qy))
    if pts:
        px = sum(p[0] for p in pts) / len(pts)
        py = sum(p[1] for p in pts) / len(pts)
    else:  # degenerate: deepest incident corner
        px, py = min(corners(inc), key=lambda p: (p[0] - fx) * rnx + (p[1] - fy) * rny)
    return nx, ny, depth, px, py


class Collider:
    """Resolves bumper contacts for all robots of a match each step."""

    def __init__(self, match) -> None:
        self.match = match
        self.inv_mass = [1.0 / r.spec.mass for r in match.robots]
        self.inv_inertia = [1.0 / moment_of_inertia(r.spec) for r in match.robots]
        self.reach = [math.hypot(r.spec.length, r.spec.width) / 2 for r in match.robots]

    def _inv(self, r) -> tuple[float, float]:
        return (self.inv_mass[r.index], self.inv_inertia[r.index]) if r.mobile else (0.0, 0.0)

    def resolve(self) -> set[tuple[int, int]]:
        m = self.match
        rs = m.robots
        contacts: set[tuple[int, int]] = set()
        e = m.cfg.robot_restitution
        # broad phase, once per step, with a margin for the pushes the passes below can cause
        pairs = []
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                reach = self.reach[i] + self.reach[j] + 0.1
                dx, dy = rs[j].x - rs[i].x, rs[j].y - rs[i].y
                if abs(dx) < reach and abs(dy) < reach and dx * dx + dy * dy < reach * reach:
                    pairs.append((i, j))
        statics = []  # (robot, near a field wall, field elements it may touch)
        for r in rs:
            if not r.mobile:
                continue
            ex, ey = half_extents(robot_box(r))
            ex, ey = ex + 0.2, ey + 0.2
            walls = r.x < ex or r.x > C.FIELD_LENGTH - ex or r.y < ey or r.y > C.FIELD_WIDTH - ey
            boxes = [b for b in m._robot_boxes[not r.spec.can_trench]
                     if r.x + ex > b[0] and r.x - ex < b[1] and r.y + ey > b[2] and r.y - ey < b[3]]
            if walls or boxes:
                statics.append((r, walls, boxes))
        if not pairs and not statics:
            return contacts
        for _ in range(ITERATIONS):
            for r, walls, boxes in statics:
                self._static(r, walls, boxes)
            for i, j in pairs:
                a, b = rs[i], rs[j]
                if abs(b.x - a.x) >= a.extent_x + b.extent_x:  # cheap bounding-box miss
                    continue
                hit = box_contact(robot_box(a), robot_box(b))
                if hit is not None:
                    contacts.add((i, j))
                    self._pair(a, b, hit, e)
        for r, walls, boxes in statics:  # field walls and elements win over robot pushes
            self._static(r, walls, boxes)
        return contacts

    # ---------------------------------------------------------------- contacts
    def _pair(self, a, b, hit, e: float) -> None:
        nx, ny, depth, px, py = hit
        ia, ja = self._inv(a)
        ib, jb = self._inv(b)
        if ia + ib == 0.0:
            return
        _impulse(a, ia, ja, b, ib, jb, nx, ny, px, py, e, BUMPER_FRICTION)
        corr = max(depth - SLOP, 0.0) * PUSH / (ia + ib)
        a.x -= nx * corr * ia
        a.y -= ny * corr * ia
        b.x += nx * corr * ib
        b.y += ny * corr * ib

    def _static(self, r, walls: bool, boxes) -> None:
        ib, jb = self._inv(r)
        if ib == 0.0:
            return
        if walls:
            self._walls(r, robot_box(r), ib, jb)
        for x0, x1, y0, y1 in boxes:
            hit = box_contact(aabb_box(x0, x1, y0, y1), robot_box(r))
            if hit is None:
                continue
            nx, ny, depth, px, py = hit
            _impulse(None, 0.0, 0.0, r, ib, jb, nx, ny, px, py, WALL_RESTITUTION, WALL_FRICTION)
            r.x += nx * depth
            r.y += ny * depth

    def _walls(self, r, box, ib: float, jb: float) -> None:
        cs = corners(box)
        for nx, ny, limit in ((1.0, 0.0, 0.0), (-1.0, 0.0, -C.FIELD_LENGTH), (0.0, 1.0, 0.0),
                              (0.0, -1.0, -C.FIELD_WIDTH)):
            pens = [limit - (x * nx + y * ny) for x, y in cs]  # > 0: this corner is through the wall
            depth = max(pens)
            if depth <= 0.0:
                continue
            deep = [c for c, p in zip(cs, pens) if p > depth - 0.01]
            px = sum(c[0] for c in deep) / len(deep)
            py = sum(c[1] for c in deep) / len(deep)
            _impulse(None, 0.0, 0.0, r, ib, jb, nx, ny, px, py, WALL_RESTITUTION, WALL_FRICTION)
            r.x += nx * depth
            r.y += ny * depth
            cs = [(x + nx * depth, y + ny * depth) for x, y in cs]


def _impulse(a, ia: float, ja: float, b, ib: float, jb: float, nx: float, ny: float,
             px: float, py: float, e: float, mu: float) -> None:
    """Normal impulse (with restitution e) and Coulomb friction at contact point (px, py) between
    robot a (None: the field) and robot b; the normal points from a to b."""
    rbx, rby = px - b.x, py - b.y
    if a is not None:
        rax, ray = px - a.x, py - a.y
        vax, vay = a.vx - a.omega * ray, a.vy + a.omega * rax
    else:
        rax = ray = vax = vay = 0.0
    vbx, vby = b.vx - b.omega * rby, b.vy + b.omega * rbx
    vn = (vbx - vax) * nx + (vby - vay) * ny
    if vn >= 0.0:
        return
    ran, rbn = rax * ny - ray * nx, rbx * ny - rby * nx
    k = ia + ib + ja * ran * ran + jb * rbn * rbn
    jn = -(1.0 + e) * vn / k
    _apply(a, ia, ja, rax, ray, -jn * nx, -jn * ny)
    _apply(b, ib, jb, rbx, rby, jn * nx, jn * ny)
    tx, ty = -ny, nx
    if a is not None:
        vax, vay = a.vx - a.omega * ray, a.vy + a.omega * rax
    vbx, vby = b.vx - b.omega * rby, b.vy + b.omega * rbx
    vt = (vbx - vax) * tx + (vby - vay) * ty
    rat, rbt = rax * ty - ray * tx, rbx * ty - rby * tx
    kt = ia + ib + ja * rat * rat + jb * rbt * rbt
    jt = max(-mu * jn, min(mu * jn, -vt / kt))
    _apply(a, ia, ja, rax, ray, -jt * tx, -jt * ty)
    _apply(b, ib, jb, rbx, rby, jt * tx, jt * ty)


def _apply(r, inv_m: float, inv_i: float, rx: float, ry: float, jx: float, jy: float) -> None:
    if r is None or inv_m == 0.0:
        return
    r.vx += jx * inv_m
    r.vy += jy * inv_m
    r.omega += (rx * jy - ry * jx) * inv_i


def push_fuel(match, g: np.ndarray) -> None:
    """Robots bulldoze FUEL on the floor: balls (indices ``g``) inside a bumper footprint are
    pushed out through the nearest side with the speed of the bumper at that point (the
    rectangle version of ``Match._push_fuel``)."""
    P = match.f_pos[g]
    rb = C.FUEL_RADIUS
    changed = False
    for r in match.robots:
        hl, hw = r.spec.length / 2 + rb, r.spec.width / 2 + rb
        reach = math.hypot(hl, hw)
        dx, dy = P[:, 0] - r.x, P[:, 1] - r.y
        near = (np.abs(dx) < reach) & (np.abs(dy) < reach)
        if not near.any():
            continue
        k = np.flatnonzero(near)
        c, s = math.cos(r.heading), math.sin(r.heading)
        lx = dx[k] * c + dy[k] * s
        ly = -dx[k] * s + dy[k] * c
        pen_x, pen_y = hl - np.abs(lx), hw - np.abs(ly)
        inside = (pen_x > 0) & (pen_y > 0)
        if not inside.any():
            continue
        k, lx, ly, pen_x, pen_y = k[inside], lx[inside], ly[inside], pen_x[inside], pen_y[inside]
        along_x = pen_x < pen_y
        sx = np.where(lx >= 0, 1.0, -1.0)
        sy = np.where(ly >= 0, 1.0, -1.0)
        lx = np.where(along_x, sx * (hl + 0.004), lx)
        ly = np.where(along_x, ly, sy * (hw + 0.004))
        nlx = np.where(along_x, sx, 0.0)
        nly = np.where(along_x, 0.0, sy)
        nx, ny = nlx * c - nly * s, nlx * s + nly * c
        wx, wy = lx * c - ly * s, lx * s + ly * c  # new offset from the robot center, field frame
        P[k, 0] = r.x + wx
        P[k, 1] = r.y + wy
        bvx = r.vx - r.omega * wy  # bumper velocity at each ball
        bvy = r.vy + r.omega * wx
        push = np.maximum(bvx * nx + bvy * ny, 0.0) * 1.15 + 0.05
        gi = g[k]
        match.f_vel[gi, 0] = nx * push + match.rng.normal(0, 0.05, k.size)
        match.f_vel[gi, 1] = ny * push + match.rng.normal(0, 0.05, k.size)
        match.f_settle[gi] = 3
        changed = True
    if changed:
        match.f_pos[g] = P
