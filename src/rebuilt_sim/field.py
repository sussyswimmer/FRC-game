"""Static field geometry: obstacles, zones, BUMPs, TRENCHes, TOWERs, DEPOTs, OUTPOSTs.

All boxes are axis-aligned. Blue elements are defined directly; red ones are mirrored
through the field center (the REBUILT field is point-symmetric in 2D).

Some obstacles are overhead: a robot drives under them if it is short enough. A robot's
*clearance class* (``Field.clearance``) says which ones stop it:

- 0: passes under everything (the TRENCH arms at 22.25 in, the TOWER RUNGS, the TOWER supports);
- 1: taller than the TRENCH clearance;
- 2: also taller than the LOW RUNG, so it can't cross the TOWER's front (the UPRIGHTS and the
  RUNGS between and past them);
- 3: also taller than the TOWER's supports, so it can't pass behind the UPRIGHTS along the wall.

The TOWER is an open frame (Game Manual 5.8). Its floor plate is flat enough to drive on, so for
most robots only the two UPRIGHTS are in the way: a robot that hugs the alliance wall passes
behind them, and FUEL rolls under the RUNGS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C
from .constants import Alliance


@dataclass(frozen=True)
class Box:
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return self.x0 - margin <= x <= self.x1 + margin and self.y0 - margin <= y <= self.y1 + margin

    def mirrored(self) -> "Box":
        return Box(C.FIELD_LENGTH - self.x1, C.FIELD_LENGTH - self.x0, C.FIELD_WIDTH - self.y1, C.FIELD_WIDTH - self.y0)


def _pair(blue: Box) -> tuple[Box, Box]:
    return blue, blue.mirrored()


# --- blue-side definitions -------------------------------------------------------------
_HX, _HY = C.HUB_CENTER_BLUE
_HUB = Box(_HX - C.HUB_HALF, _HX + C.HUB_HALF, _HY - C.HUB_HALF, _HY + C.HUB_HALF)
_BUMP_X = (_HX - C.BUMP_DEPTH / 2, _HX + C.BUMP_DEPTH / 2)
_BUMPS = (
    Box(*_BUMP_X, _HUB.y0 - C.BUMP_WIDTH, _HUB.y0),
    Box(*_BUMP_X, _HUB.y1, _HUB.y1 + C.BUMP_WIDTH),
)
_TRENCH_X = (_HX - C.TRENCH_DEPTH / 2, _HX + C.TRENCH_DEPTH / 2)
_TRENCH_OPENINGS = (
    Box(*_TRENCH_X, 0.0, C.TRENCH_OPENING_WIDTH),
    Box(*_TRENCH_X, C.FIELD_WIDTH - C.TRENCH_OPENING_WIDTH, C.FIELD_WIDTH),
)
# Support columns between each TRENCH opening and the neighbouring BUMP
_TRENCH_COLUMNS = (
    Box(*_TRENCH_X, C.TRENCH_OPENING_WIDTH, _BUMPS[0].y0),
    Box(*_TRENCH_X, _BUMPS[1].y1, C.FIELD_WIDTH - C.TRENCH_OPENING_WIDTH),
)
_TOWER = Box(0.0, C.TOWER_DEPTH, C.TOWER_CENTER_Y_BLUE - C.TOWER_WIDTH / 2, C.TOWER_CENTER_Y_BLUE + C.TOWER_WIDTH / 2)
_TY, _UX = C.TOWER_CENTER_Y_BLUE, C.TOWER_UPRIGHT_X
_UPRIGHT_IN, _UPRIGHT_OUT = C.TOWER_UPRIGHT_GAP / 2, C.TOWER_UPRIGHT_GAP / 2 + C.TOWER_UPRIGHT_THICKNESS
_TOWER_UPRIGHTS = (Box(*_UX, _TY - _UPRIGHT_OUT, _TY - _UPRIGHT_IN), Box(*_UX, _TY + _UPRIGHT_IN, _TY + _UPRIGHT_OUT))
# the TOWER's front: the UPRIGHTS and the RUNGS between and past them (a wall for robots too tall
# to pass under the LOW RUNG)
_TOWER_FRONT = Box(*_UX, _TY - C.TOWER_RUNG_HALF_LENGTH, _TY + C.TOWER_RUNG_HALF_LENGTH)
# supports from each UPRIGHT back to the wall (1.75 in tubes and braces), from 28.4 in up
_TOWER_SUPPORTS = tuple(Box(0.0, _UX[0], (u.y0 + u.y1) / 2 - 0.022, (u.y0 + u.y1) / 2 + 0.022) for u in _TOWER_UPRIGHTS)
_DEPOT = Box(0.0, C.DEPOT_DEPTH, C.DEPOT_CENTER_Y_BLUE - C.DEPOT_WIDTH / 2, C.DEPOT_CENTER_Y_BLUE + C.DEPOT_WIDTH / 2)
_ALLIANCE_ZONE = Box(0.0, C.ALLIANCE_ZONE_DEPTH, 0.0, C.FIELD_WIDTH)
NEUTRAL_ZONE = Box(C.BAND_FAR_X, C.FIELD_LENGTH - C.BAND_FAR_X, 0.0, C.FIELD_WIDTH)

# Climbing positions in front of the TOWER, as y offsets from its center. A climbing robot must
# touch a RUNG or an UPRIGHT and may touch another robot (Game Manual 6.5.2), so up to three can
# hang side by side: one in the middle, and one on each side hooked onto the 5.875 in of RUNG past
# an UPRIGHT (the RUNGS end 0.60 m from the center, so a side robot centered 0.95 m out overlaps
# them with its inner ~0.1 m). APPROX: the manual only implies at least two (up to 2 robots score in
# AUTO, and TRAVERSAL needs 50 TOWER points).
CLIMB_SLOT_OFFSETS = (0.0, -0.95, 0.95)

# Crossing points of a HUB line, as y coordinates (identical on both lines by symmetry)
TRENCH_PORTAL_Y = tuple((b.y0 + b.y1) / 2 for b in _TRENCH_OPENINGS)
BUMP_PORTAL_Y = tuple((b.y0 + b.y1) / 2 for b in _BUMPS)
# x extent of each HUB line (blue line first)
BAND_X = (
    (min(_HUB.x0, _TRENCH_X[0]), max(_HUB.x1, _TRENCH_X[1])),
    (C.FIELD_LENGTH - max(_HUB.x1, _TRENCH_X[1]), C.FIELD_LENGTH - min(_HUB.x0, _TRENCH_X[0])),
)


class Field:
    """Geometry queries used by the physics, the bots and the renderer."""

    def __init__(self) -> None:
        self.hubs = _pair(_HUB)
        self.hub_centers = tuple(b.center for b in self.hubs)
        self.bumps = tuple(_pair(b) for b in _BUMPS)  # [i][alliance]
        self.trench_openings = tuple(_pair(b) for b in _TRENCH_OPENINGS)
        self.trench_columns = tuple(_pair(b) for b in _TRENCH_COLUMNS)
        self.towers = _pair(_TOWER)  # the whole footprint: the TOWER WALL's width, out to the floor plate's front
        self.tower_fronts = _pair(_TOWER_FRONT)  # where climbing robots line up
        self.tower_uprights = tuple(_pair(b) for b in _TOWER_UPRIGHTS)
        self.depots = _pair(_DEPOT)
        self.alliance_zones = _pair(_ALLIANCE_ZONE)
        self.neutral_zone = NEUTRAL_ZONE

        all_bumps = [b for pair in self.bumps for b in pair]
        self.bump_boxes = tuple(all_bumps)
        columns = [b for pair in self.trench_columns for b in pair]
        openings = [b for pair in self.trench_openings for b in pair]
        uprights = [b for pair in self.tower_uprights for b in pair]
        fronts = list(self.tower_fronts)
        supports = [b for s in _TOWER_SUPPORTS for b in _pair(s)]
        # obstacles by clearance class (see the module docstring)
        base = list(self.hubs) + columns + uprights
        self.robot_obstacles = (
            tuple(base),
            tuple(base + openings),
            tuple(base + openings + fronts),
            tuple(base + openings + fronts + supports),
        )
        self.robot_obstacles_short = self.robot_obstacles[0]  # robots that fit under a TRENCH arm
        self.robot_obstacles_tall = self.robot_obstacles[2]  # the others (every robot tier is one of these two)
        # FUEL rolls under the TRENCH arms and the TOWER's RUNGS and supports
        self.fuel_obstacles = tuple(base)

        self.obstacle_arrays = {k: _boxes_array(b) for k, b in enumerate(self.robot_obstacles)}
        self.fuel_obstacle_array = _boxes_array(self.fuel_obstacles)
        self.bump_array = _boxes_array(self.bump_boxes)

        # Four HUB exits on the face toward the NEUTRAL ZONE
        exits = []
        for a in Alliance:
            hub = self.hubs[a]
            sign = 1.0 if a == Alliance.BLUE else -1.0
            face_x = hub.x1 if a == Alliance.BLUE else hub.x0
            ys = hub.center[1] + np.array([-0.42, -0.14, 0.14, 0.42])
            exits.append((np.stack([np.full(4, face_x + sign * (C.FUEL_RADIUS + 0.02)), ys], axis=1), sign))
        self.hub_exits = tuple(exits)

    @staticmethod
    def clearance(spec) -> int:
        """A robot's clearance class: how many kinds of overhead obstacle it is too tall for."""
        return sum(spec.height > h for h in (C.TRENCH_CLEARANCE, C.LOW_RUNG_CLEARANCE, C.TOWER_SUPPORT_CLEARANCE))

    # --- zones -------------------------------------------------------------------------
    def in_alliance_zone(self, alliance: Alliance, x: float, radius: float = 0.0) -> bool:
        """True if a robot footprint of ``radius`` at ``x`` overlaps its ALLIANCE ZONE (G407)."""
        if alliance == Alliance.BLUE:
            return x - radius < C.ALLIANCE_ZONE_DEPTH
        return x + radius > C.FIELD_LENGTH - C.ALLIANCE_ZONE_DEPTH

    def fully_across_center(self, alliance: Alliance, x: float, radius: float) -> bool:
        """G403: in AUTO a robot may not have its bumpers completely past the CENTER LINE."""
        if alliance == Alliance.BLUE:
            return x - radius > C.CENTER_X
        return x + radius < C.CENTER_X

    def on_bump(self, x: float, y: float) -> bool:
        return any(b.contains(x, y) for b in self.bump_boxes)

    def in_climb_zone(self, alliance: Alliance, x: float, y: float, radius: float) -> bool:
        """Bumpers against the TOWER's front at one of its climbing positions."""
        tower = self.tower_fronts[alliance]
        cy = tower.center[1]
        if min(abs(y - (cy + off)) for off in CLIMB_SLOT_OFFSETS) > 0.3:
            return False
        if alliance == Alliance.BLUE:
            gap = (x - radius) - tower.x1
        else:
            gap = tower.x0 - (x + radius)
        return -0.05 <= gap <= 0.25

    def climb_spots(self, alliance: Alliance, radius: float) -> list[tuple[float, float]]:
        """Where a robot parks to climb, one per climbing position across the TOWER's front."""
        tower = self.tower_fronts[alliance]
        cy = tower.center[1]
        x = tower.x1 + radius + 0.08 if alliance == Alliance.BLUE else tower.x0 - radius - 0.08
        return [(x, cy + off) for off in CLIMB_SLOT_OFFSETS]

    def climb_spot(self, alliance: Alliance, radius: float) -> tuple[float, float]:
        return self.climb_spots(alliance, radius)[0]

    def in_outpost_feed(self, alliance: Alliance, x: float, y: float, radius: float) -> bool:
        cy = C.OUTPOST_CENTER_Y_BLUE if alliance == Alliance.BLUE else C.FIELD_WIDTH - C.OUTPOST_CENTER_Y_BLUE
        if abs(y - cy) > C.OUTPOST_FEED_HALF_WIDTH:
            return False
        dist = x - radius if alliance == Alliance.BLUE else C.FIELD_LENGTH - (x + radius)
        return dist <= C.OUTPOST_FEED_DEPTH

    def outpost_spot(self, alliance: Alliance, radius: float) -> tuple[float, float]:
        if alliance == Alliance.BLUE:
            return radius + 0.2, C.OUTPOST_CENTER_Y_BLUE
        return C.FIELD_LENGTH - radius - 0.2, C.FIELD_WIDTH - C.OUTPOST_CENTER_Y_BLUE

    def start_positions(self, alliance: Alliance, radius: float) -> list[tuple[float, float]]:
        """Default starting spots: bumpers against the ROBOT STARTING LINE, between the crossing
        lanes so a robot that never moves doesn't block a BUMP or TRENCH."""
        x = C.ALLIANCE_ZONE_DEPTH - radius - 0.03
        spots = [(x, 1.60), (x, C.CENTER_Y), (x, C.FIELD_WIDTH - 1.60)]
        if alliance == Alliance.RED:
            spots = [C.mirror_point(px, py) for px, py in spots]
        return spots


def _boxes_array(boxes) -> np.ndarray:
    return np.array([[b.x0, b.x1, b.y0, b.y1] for b in boxes], dtype=np.float64).reshape(-1, 4)


FIELD = Field()
