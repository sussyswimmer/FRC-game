"""Static field geometry: obstacles, zones, BUMPs, TRENCHes, TOWERs, DEPOTs, OUTPOSTs.

All boxes are axis-aligned. Blue elements are defined directly; red ones are mirrored
through the field center (the REBUILT field is point-symmetric).
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
_DEPOT = Box(0.0, C.DEPOT_DEPTH, C.DEPOT_CENTER_Y_BLUE - C.DEPOT_WIDTH / 2, C.DEPOT_CENTER_Y_BLUE + C.DEPOT_WIDTH / 2)
_ALLIANCE_ZONE = Box(0.0, C.ALLIANCE_ZONE_DEPTH, 0.0, C.FIELD_WIDTH)
NEUTRAL_ZONE = Box(C.BAND_FAR_X, C.FIELD_LENGTH - C.BAND_FAR_X, 0.0, C.FIELD_WIDTH)

# Climbing positions across the TOWER face, as y offsets from its center. APPROX: the manual
# credits up to 2 robots in AUTO and TRAVERSAL needs 50 TOWER points, so at least two robots
# can hang at once; three side-by-side positions is an assumption to check against the drawings.
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
        self.towers = _pair(_TOWER)
        self.depots = _pair(_DEPOT)
        self.alliance_zones = _pair(_ALLIANCE_ZONE)
        self.neutral_zone = NEUTRAL_ZONE

        all_bumps = [b for pair in self.bumps for b in pair]
        self.bump_boxes = tuple(all_bumps)
        columns = [b for pair in self.trench_columns for b in pair]
        openings = [b for pair in self.trench_openings for b in pair]
        # Robots taller than the TRENCH clearance treat the openings as walls.
        self.robot_obstacles_short = tuple(list(self.hubs) + columns + list(self.towers))
        self.robot_obstacles_tall = tuple(list(self.robot_obstacles_short) + openings)
        # FUEL rolls under the TRENCH arm but not through HUBs, columns or TOWERs.
        self.fuel_obstacles = self.robot_obstacles_short

        self.obstacle_arrays = {
            False: _boxes_array(self.robot_obstacles_short),
            True: _boxes_array(self.robot_obstacles_tall),
        }
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
        """Bumpers against the TOWER face at one of its climbing positions."""
        tower = self.towers[alliance]
        cy = tower.center[1]
        if min(abs(y - (cy + off)) for off in CLIMB_SLOT_OFFSETS) > 0.3:
            return False
        if alliance == Alliance.BLUE:
            gap = (x - radius) - tower.x1
        else:
            gap = tower.x0 - (x + radius)
        return -0.05 <= gap <= 0.25

    def climb_spots(self, alliance: Alliance, radius: float) -> list[tuple[float, float]]:
        """Where a robot parks to climb, one per climbing position across the TOWER face."""
        tower = self.towers[alliance]
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
