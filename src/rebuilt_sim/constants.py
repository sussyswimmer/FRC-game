"""Field geometry, match timing and scoring constants for REBUILT (FRC 2026).

Coordinates follow WPILib's field convention (blue-alliance origin): x runs along the
field from the blue alliance wall (x = 0) to the red alliance wall, y runs across it,
units are meters. Positions and dimensions come from the official 2026 Field Dimension
Drawings (FE-2026 rev B, welded field) and the Game Manual (TU22), in the inches they are
given in; the official AprilTag layout (allwpilib v2026.2.1, ``2026-rebuilt-welded.json``)
agrees with them. The field is point-symmetric: a red element sits at (L - x, W - y) of its
blue counterpart (the TRENCHes are the one exception, see field.py). Values marked APPROX
are estimates.
"""

from __future__ import annotations

import math
from enum import IntEnum

IN = 0.0254  # meters per inch


class Alliance(IntEnum):
    BLUE = 0
    RED = 1

    @property
    def other(self) -> "Alliance":
        return Alliance(1 - self)


# --- field ------------------------------------------------------------------------
FIELD_LENGTH = 16.541
FIELD_WIDTH = 8.069
CENTER_X = FIELD_LENGTH / 2
CENTER_Y = FIELD_WIDTH / 2

# HUB: a 47 in square body; the blue one's faces are 158.61 and 205.61 in from the blue wall.
# (Its AprilTags sit 0.27 in proud of the faces.)
HUB_CENTER_BLUE = (182.11 * IN, 158.84 * IN)
HUB_HALF = 23.5 * IN
HUB_OPENING_HEIGHT = 72 * IN
HUB_PROCESS_TIME = (0.4, 0.9)  # s from entering the top to passing the counter (APPROX)
HUB_EXIT_SPEED = (1.2, 3.2)  # m/s, FUEL leaves through 4 base exits into the NEUTRAL ZONE (APPROX)

# Zones along x for the blue alliance (mirror for red)
ALLIANCE_ZONE_DEPTH = 158.61 * IN  # alliance wall to the HUB near face, the far edge of the ROBOT STARTING LINE
BAND_FAR_X = 205.61 * IN  # far face of the HUB/BUMP/TRENCH line; NEUTRAL ZONE starts here

# BUMPs flank the HUB: 73.0 in wide (y) x 44.4 in deep (x), 6.5 in tall, 15 deg ramps
BUMP_DEPTH = 44.4 * IN
BUMP_WIDTH = 73.0 * IN
BUMP_HEIGHT = 6.513 * IN

# TRENCH: arm robots drive under, centered on the HUB line (tags at y 0.6445 / 7.4248)
TRENCH_DEPTH = 47.0 * IN
TRENCH_OPENING_WIDTH = 50.34 * IN
TRENCH_CLEARANCE = 22.25 * IN  # robots must be this short or less to pass underneath

# TOWER (Game Manual 5.8, drawing GE-26500): an open frame set into the alliance wall between
# DRIVER STATIONS 2 and 3, centered on its middle AprilTag (31 blue, 15 red). A floor plate, two
# UPRIGHTS, three RUNGS between and past them, and supports from the UPRIGHTS back to the wall.
TOWER_CENTER_Y_BLUE = 147.47 * IN
TOWER_WIDTH = 49.25 * IN  # the TOWER WALL
TOWER_DEPTH = 45.0 * IN  # from the wall to the front of the floor plate
TOWER_UPRIGHT_X = (40.0 * IN, 43.51 * IN)  # the UPRIGHTS' back and front faces, from the wall
TOWER_UPRIGHT_GAP = 32.25 * IN  # between the UPRIGHTS' inner faces
TOWER_UPRIGHT_THICKNESS = 1.5 * IN
TOWER_RUNG_HALF_LENGTH = 23.5 * IN  # the RUNGS are 47 in long: 5.875 in past each UPRIGHT
TOWER_RUNG_DIAMETER = 1.66 * IN
RUNG_HEIGHTS = (27.0 * IN, 45.0 * IN, 63.0 * IN)  # LOW, MID, HIGH (centers)
LOW_RUNG_CLEARANCE = RUNG_HEIGHTS[0] - TOWER_RUNG_DIAMETER / 2  # robots this short or less pass under it
TOWER_SUPPORT_CLEARANCE = 28.4 * IN  # the supports to the wall start this high: shorter robots pass under

# OUTPOST: human-player station in the alliance-wall corner, 49.84 in wide; its CHUTE and
# CORRAL openings are centered on AprilTag 29 (blue) / 13 (red)
OUTPOST_CENTER_Y_BLUE = 26.22 * IN
OUTPOST_FEED_DEPTH = 0.55  # robot center within this distance of the wall to be fed (APPROX)
OUTPOST_FEED_HALF_WIDTH = 0.45
OUTPOST_CHUTE_FUEL = 24

# DEPOT: 42 x 27 in floor area along the alliance wall, about 47 in past the TOWER's floor plate
DEPOT_CENTER_Y_BLUE = 234.85 * IN
DEPOT_WIDTH = 42.0 * IN  # along the wall (y)
DEPOT_DEPTH = 27.0 * IN  # out from the wall (x)
DEPOT_FUEL = 24

# NEUTRAL ZONE FUEL block: about 72 in (x) by 206 in (y), centered on the field, split at the center line
NEUTRAL_BLOCK_SIZE = (72 * IN, 206 * IN)
NEUTRAL_BLOCK_GRID = (12, 34)  # 408 slots = 360 staged + up to 48 un-preloaded FUEL

# --- game pieces --------------------------------------------------------------------
FUEL_RADIUS = 5.91 * IN / 2
FUEL_TOTAL = 504
MAX_PRELOAD = 8
# where each FUEL is, as tracked by the simulator
GROUND, HELD, FLIGHT, HUB, CHUTE = range(5)

# --- match timing (seconds since AUTO starts) ------------------------------------------
AUTO_LEN = 20.0
PAUSE_LEN = 3.0  # clock holds at 2:20 after AUTO; robots disabled, in-flight FUEL still counts for AUTO
TRANSITION_LEN = 10.0
SHIFT_LEN = 25.0
N_SHIFTS = 4
ENDGAME_LEN = 30.0
GRACE = 3.0  # FUEL still counts this long after a HUB deactivates and after AUTO / the match ends

TELEOP_START = AUTO_LEN + PAUSE_LEN
SHIFT1_START = TELEOP_START + TRANSITION_LEN
ENDGAME_START = SHIFT1_START + N_SHIFTS * SHIFT_LEN
MATCH_END = ENDGAME_START + ENDGAME_LEN  # 163 s: the arena clock reaches 0:00
FINAL_TIME = MATCH_END + GRACE  # scores are final once the last grace window closes

# --- scoring -----------------------------------------------------------------------
FUEL_POINTS = 1
AUTO_TOWER_L1_POINTS = 15
AUTO_TOWER_MAX_ROBOTS = 2
TELEOP_TOWER_POINTS = {1: 10, 2: 20, 3: 30}
MINOR_FOUL = 5
MAJOR_FOUL = 15
WIN_RP = 3
TIE_RP = 1

# (ENERGIZED FUEL, SUPERCHARGED FUEL, TRAVERSAL TOWER points) per event level
RP_THRESHOLDS = {
    "regional": (100, 360, 50),
    "district": (100, 360, 50),
    "dcmp": (240, 360, 50),
    "cmp": (360, 500, 50),
}

# --- rules the simulator enforces ---------------------------------------------------
PIN_LIMIT = 3.0  # G418: MINOR after 3 s of pinning, then a MAJOR every further 3 s
PIN_RELEASE_DISTANCE = 72 * IN

# --- high-fidelity physics (docs/03-driving-and-aiming.md) ---------------------------------
GRAVITY = 9.81
# The HUB top is a hexagonal funnel opening 72 in above the carpet, 41.73 in across the flats inside
# (drawing GE-26300). Its flats face the guardrails and its corners point at the alliance walls.
HUB_OPENING_ACROSS = 41.73 * IN
FUEL_MASS = 0.227  # kg, about 0.5 lb (APPROX)
FUEL_DRAG_COEF = 0.5  # sphere-like foam ball (APPROX)
AIR_DENSITY = 1.2  # kg/m^3
# 1/m: air drag slows FUEL by FUEL_DRAG * speed^2
FUEL_DRAG = 0.5 * AIR_DENSITY * FUEL_DRAG_COEF * math.pi * FUEL_RADIUS ** 2 / FUEL_MASS
BUMPER_DEPTH = 3.25 * IN  # bumper thickness outside the frame
MODULE_INSET = 0.065  # m from the frame edge to a swerve module's wheel (MK4i-style modules)
ROBOT_MAX_HEIGHT = 30.0 * IN  # R107


def mirror_point(x: float, y: float) -> tuple[float, float]:
    """Map a blue-side point to its red counterpart (point symmetry about the field center)."""
    return FIELD_LENGTH - x, FIELD_WIDTH - y
