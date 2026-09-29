"""Field geometry, match timing and scoring constants for REBUILT (FRC 2026).

Coordinates follow WPILib's field convention (blue-alliance origin): x runs along the
field from the blue alliance wall (x = 0) to the red alliance wall, y runs across it,
units are meters. Element positions are derived from the official 2026 AprilTag layout
(allwpilib v2026.2.1, ``2026-rebuilt-welded.json``) and dimensions from the Game Manual
(TU22). The field is point-symmetric: a red element sits at (L - x, W - y) of its blue
counterpart. Values marked APPROX are estimates from the event broadcast and should be
checked against the official field drawings.
"""

from __future__ import annotations

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

# HUB (from HUB tag faces: x 4.0219..5.2292, y 3.4312..4.6380 on the blue side)
HUB_CENTER_BLUE = (4.6256, 4.0346)
HUB_HALF = 0.6035  # 47 in square footprint
HUB_OPENING_HEIGHT = 72 * IN
HUB_PROCESS_TIME = (0.4, 0.9)  # s from entering the top to passing the counter (APPROX)
HUB_EXIT_SPEED = (1.2, 3.2)  # m/s, FUEL leaves through 4 base exits into the NEUTRAL ZONE (APPROX)

# Zones along x for the blue alliance (mirror for red)
ALLIANCE_ZONE_DEPTH = 4.0219  # alliance wall to the HUB near face / ROBOT STARTING LINE
BAND_FAR_X = 5.2292  # far face of the HUB/BUMP/TRENCH line; NEUTRAL ZONE starts here

# BUMPs flank the HUB: 73.0 in wide (y) x 44.4 in deep (x), 6.5 in tall, 15 deg ramps
BUMP_DEPTH = 44.4 * IN
BUMP_WIDTH = 73.0 * IN
BUMP_HEIGHT = 6.513 * IN

# TRENCH: arm robots drive under, centered on the HUB line (tags at y 0.6445 / 7.4248)
TRENCH_DEPTH = 47.0 * IN
TRENCH_OPENING_WIDTH = 50.34 * IN
TRENCH_CLEARANCE = 22.25 * IN  # robots must be this short or less to pass underneath

# TOWER: set into the alliance wall between driver stations 2 and 3 (tags at y 3.7457 / 4.1775)
TOWER_CENTER_Y_BLUE = 3.9616
TOWER_WIDTH = 49.25 * IN
TOWER_DEPTH = 45.0 * IN
RUNG_HEIGHTS = (27.0 * IN, 45.0 * IN, 63.0 * IN)  # LOW, MID, HIGH

# OUTPOST: human-player station at the corner of the alliance wall (tags at y 0.6660 / 1.0978)
OUTPOST_CENTER_Y_BLUE = 0.8819
OUTPOST_FEED_DEPTH = 0.55  # robot center within this distance of the wall to be fed (APPROX)
OUTPOST_FEED_HALF_WIDTH = 0.45
OUTPOST_CHUTE_FUEL = 24

# DEPOT: 42 x 27 in floor area along the alliance wall, opposite corner from the OUTPOST (APPROX y)
DEPOT_CENTER_Y_BLUE = 7.03
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


def mirror_point(x: float, y: float) -> tuple[float, float]:
    """Map a blue-side point to its red counterpart (point symmetry about the field center)."""
    return FIELD_LENGTH - x, FIELD_WIDTH - y
