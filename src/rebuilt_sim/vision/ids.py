"""Object ids in the renderer's id buffer (and the instance masks a dataset can save).

Every labeled thing has its own id, so a pixel's id says exactly what it shows. -1 is nothing (the
arena backdrop).
"""

from __future__ import annotations

FUEL = 0  # FUEL i -> i (0..503)
ROBOT = 1000  # robot i (0-2 blue, 3-5 red) -> 1000 + i
TAG = 2000  # AprilTag n -> 2000 + n (the printed 10 x 10 square)
HUB = 3000  # + alliance (0 blue, 1 red); includes its cap, funnel, net and tag backing panels
TOWER = 3010  # + alliance
OUTPOST = 3020  # + alliance
DEPOT = 3030  # + alliance
BUMP = 3040  # + 2 * alliance + side (0: low y, 1: high y, for both alliances)
TRENCH = 3050  # + 2 * alliance + side (0: low y, the fixed TRENCH on the scoring-table side; 1: high y)
ALLIANCE_WALL = 3100  # + alliance: driver stations, TOWER wall, team signs
GUARDRAIL = 3110
CARPET = 3120
