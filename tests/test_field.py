"""Field geometry matches the official 2026 AprilTag layout and is point-symmetric."""

from dataclasses import replace

import pytest

from rebuilt_sim import constants as C
from rebuilt_sim.apriltags import TAGS
from rebuilt_sim.constants import Alliance
from rebuilt_sim.field import FIELD
from rebuilt_sim.robot import TIERS, RobotCommand
from rebuilt_sim.sim import Match

_TAG_Y = {t[0]: t[2] for t in TAGS}
_FAR = [(15.9, 0.6), (15.9, 1.7), (15.9, 2.8), (15.9, 5.3), (15.9, 6.4)]


def test_hub_positions_from_apriltags():
    # tag faces (2026-rebuilt-welded.json): blue HUB x 4.0219..5.2292, y 3.4312..4.6380
    blue = FIELD.hubs[Alliance.BLUE]
    assert blue.x0 == pytest.approx(4.0219, abs=0.01) and blue.x1 == pytest.approx(5.2292, abs=0.01)
    assert blue.y0 == pytest.approx(3.4312, abs=0.01) and blue.y1 == pytest.approx(4.6380, abs=0.01)
    red = FIELD.hubs[Alliance.RED]  # red HUB tags x 11.3119..12.5192
    assert red.x0 == pytest.approx(11.3119, abs=0.01) and red.x1 == pytest.approx(12.5192, abs=0.01)


def test_point_symmetry():
    for pair in (FIELD.hubs, FIELD.towers, FIELD.depots, FIELD.alliance_zones):
        b, r = pair
        assert r.x0 == pytest.approx(C.FIELD_LENGTH - b.x1)
        assert r.y0 == pytest.approx(C.FIELD_WIDTH - b.y1)


def test_crossings_tile_the_hub_line():
    # guardrail | TRENCH opening | column | BUMP | HUB | BUMP | column | TRENCH opening | guardrail
    t0, t1 = (p[Alliance.BLUE] for p in FIELD.trench_openings)
    c0, c1 = (p[Alliance.BLUE] for p in FIELD.trench_columns)
    b0, b1 = (p[Alliance.BLUE] for p in FIELD.bumps)
    hub = FIELD.hubs[Alliance.BLUE]
    assert t0.y0 == 0 and t0.y1 == pytest.approx(c0.y0)
    assert c0.y1 == pytest.approx(b0.y0) and b0.y1 == pytest.approx(hub.y0)
    assert hub.y1 == pytest.approx(b1.y0) and b1.y1 == pytest.approx(c1.y0)
    assert c1.y1 == pytest.approx(t1.y0) and t1.y1 == pytest.approx(C.FIELD_WIDTH)


def test_zone_rules():
    assert FIELD.in_alliance_zone(Alliance.BLUE, 4.3, radius=0.46)  # bumper still over the line
    assert not FIELD.in_alliance_zone(Alliance.BLUE, 4.6, radius=0.46)
    assert FIELD.in_alliance_zone(Alliance.RED, C.FIELD_LENGTH - 1.0)
    assert FIELD.fully_across_center(Alliance.BLUE, C.CENTER_X + 0.5, radius=0.46)
    assert not FIELD.fully_across_center(Alliance.BLUE, C.CENTER_X + 0.3, radius=0.46)


def test_trench_is_height_gated():
    opening = FIELD.trench_openings[0][Alliance.BLUE]
    tall = FIELD.robot_obstacles_tall
    short = FIELD.robot_obstacles_short
    assert opening in tall and opening not in short


def test_alliance_wall_elements_from_apriltags():
    # the TOWER is centered on its middle tag (31 blue, 15 red), the OUTPOST openings on tag 29 / 13
    assert C.TOWER_CENTER_Y_BLUE == pytest.approx(_TAG_Y[31], abs=0.002)
    assert C.FIELD_WIDTH - C.TOWER_CENTER_Y_BLUE == pytest.approx(_TAG_Y[15], abs=0.002)
    assert C.OUTPOST_CENTER_Y_BLUE == pytest.approx(_TAG_Y[29], abs=0.002)
    assert C.FIELD_WIDTH - C.OUTPOST_CENTER_Y_BLUE == pytest.approx(_TAG_Y[13], abs=0.002)
    # the HUB's near face (its tags sit 0.27 in proud of it) is where the ALLIANCE ZONE ends
    assert FIELD.hubs[Alliance.BLUE].x0 - 0.27 * C.IN == pytest.approx(TAGS[25][1], abs=0.001)  # tag 26
    assert FIELD.alliance_zones[Alliance.BLUE].x1 == pytest.approx(FIELD.hubs[Alliance.BLUE].x0)


def test_clearance_classes():
    assert [FIELD.clearance(TIERS[t]) for t in ("elite", "strong", "mid", "low")] == [0, 0, 2, 2]
    assert FIELD.clearance(replace(TIERS["low"], height=0.60)) == 1  # over the TRENCH arm, under the LOW RUNG
    assert FIELD.clearance(replace(TIERS["low"], height=0.75)) == 3  # too tall for the TOWER's supports


def _drive(height: float, start, v, seconds: float, radius: float = TIERS["low"].radius) -> tuple[float, float]:
    """Drive a lone blue robot of this size at a constant field-relative velocity."""
    spec = replace(TIERS["low"], height=height, radius=radius)
    m = Match([spec] + [TIERS["broken"]] * 5, seed=0, starts=[start] + _FAR)
    while m.t < seconds:
        m.step([RobotCommand(*v)] + [None] * 5)
    return m.robots[0].x, m.robots[0].y


def test_robots_meet_the_uprights_or_the_rungs():
    # straight at the TOWER WALL between the UPRIGHTS. A usual robot (0.92 m across its bumpers) is
    # wider than the 0.82 m gap; a narrow one passes under the LOW RUNG if it is short enough.
    front, upright = FIELD.tower_fronts[Alliance.BLUE], FIELD.tower_uprights[0][Alliance.BLUE]
    start = (2.5, C.TOWER_CENTER_Y_BLUE)
    x, _ = _drive(0.55, start, (-1.5, 0.0), 3.0)
    assert x > front.x1 + 0.1
    assert C.TOWER_UPRIGHT_GAP < 2 * TIERS["low"].radius and upright.y1 < C.TOWER_CENTER_Y_BLUE
    x, _ = _drive(0.60, start, (-1.5, 0.0), 3.0, radius=0.38)
    assert x < front.x0 - 0.3
    x, _ = _drive(0.70, start, (-1.5, 0.0), 3.0, radius=0.38)
    assert x >= front.x1 + 0.38 - 0.02


def test_robots_along_the_wall_pass_behind_the_uprights():
    # the TOWER is an open frame: under its supports, a robot can drive between the UPRIGHTS and the wall
    start, v = (0.5, C.TOWER_CENTER_Y_BLUE - 1.3), (0.0, 1.5)
    _, y = _drive(0.70, start, v, 3.0)
    assert y > C.TOWER_CENTER_Y_BLUE + 1.0
    _, y = _drive(0.75, start, v, 3.0)
    upright = FIELD.tower_uprights[0][Alliance.BLUE]
    assert y <= upright.y0 - TIERS["low"].radius + 0.02


def test_fuel_rolls_under_the_rungs():
    for a in Alliance:
        assert FIELD.tower_fronts[a] not in FIELD.fuel_obstacles
        assert all(u[a] in FIELD.fuel_obstacles for u in FIELD.tower_uprights)


@pytest.mark.parametrize("alliance", list(Alliance))
def test_climb_spots_are_in_the_climb_zone(alliance):
    front = FIELD.tower_fronts[alliance]
    for tier in ("elite_climber", "climber", "mid", "low"):
        rad = TIERS[tier].radius
        for x, y in FIELD.climb_spots(alliance, rad):
            assert FIELD.in_climb_zone(alliance, x, y, rad)
            assert min(abs(x - front.x0), abs(x - front.x1)) > rad  # clear of the UPRIGHTS and RUNGS
            assert front.y0 - 0.4 < y < front.y1 + 0.4
