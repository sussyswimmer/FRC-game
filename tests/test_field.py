"""Field geometry matches the official 2026 AprilTag layout and is point-symmetric."""

import pytest

from rebuilt_sim import constants as C
from rebuilt_sim.constants import Alliance
from rebuilt_sim.field import FIELD


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
