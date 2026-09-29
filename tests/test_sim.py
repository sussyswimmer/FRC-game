"""Simulator invariants: FUEL conservation, determinism, scoring only into active HUBs."""

import math

from rebuilt_sim import constants as C
from rebuilt_sim.bots import ScriptedPolicy
from rebuilt_sim.constants import Alliance
from rebuilt_sim.robot import RobotCommand
from rebuilt_sim.sim import make_match


def test_fuel_is_conserved_over_a_match():
    m = make_match(["elite", "strong", "mid"], ["strong", "low", "broken"], seed=11)
    assert sum(m.fuel_census().values()) == C.FUEL_TOTAL
    pol = ScriptedPolicy(m, seed=11)
    while not m.done:
        m.step(pol())
        if m.step_count % 400 == 0:
            assert sum(m.fuel_census().values()) == C.FUEL_TOTAL
    assert sum(m.fuel_census().values()) == C.FUEL_TOTAL
    assert m.t >= C.FINAL_TIME - 1e-6


def test_starting_fuel_layout():
    m = make_match(["mid"] * 3, ["mid"] * 3, seed=0)
    census = m.fuel_census()
    assert census["held"] == 6 * C.MAX_PRELOAD
    assert census["chute"] == 2 * C.OUTPOST_CHUTE_FUEL
    assert m.depot_count(Alliance.BLUE) == C.DEPOT_FUEL and m.depot_count(Alliance.RED) == C.DEPOT_FUEL
    assert census["ground"] == 504 - 48 - 48


def test_same_seed_same_match():
    results = []
    for _ in range(2):
        m = make_match(["elite", "mid", "low"], ["strong", "mid", "low"], seed=3)
        s = m.run(ScriptedPolicy(m, seed=3))
        results.append([(x.auto_fuel, x.teleop_fuel, x.total) for x in s.scores])
    assert results[0] == results[1]


def _aim_and_fire(m, index):
    r = m.robots[index]
    hx, hy = m.field.hub_centers[r.alliance]
    err = (math.atan2(hy - r.y, hx - r.x) - r.heading + math.pi) % (2 * math.pi) - math.pi
    return RobotCommand(0.0, 0.0, 6.0 * err, shoot=True)


def test_preload_scores_in_auto():
    m = make_match(["strong", "broken", "broken"], ["broken"] * 3, seed=1)
    while m.t < C.TELEOP_START:
        m.step([_aim_and_fire(m, 0)] + [None] * 5)
    blue = m.scores[Alliance.BLUE]
    assert blue.auto_fuel > 0
    assert blue.auto_fuel == m.robots[0].stats.fuel_scored
    assert m.robots[0].stats.fuel_shot == C.MAX_PRELOAD


def test_inactive_hub_scores_nothing():
    m = make_match(["strong", "broken", "broken"], ["broken"] * 3, seed=2)
    idle = [None] * 6
    while m.t < C.TELEOP_START:  # blue scores nothing in AUTO -> tie -> coin flip decides
        m.step(idle)
    blue_off = m.schedule.first_inactive == Alliance.BLUE
    # a window where the blue HUB is off: SHIFT 1 if the coin made blue "first inactive", else SHIFT 2
    start = C.SHIFT1_START if blue_off else C.SHIFT1_START + C.SHIFT_LEN
    # wait out the 3 s grace: FUEL counted within 3 s of the switch still scores (manual 6.5)
    while m.t < start + C.GRACE + 0.5:
        m.step(idle)
    before = m.scores[Alliance.BLUE].fuel
    while m.t < start + 15:
        m.step([_aim_and_fire(m, 0)] + [None] * 5)
    assert m.robots[0].stats.fuel_shot > 0
    assert m.scores[Alliance.BLUE].fuel == before
    assert m.scores[Alliance.BLUE].wasted_fuel > 0


def test_shooting_outside_zone_is_a_major_foul():
    m = make_match(["elite", "broken", "broken"], ["broken"] * 3, seed=4,)
    m.robots[0].x = 6.0  # in the neutral zone, within the elite's range of its HUB
    m.robots[0].y = C.CENTER_Y
    for _ in range(20):
        m.step([RobotCommand(shoot=True)] + [None] * 5)
    assert any(f.rule.startswith("G407") for f in m.fouls.fouls)
    assert m.scores[Alliance.RED].penalty_points >= C.MAJOR_FOUL


def test_robots_stay_on_the_field():
    m = make_match(["mid"] * 3, ["mid"] * 3, seed=5)
    for _ in range(300):
        m.step([RobotCommand(vx=-5.0, vy=5.0)] * 6)
    for r in m.robots:
        assert r.spec.radius - 1e-6 <= r.x <= C.FIELD_LENGTH - r.spec.radius + 1e-6
        assert r.spec.radius - 1e-6 <= r.y <= C.FIELD_WIDTH - r.spec.radius + 1e-6
