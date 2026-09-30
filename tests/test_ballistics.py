"""High-fidelity shooting: FUEL flight, the HUB opening, and the robots' aim software."""

import math
from dataclasses import replace

import numpy as np

from rebuilt_sim import constants as C
from rebuilt_sim.ballistics import APOTHEM, flight_step, in_opening, shot_table
from rebuilt_sim.constants import Alliance
from rebuilt_sim.field import FIELD
from rebuilt_sim.robot import TIERS, RobotCommand
from rebuilt_sim.sim import FLIGHT, GROUND, HiFiConfig, Match, MatchConfig

FAR = [(15.9, 0.6), (15.9, 1.7), (15.9, 2.8), (15.9, 5.3), (15.9, 6.4)]
HUB_X, HUB_Y = FIELD.hub_centers[Alliance.BLUE]


def _perfect(tier: str):
    """A robot whose shooter has no shot-to-shot spread and never jams."""
    spec = TIERS[tier]
    return replace(spec, jam_rate=0.0, shooter=replace(spec.shooter, speed_sd=0.0, angle_sd=0.0))


def _shooter_at(spec, distance: float, angle: float = 0.0, **hifi) -> Match:
    """A robot ``distance`` m from the blue HUB, facing it, with nobody else on the field."""
    hifi.setdefault("sensor_noise", 0.0)
    x, y = HUB_X - distance * math.cos(angle), HUB_Y - distance * math.sin(angle)
    m = Match([spec] + [TIERS["broken"]] * 5, MatchConfig(hifi=HiFiConfig(**hifi)), seed=0, starts=[(x, y)] + FAR)
    m.robots[0].heading = angle
    return m


def _fire_all(m: Match, cmd: RobotCommand, until: float = 16.0) -> None:
    while m.t < until and (m.robots[0].fuel > 0 or m.fuel_census()["flight"] or m.fuel_census()["hub"]):
        m.step([cmd] + [None] * 5)


def test_hexagonal_opening():
    assert in_opening(0.0, 0.0) and in_opening(APOTHEM - 0.01, 0.0)
    assert not in_opening(APOTHEM + 0.01, 0.0)
    assert in_opening(0.0, APOTHEM + 0.05)  # a corner points along y: more room that way
    assert not in_opening(0.0, APOTHEM / math.cos(math.pi / 6) + 0.01)


def test_drag_shortens_the_flight():
    def landing(drag: bool) -> float:
        pos, z = np.zeros((1, 2)), np.array([0.5])
        vel, vz = np.array([[7.0, 0.0]]), np.array([7.0])
        saved = C.FUEL_DRAG
        try:
            C.FUEL_DRAG = saved if drag else 0.0
            while z[0] > 0:
                flight_step(pos, z, vel, vz, 0.001)
        finally:
            C.FUEL_DRAG = saved
        return float(pos[0, 0])
    vacuum = 7.0 * (7.0 + math.sqrt(49 + 2 * C.GRAVITY * 0.5)) / C.GRAVITY
    assert abs(landing(False) - vacuum) < 0.02
    assert landing(True) < 0.9 * vacuum


def test_every_tier_can_score_from_where_it_shoots():
    for tier in ("elite", "strong", "mid", "low", "climber", "elite_climber"):
        spec = TIERS[tier]
        lo, hi = shot_table(spec.shooter, 0.02).reach()
        assert lo <= spec.sweet_range - 0.2 and hi >= spec.max_range, tier


def test_aim_software_scores_every_shot_without_spread():
    corner = math.atan2(HUB_Y, HUB_X)  # toward the field corner: the longest shot in the ALLIANCE ZONE
    for tier, d, angle in (("elite", 3.2, 0.4), ("strong", 2.4, 0.4), ("mid", 2.0, -0.3), ("low", 1.6, 0.0),
                           ("elite", 5.5, corner)):
        m = _shooter_at(_perfect(tier), d, angle=angle)
        r = m.robots[0]
        preload = r.fuel
        _fire_all(m, RobotCommand(shoot=True))
        assert r.stats.fuel_shot == preload and m.scores[Alliance.BLUE].fuel == preload, (tier, d)


def test_first_shot_waits_for_the_flywheel():
    m = _shooter_at(_perfect("strong"), 2.4)
    r = m.robots[0]
    while r.stats.fuel_shot == 0:
        m.step([RobotCommand(shoot=True)] + [None] * 5)
    assert m.t > r.spec.shooter.spinup  # it spun up first
    assert abs(r.flywheel / (1 - r.spec.shooter.recovery) - m.shooters.targets[0][0]) < 0.03 * r.flywheel


def test_short_and_long_shots_land_on_the_carpet():
    for speed, beyond in ((5.0, False), (10.0, True)):
        m = _shooter_at(_perfect("strong"), 2.4)
        hood = m.robots[0].spec.shooter.hood_range[0]
        while m.t < 2.0:  # spin up
            m.step([RobotCommand(shot_speed=speed, hood=hood)] + [None] * 5)
        while not (m.f_state == FLIGHT).any():
            m.step([RobotCommand(shoot=True, shot_speed=speed, hood=hood)] + [None] * 5)
        ball = int(np.flatnonzero(m.f_state == FLIGHT)[0])
        while m.f_state[ball] == FLIGHT:
            m.step([RobotCommand(shot_speed=speed, hood=hood)] + [None] * 5)
        assert m.f_state[ball] == GROUND and m.scores[Alliance.BLUE].fuel == 0
        hub = FIELD.hubs[Alliance.BLUE]
        assert (m.f_pos[ball, 0] > hub.x1) == beyond, speed


def test_fuel_dropped_on_the_rim_bounces_off():
    m = _shooter_at(_perfect("strong"), 2.4)
    i = int(np.flatnonzero(m.f_state == GROUND)[0])
    m.f_state[i] = FLIGHT
    m.f_shooter[i] = 0
    m.f_pos[i] = (HUB_X + C.HUB_HALF - 0.02, HUB_Y)  # over the HUB's top edge, outside the opening
    m.f_vel[i] = (0.0, 0.0)
    m.f_z[i], m.f_vz[i] = C.HUB_OPENING_HEIGHT + 0.3, 0.0
    while m.f_state[i] == FLIGHT:
        m.step([None] * 6)
    assert m.f_state[i] == GROUND and m.f_pos[i, 0] > HUB_X + C.HUB_HALF  # rolled off the outside


def test_moving_shots_need_the_aim_software_to_lead():
    def moving_hits(lead: bool) -> int:
        spec = _perfect("elite")
        spec = replace(spec, shooter=replace(spec.shooter, lead=lead))
        m = _shooter_at(spec, 3.0, angle=0.0)
        r = m.robots[0]
        r.x, r.y = HUB_X - 3.0, HUB_Y - 2.0
        while m.t < 0.6:  # spin up, then sweep across in front of the HUB at 1.5 m/s
            m.step([RobotCommand(vy=0.0)] + [None] * 5)
        while m.t < 4.0:
            m.step([RobotCommand(vy=1.5 if r.y < HUB_Y + 2.0 else -1.5, shoot=True)] + [None] * 5)
        while m.fuel_census()["flight"] or m.fuel_census()["hub"]:
            m.step([None] * 6)
        return m.scores[Alliance.BLUE].fuel
    assert moving_hits(lead=True) >= 7
    assert moving_hits(lead=False) <= 2


def test_manual_aim_fires_as_commanded():
    m = _shooter_at(_perfect("strong"), 2.4)
    r = m.robots[0]
    sol = m.shooters.tables[0].solve(2.4)
    hood, speed = sol[0], sol[1]
    while m.t < 1.5:  # spin up and set the hood without firing
        m.step([RobotCommand(shot_speed=speed, hood=hood)] + [None] * 5)
    _fire_all(m, RobotCommand(shoot=True, shot_speed=speed, hood=hood))
    assert r.stats.fuel_shot == C.MAX_PRELOAD and m.scores[Alliance.BLUE].fuel >= C.MAX_PRELOAD - 1
