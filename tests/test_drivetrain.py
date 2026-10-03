"""High-fidelity driving: swerve dynamics, command latency and rectangular bumpers."""

import math
from dataclasses import replace

from rebuilt_sim import constants as C
from rebuilt_sim.collision import aabb_box, box_contact, corners, robot_box
from rebuilt_sim.robot import TIERS, DriveSpec, RobotCommand
from rebuilt_sim.sim import HiFiConfig, Match, MatchConfig

FAR = [(15.9, 0.6), (15.9, 1.7), (15.9, 2.8), (15.9, 5.3), (15.9, 6.4)]


def _solo(spec, start, **hifi) -> Match:
    hifi.setdefault("sensor_noise", 0.0)
    return Match([spec] + [TIERS["broken"]] * 5, MatchConfig(hifi=HiFiConfig(**hifi)), seed=0,
                 starts=[start] + FAR)


def _drive(m: Match, cmd: RobotCommand, seconds: float) -> None:
    end = m.t + seconds
    while m.t < end - 1e-9:
        m.step([cmd] + [None] * 5)


def test_strategy_physics_is_the_default():
    m = Match([TIERS["mid"]] * 6, seed=0)
    assert m.cfg.dt == 0.05 and m.hifi is None
    assert m.drive is None and m.collider is None and m.shooters is None and m.sensors is None
    assert MatchConfig(hifi=HiFiConfig()).dt == 0.02


def test_swerve_accelerates_like_the_tier():
    for tier in ("elite", "strong", "mid", "low"):
        spec = TIERS[tier]
        m = _solo(spec, (5.8, 2.6), latency=0.0)
        r = m.robots[0]
        peak, prev = 0.0, 0.0
        while m.t < 1.2:
            m.step([RobotCommand(vx=spec.max_speed)] + [None] * 5)
            peak, prev = max(peak, (r.vx - prev) / m.cfg.dt), r.vx
        assert abs(peak - spec.max_accel) < 0.1 * spec.max_accel, tier  # current limits match max_accel
        assert abs(r.vx - spec.max_speed) < 0.05, tier
        assert abs(r.vy) < 0.01 and abs(r.heading) < 1e-3, tier


def test_traction_caps_acceleration():
    grippy = replace(TIERS["strong"], drive=DriveSpec(current_limit=150.0))  # motors could beat the tires
    m = _solo(grippy, (5.8, 2.6), latency=0.0)
    r = m.robots[0]
    peak, prev, slipped = 0.0, 0.0, False
    while m.t < 0.5:
        m.step([RobotCommand(vx=4.0)] + [None] * 5)
        peak, prev = max(peak, (r.vx - prev) / m.cfg.dt), r.vx
        slipped |= bool(m.drive.slipping[0].any())
    assert slipped
    assert peak <= grippy.drive.wheel_cof * C.GRAVITY + 0.2


def test_modules_must_turn_before_the_robot_changes_direction():
    m = _solo(TIERS["strong"], (6.0, 1.5), latency=0.0)
    r = m.robots[0]
    _drive(m, RobotCommand(vx=3.0), 1.0)
    m.step([RobotCommand(vy=3.0)] + [None] * 5)
    assert r.vx > 2.0  # still carried along +x: the wheels were pointing that way
    _drive(m, RobotCommand(vy=3.0), 1.5)
    assert abs(r.vx) < 0.05 and abs(r.vy - 3.0) < 0.05


def test_stopped_robot_holds_still():
    m = _solo(TIERS["mid"], (6.0, 4.0), latency=0.0)
    r = m.robots[0]
    _drive(m, RobotCommand(vx=3.0, omega=2.0), 0.6)
    _drive(m, RobotCommand(), 1.0)
    x, y, h = r.x, r.y, r.heading
    _drive(m, RobotCommand(), 2.0)
    assert math.hypot(r.x - x, r.y - y) < 0.01 and abs(r.heading - h) < 0.01


def test_latency_delays_every_command():
    for latency, first_move in ((0.04, 3), (0.0, 1)):
        m = _solo(TIERS["strong"], (6.0, 4.0), latency=latency)
        steps = 0
        while m.robots[0].speed == 0.0:
            m.step([RobotCommand(vx=3.0)] + [None] * 5)
            steps += 1
        assert steps == first_move


def test_stronger_drive_wins_the_pushing_match():
    def shove(pusher: str, victim: str) -> float:
        m = Match([TIERS[pusher], TIERS[victim]] + [TIERS["broken"]] * 4,
                  MatchConfig(hifi=HiFiConfig(sensor_noise=0.0, latency=0.0)), seed=0,
                  starts=[(6.0, 4.0), (6.9, 4.0)] + FAR[:4])
        while m.t < 3.0:
            m.step([RobotCommand(vx=m.robots[0].spec.max_speed), RobotCommand()] + [None] * 4)
        return m.robots[1].x - 6.9
    assert shove("elite", "low") > 2.0
    assert shove("low", "elite") < 1.0


def test_off_center_hit_spins_the_other_robot():
    def hit(offset: float) -> float:
        m = Match([TIERS["elite"], TIERS["mid"]] + [TIERS["broken"]] * 4,
                  MatchConfig(hifi=HiFiConfig(sensor_noise=0.0, latency=0.0)), seed=0,
                  starts=[(6.0, 4.0), (7.5, 4.0 + offset)] + FAR[:4])
        victim = m.robots[1]
        while m.t < 1.0:
            m.step([RobotCommand(vx=4.5), None] + [None] * 4)  # the victim is disabled: it coasts
        return abs(victim.heading)
    assert hit(0.0) < 0.01  # square hit: no spin
    assert hit(0.35) > 0.1  # corner hit: it turns


def test_bumpers_stay_on_the_field_and_out_of_each_other():
    specs = [TIERS[t] for t in ("elite", "strong", "mid", "low", "mid", "strong")]
    m = Match(specs, MatchConfig(hifi=HiFiConfig(sensor_noise=0.0)), seed=3)
    for k in range(600):
        a = k * 0.05
        cmds = [RobotCommand(vx=5.0 * math.cos(a + i), vy=5.0 * math.sin(1.3 * a + i), omega=3.0 * math.sin(a - i))
                for i in range(6)]
        m.step(cmds)
        for r in m.robots:
            for x, y in corners(robot_box(r)):
                assert -0.02 <= x <= C.FIELD_LENGTH + 0.02 and -0.02 <= y <= C.FIELD_WIDTH + 0.02
            for box in m._robot_boxes[not r.spec.can_trench]:
                hit = box_contact(aabb_box(*box), robot_box(r))
                assert hit is None or hit[2] < 0.03
        for i in range(6):
            for j in range(i + 1, 6):
                hit = box_contact(robot_box(m.robots[i]), robot_box(m.robots[j]))
                assert hit is None or hit[2] < 0.05


def test_only_short_robots_fit_under_the_trench():
    lane = (C.TRENCH_OPENING_WIDTH / 2)  # centered in the blue side's lower TRENCH opening
    for tier, passes in (("strong", True), ("mid", False)):
        m = _solo(TIERS[tier], (3.0, lane))
        _drive(m, RobotCommand(vx=2.0), 3.0)
        assert (m.robots[0].x > 5.3) == passes, tier


def test_zone_rules_use_the_bumper_footprint():
    m = _solo(TIERS["strong"], (3.0, 2.0))
    r = m.robots[0]
    assert r.extent_x == r.spec.length / 2
    r.heading = math.pi / 4
    assert math.isclose(r.extent_x, (r.spec.length + r.spec.width) / 2 / math.sqrt(2))
    standard = Match([TIERS["strong"]] + [TIERS["broken"]] * 5, seed=0)
    assert standard.robots[0].extent_x == TIERS["strong"].radius
