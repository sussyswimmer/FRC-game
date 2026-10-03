"""High-fidelity matches end to end: sensors, observations, invariants and environments."""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

import rebuilt_sim.env  # noqa: F401  (registers the gym ids)
from rebuilt_sim import constants as C
from rebuilt_sim import sensors
from rebuilt_sim.apriltags import TAGS
from rebuilt_sim.bots import ScriptedPolicy
from rebuilt_sim.controller import MacroController
from rebuilt_sim.env import EnvConfig, MatchRunner, RebuiltEnv
from rebuilt_sim.obs import OBS_SIZE_HIFI, observe
from rebuilt_sim.robot import TIERS, RobotCommand
from rebuilt_sim.sim import HiFiConfig, Match, MatchConfig, make_match

ROOT = Path(__file__).resolve().parents[1]
FAR = [(15.9, 0.6), (15.9, 1.7), (15.9, 2.8), (15.9, 5.3), (15.9, 6.4)]


def test_tags_match_the_official_layout():
    layout = json.loads((ROOT / "data" / "2026-rebuilt-welded.json").read_text())["tags"]
    assert len(TAGS) == len(layout) == 32
    for (tid, x, y, z, yaw), tag in zip(TAGS, layout):
        p, q = tag["pose"]["translation"], tag["pose"]["rotation"]["quaternion"]
        assert tid == tag["ID"]
        assert abs(x - p["x"]) < 1e-6 and abs(y - p["y"]) < 1e-6 and abs(z - p["z"]) < 1e-5
        facing = math.degrees(2 * math.atan2(q["Z"], q["W"]))
        assert abs((facing - yaw + 180) % 360 - 180) < 0.1


def _hifi_match(seed: int) -> Match:
    return make_match(["elite", "strong", "mid"], ["strong", "low", "elite_climber"], seed=seed,
                      config=MatchConfig(hifi=HiFiConfig()))


def test_high_fidelity_match_invariants():
    """A full match with bots: FUEL conserved, the elite robot still carries, pose estimates sane."""
    m = _hifi_match(7)
    pol = ScriptedPolicy(m, seed=7)
    errors = []
    while not m.done:
        m.step(pol())
        if m.step_count % 100 == 0:
            assert sum(m.fuel_census().values()) == C.FUEL_TOTAL
            errors += [math.hypot(m.perceived(i)[0] - r.x, m.perceived(i)[1] - r.y) for i, r in enumerate(m.robots)]
    assert sum(m.fuel_census().values()) == C.FUEL_TOTAL
    assert m.scores[0].fuel > 100 and m.robots[0].stats.fuel_scored > 80
    assert np.median(errors) < 0.05 and max(errors) < 1.5  # a broken estimator drifts meters


def test_same_seed_same_high_fidelity_match():
    results = []
    for _ in range(2):
        m = _hifi_match(8)
        pol = ScriptedPolicy(m, seed=8)
        while m.t < 40.0:
            m.step(pol())
        results.append(([(s.auto_fuel, s.teleop_fuel, s.total) for s in m.scores],
                        [(r.x, r.y, r.heading) for r in m.robots], m.perceived(0)))
    assert results[0] == results[1]


def test_perfect_sensing_sees_the_truth():
    m = make_match(["strong"] * 3, ["mid"] * 3, seed=1, config=MatchConfig(hifi=HiFiConfig(sensor_noise=0.0)))
    pol = ScriptedPolicy(m, seed=1)
    while m.t < 8.0:
        m.step(pol())
    assert m.sensors is None
    for i, r in enumerate(m.robots):
        assert m.perceived(i) == (r.x, r.y, r.heading, r.vx, r.vy)


def test_odometry_drifts_when_shoved_and_vision_pulls_it_back(monkeypatch):
    elite = TIERS["elite"]
    bully = replace(elite, drive=replace(elite.drive, current_limit=120.0, wheel_cof=1.6))  # out-grips the victim
    m = Match([bully, TIERS["mid"]] + [TIERS["broken"]] * 4, MatchConfig(hifi=HiFiConfig()), seed=0,
              starts=[(6.3, 3.0), (6.3, 3.9)] + FAR[:4])
    victim = m.robots[1]
    monkeypatch.setattr(sensors, "VISION_RANGE", 0.0)  # cameras blind: odometry only
    while m.t < 2.5:  # shoved sideways, the braking robot's wheels slide instead of rolling
        m.step([RobotCommand(vy=4.5), RobotCommand()] + [None] * 4)
    drift = math.hypot(m.perceived(1)[0] - victim.x, m.perceived(1)[1] - victim.y)
    assert victim.y > 5.0 and drift > 0.5
    monkeypatch.setattr(sensors, "VISION_RANGE", 5.0)
    while m.t < 4.0:
        m.step([RobotCommand(), RobotCommand()] + [None] * 4)
    assert math.hypot(m.perceived(1)[0] - victim.x, m.perceived(1)[1] - victim.y) < 0.1


def test_observation_is_built_from_the_pose_estimate():
    m = make_match(["strong"] * 3, ["mid"] * 3, seed=2, config=MatchConfig(hifi=HiFiConfig(sensor_noise=3.0)))
    pol = ScriptedPolicy(m, seed=2)
    while m.t < 12.0:
        m.step(pol())
    ctl = MacroController(m)
    for i in (0, 4):
        o = observe(m, ctl, i)
        assert o.shape == (OBS_SIZE_HIFI,) and np.abs(o).max() <= 2.0
        x, y = m.perceived(i)[:2]
        if m.robots[i].alliance == 1:
            x, y = C.FIELD_LENGTH - x, C.FIELD_WIDTH - y
        assert np.allclose(o[17:19], (x / C.FIELD_LENGTH, y / C.FIELD_WIDTH), atol=1e-6)
        r = m.robots[i]
        assert np.isclose(o[162], r.flywheel / 15.0, atol=1e-6)


def test_high_fidelity_envs_pass_the_checker():
    for mode in ("macro", "continuous", "continuous_aim"):
        env = RebuiltEnv(action_mode=mode, hifi=HiFiConfig())
        check_env(env, skip_render_check=True)
        assert env.observation_space.shape == (OBS_SIZE_HIFI,)
        env.close()


def test_continuous_aim_sets_the_shooter():
    runner = MatchRunner(EnvConfig(action_mode="continuous_aim", hifi=HiFiConfig()), [0])
    runner.reset(np.random.default_rng(0), {0: "elite"})
    sh = runner.match.robots[0].spec.shooter
    cmd = runner.to_command(0, np.array([0, 0, 0, 0, 1, 0, 1.0, -1.0, 0.5]))
    assert cmd.shoot and cmd.shot_speed == sh.speed_range[1] and cmd.hood == sh.hood_range[0]
    assert math.isclose(cmd.turret, 0.5 * sh.turret_range)
    try:
        EnvConfig(action_mode="continuous_aim")
    except ValueError:
        pass
    else:
        raise AssertionError("continuous_aim without the high-fidelity ballistics should be refused")
