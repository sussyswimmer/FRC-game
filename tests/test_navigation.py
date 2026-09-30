"""Regression tests for navigation, climbing and environment fixes from the code review."""

import numpy as np

from rebuilt_sim import constants as C
from rebuilt_sim.constants import Alliance
from rebuilt_sim.controller import Macro, MacroController
from rebuilt_sim.env import EnvConfig
from rebuilt_sim.field import BUMP_PORTAL_Y, FIELD
from rebuilt_sim.obs import observe
from rebuilt_sim.robot import TIERS, RobotCommand
from rebuilt_sim.runner import PolicyMatch
from rebuilt_sim.sim import GROUND, Match

# out-of-the-way parking spots for robots that shouldn't interfere
_FAR = [(15.9, 0.6), (15.9, 1.7), (15.9, 2.8), (15.9, 5.3), (15.9, 6.4)]


def _solo(tier: str, start, seed: int = 0, others=None) -> Match:
    specs = [TIERS[tier]] + [TIERS["broken"]] * 5
    return Match(specs, seed=seed, starts=[start] + (others or _FAR))


def _drive(m: Match, macro: Macro, seconds: float, index: int = 0) -> None:
    ctl = MacroController(m)
    end = m.t + seconds
    while m.t < end:
        cmds = [None] * 6
        cmds[index] = ctl.command(m.robots[index], macro)
        m.step(cmds)


def test_robot_in_the_old_gap_crosses_the_hub_line():
    # (3.86, lane) used to be between the "lined up" window and the HUB line: it oscillated forever
    m = _solo("low", (3.86, BUMP_PORTAL_Y[0]))
    _drive(m, Macro.COLLECT, 6.0)
    assert m.robots[0].x > FIELD.neutral_zone.x0 or m.robots[0].stats.fuel_collected > 0


def test_parked_robot_on_a_bump_lane_is_routed_around():
    blocker = (C.ALLIANCE_ZONE_DEPTH - 0.5, BUMP_PORTAL_Y[0])
    specs = [TIERS["mid"], TIERS["broken"]] + [TIERS["broken"]] * 4
    m = Match(specs, seed=1, starts=[(2.0, BUMP_PORTAL_Y[0]), blocker] + _FAR[:4])
    ctl = MacroController(m)
    goal = (6.5, BUMP_PORTAL_Y[0])  # straight ahead, past the parked robot
    while m.t < 8.0:
        m.step([ctl.goto(m.robots[0], *goal)] + [None] * 5)
    assert m.robots[0].x > FIELD.neutral_zone.x0  # went through the other BUMP (a mid robot can't TRENCH)


def test_depot_and_outpost_are_reachable_around_the_tower():
    out_x, out_y = FIELD.outpost_spot(Alliance.BLUE, TIERS["mid"].radius)
    m = _solo("mid", (out_x, out_y))
    _drive(m, Macro.COLLECT_DEPOT, 9.0)
    depot = FIELD.depots[Alliance.BLUE]
    assert m.robots[0].y > depot.y0 - 0.5 and m.robots[0].x < 1.4
    _drive(m, Macro.COLLECT_OUTPOST, 9.0)
    assert m.robots[0].y < out_y + 0.6 and m.robots[0].x < 1.4


def test_three_robots_can_climb_and_earn_traversal():
    specs = [TIERS["elite_climber"]] * 3 + [TIERS["broken"]] * 3
    m = Match(specs, seed=2, starts=[(2.0, 1.6), (2.0, 4.0), (2.0, 6.4)] + _FAR[:3])
    idle = [RobotCommand()] * 3 + [None] * 3
    while m.t < C.ENDGAME_START:
        m.step(idle)
    ctl = MacroController(m)
    while not m.done:
        m.step([ctl.command(m.robots[i], Macro.CLIMB) for i in range(3)] + [None] * 3)
    climbed = sum(r.climb_level > 0 for r in m.robots[:3])
    assert climbed >= 2
    assert m.scores[Alliance.BLUE].tower >= 50
    assert m.rp[Alliance.BLUE].traversal == 1


def test_fixed_tiers_apply_to_every_seat():
    pm = PolicyMatch(EnvConfig(), {}, tiers={0: "elite", 1: "elite", 3: "broken"}, seed=5)
    names = [r.spec.name for r in pm.match.robots]
    assert names[0] == "elite" and names[1] == "elite" and names[3] == "broken"


def test_observation_stays_in_bounds_with_huge_scores():
    pm = PolicyMatch(EnvConfig(), {}, seed=6)
    m = pm.match
    m.scores[Alliance.BLUE].teleop_fuel = 900
    m.scores[Alliance.RED].penalty_points = 600
    for i in range(6):
        o = observe(m, pm.runner.ctl, i)
        assert np.abs(o).max() <= 2.0


def test_coincident_fuel_separates():
    m = _solo("broken", (1.0, 1.0))
    g = np.flatnonzero(m.f_state == GROUND)
    a, b = g[0], g[1]
    m.f_pos[a] = m.f_pos[b] = (8.0, 1.0)
    m.f_settle[[a, b]] = 3
    for _ in range(10):
        m.step([None] * 6)
    assert np.hypot(*(m.f_pos[a] - m.f_pos[b])) > C.FUEL_RADIUS


def test_depot_fuel_pushed_against_the_wall_is_collected():
    # from its usual spot at the DEPOT a robot's intake doesn't reach the wall; it used to wait
    # there for FUEL pushed back against the wall until the match ended
    box = FIELD.depots[Alliance.BLUE]
    m = _solo("elite", (2.0, box.center[1]))
    g = np.flatnonzero(m.f_state == GROUND)
    depot = g[(m.f_pos[g, 0] < box.x1) & (m.f_pos[g, 1] > box.y0) & (m.f_pos[g, 1] < box.y1)]
    m.f_pos[depot[:6]] = [(C.FUEL_RADIUS + 0.01, box.y0 + 0.15 + 0.155 * k) for k in range(6)]
    m.f_pos[depot[6:]] = [(C.CENTER_X, 1.0 + 0.16 * k) for k in range(len(depot) - 6)]
    m.f_vel[depot] = 0.0
    m.f_settle[depot] = 3
    assert m.depot_count(Alliance.BLUE) == 6
    _drive(m, Macro.COLLECT_DEPOT, 10.0)
    assert m.depot_count(Alliance.BLUE) == 0


def test_fuel_inside_the_tower_does_not_trap_a_collector():
    # FUEL rolls in under the RUNGS, but the UPRIGHTS are closer together than a robot is wide:
    # a collector used to circle in front of the TOWER for the rest of the match
    cy = C.TOWER_CENTER_Y_BLUE
    m = _solo("elite", (2.2, cy))
    g = np.flatnonzero(m.f_state == GROUND)[:12]
    m.f_pos[g] = [(0.08 + 0.15 * (k % 3), cy - 0.4 + 0.15 * (k // 3)) for k in range(12)]
    m.f_vel[g] = 0.0
    m.f_settle[g] = 3
    _drive(m, Macro.COLLECT, 10.0)
    assert m.robots[0].stats.fuel_collected >= 5
