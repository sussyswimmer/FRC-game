"""Drive a robot yourself against scripted bots.

    python scripts/play.py                                   # you drive blue_1, a "strong" robot
    python scripts/play.py --robot red_0 --tier elite
    python scripts/play.py --teammates mid low --opponents elite strong mid
    python scripts/play.py --drive-auto                      # also drive during AUTO
    python scripts/play.py --hifi                            # swerve modules, latency, real ballistics

Keyboard: WASD / arrow keys move (screen directions), Q/E rotate, hold SPACE to intake,
hold F or J to shoot, C to climb, P pause, R restart, ESC quit.
Gamepad: left stick move, right stick rotate, left trigger intake, right trigger shoot, A climb.
Robots without a turret auto-aim at their HUB while you hold shoot (turn off with --no-assist).
During AUTO your robot runs its scripted autonomous routine unless you pass --drive-auto.
With --hifi (docs/03-driving-and-aiming.md) the robot's aim software sets the shooter and fires
only once it is on target; the thin outline shows where your robot believes it is.
"""

from __future__ import annotations

import argparse
import math

from rebuilt_sim.bots import ScriptedPolicy
from rebuilt_sim.robot import TIERS, RobotCommand
from rebuilt_sim.rules import Period
from rebuilt_sim.sim import HiFiConfig, Match, MatchConfig
from rebuilt_sim.viewer import Viewer

AGENTS = ("blue_0", "blue_1", "blue_2", "red_0", "red_1", "red_2")


def read_input(pg, joystick, robot, assist: bool, hub) -> RobotCommand:
    keys = pg.key.get_pressed()
    x = (keys[pg.K_d] or keys[pg.K_RIGHT]) - (keys[pg.K_a] or keys[pg.K_LEFT])
    y = (keys[pg.K_w] or keys[pg.K_UP]) - (keys[pg.K_s] or keys[pg.K_DOWN])
    turn = keys[pg.K_q] - keys[pg.K_e]
    intake = keys[pg.K_SPACE]
    shoot = keys[pg.K_f] or keys[pg.K_j]
    climb = keys[pg.K_c]
    if joystick is not None:
        dead = lambda v: 0.0 if abs(v) < 0.12 else v  # noqa: E731
        x += dead(joystick.get_axis(0))
        y -= dead(joystick.get_axis(1))
        turn -= dead(joystick.get_axis(2))
        if joystick.get_numaxes() >= 6:
            intake = intake or joystick.get_axis(4) > 0.2
            shoot = shoot or joystick.get_axis(5) > 0.2
        climb = climb or joystick.get_button(0)
    s = robot.spec
    mag = math.hypot(x, y)
    if mag > 1:
        x, y = x / mag, y / mag
    omega = turn * s.max_omega * 0.7
    if shoot and assist and not s.turret:  # turn to face the HUB so the shot can fire
        err = (math.atan2(hub[1] - robot.y, hub[0] - robot.x) - robot.heading + math.pi) % (2 * math.pi) - math.pi
        omega = max(-s.max_omega, min(s.max_omega, 6 * err))
    return RobotCommand(vx=x * s.max_speed, vy=y * s.max_speed, omega=omega, intake=bool(intake),
                        shoot=bool(shoot), climb=s.climb_level if climb else 0)


def new_match(args) -> tuple[Match, int, ScriptedPolicy]:
    me = AGENTS.index(args.robot)
    mine = [0, 1, 2] if me < 3 else [3, 4, 5]
    tiers: list[str | None] = [None] * 6
    tiers[me] = args.tier
    for i, t in zip([i for i in mine if i != me], args.teammates):
        tiers[i] = t
    for i, t in zip([i for i in range(6) if i not in mine], args.opponents):
        tiers[i] = t
    cfg = MatchConfig(hifi=HiFiConfig()) if args.hifi else MatchConfig()
    m = Match([TIERS[t] for t in tiers], cfg, seed=args.seed)
    return m, me, ScriptedPolicy(m, seed=args.seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default="blue_1", choices=AGENTS)
    ap.add_argument("--tier", default="strong", choices=sorted(TIERS))
    ap.add_argument("--teammates", nargs=2, default=["mid", "low"], choices=sorted(TIERS))
    ap.add_argument("--opponents", nargs=3, default=["strong", "mid", "low"], choices=sorted(TIERS))
    ap.add_argument("--drive-auto", action="store_true", help="drive your robot in AUTO too")
    ap.add_argument("--no-assist", action="store_true", help="no auto-aim while shooting")
    ap.add_argument("--speed", type=float, default=1.0, help="simulation speed multiplier")
    ap.add_argument("--hifi", action="store_true", help="high-fidelity physics (docs/03-driving-and-aiming.md)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    viewer = Viewer()
    pg = viewer.pg
    pg.joystick.init()
    joystick = pg.joystick.Joystick(0) if pg.joystick.get_count() else None
    m, me, bots = new_match(args)
    paused = False
    fps = 1.0 / m.cfg.dt * args.speed
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                viewer.close()
                return
            if event.type == pg.KEYDOWN and event.key == pg.K_p:
                paused = not paused
            if event.type == pg.KEYDOWN and event.key == pg.K_r:
                m, me, bots = new_match(args)
        if not paused and not m.done:
            cmds = bots.commands()
            if args.drive_auto or m.period != Period.AUTO:
                r = m.robots[me]
                cmds[me] = read_input(pg, joystick, r, not args.no_assist, m.field.hub_centers[r.alliance])
            m.step(cmds)
        r = m.robots[me]
        status = f"you: {r.name} ({r.spec.name})  FUEL held {r.fuel}  scored {r.stats.fuel_scored}"
        if m.done:
            status = "match over - R to play again, ESC to quit"
        elif paused:
            status = "paused - P to resume"
        elif m.period == Period.AUTO and not args.drive_auto:
            status = "AUTO: your robot is running its autonomous routine"
        viewer.draw(m, highlight=me, message=status)
        viewer.tick(fps)


if __name__ == "__main__":
    main()
