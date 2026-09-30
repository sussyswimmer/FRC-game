"""Watch a match: scripted bots, or a trained model driving one or all robots.

    python scripts/watch.py                                              # bots only, random Day 1 robots
    python scripts/watch.py --blue elite strong mid --red strong low low
    python scripts/watch.py --model runs/strategy/final_model.zip --seat blue_1 --tier strong
    python scripts/watch.py --model runs/selfplay/latest.pt --all        # the model drives all six robots
    python scripts/watch.py --hifi                                       # high-fidelity physics

Keys: SPACE pause, N next match, UP/DOWN speed, ESC quit.
"""

from __future__ import annotations

import argparse

import numpy as np

from rebuilt_sim.robot import DAY1_TIER_WEIGHTS, TIERS
from rebuilt_sim.runner import AGENTS, PolicyMatch, run_config_for
from rebuilt_sim.env import EnvConfig
from rebuilt_sim.sim import HiFiConfig
from rebuilt_sim.viewer import Viewer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help=".zip (train_ppo.py) or .pt (train_selfplay.py)")
    ap.add_argument("--seat", default="blue_1", choices=AGENTS, help="robot the model drives")
    ap.add_argument("--all", action="store_true", help="the model drives all six robots")
    ap.add_argument("--tier", default=None, choices=sorted(TIERS), help="robot type in the model's seat")
    ap.add_argument("--blue", nargs=3, choices=sorted(TIERS))
    ap.add_argument("--red", nargs=3, choices=sorted(TIERS))
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--hifi", action="store_true",
                    help="high-fidelity physics for the bot-only view (a model uses what it was trained on)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    policies, cfg = {}, EnvConfig(hifi=HiFiConfig() if args.hifi else None)
    if args.model:
        from rebuilt_sim.policies import load_policy

        fn, mode = load_policy(args.model)
        cfg = run_config_for(args.model, mode)
        seats = range(6) if args.all else [AGENTS.index(args.seat)]
        policies = {i: fn for i in seats}

    viewer = Viewer(title="REBUILT - watch")
    pg = viewer.pg
    seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(1 << 30))

    def new_match(seed):
        tiers = {}
        rng = np.random.default_rng(seed)
        names = list(DAY1_TIER_WEIGHTS)
        p = np.array([DAY1_TIER_WEIGHTS[n] for n in names])
        for i in range(6):
            given = (args.blue or [None] * 3)[i] if i < 3 else (args.red or [None] * 3)[i - 3]
            tiers[i] = given or str(rng.choice(names, p=p / p.sum()))
        if args.tier and policies:
            for i in policies:
                tiers[i] = args.tier
        return PolicyMatch(cfg, policies, tiers=tiers, seed=seed)

    pm = new_match(seed)
    speed, paused = args.speed, False
    highlight = next(iter(policies), None) if not args.all else None
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                viewer.close()
                return
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    paused = not paused
                elif event.key == pg.K_n:
                    seed += 1
                    pm = new_match(seed)
                elif event.key == pg.K_UP:
                    speed = min(16.0, speed * 2)
                elif event.key == pg.K_DOWN:
                    speed = max(0.25, speed / 2)
        if not paused and not pm.match.done:
            pm.step()
        m = pm.match
        tiers = " ".join(f"{r.name}:{r.spec.name}" for r in m.robots)
        msg = f"seed {seed}  x{speed:g}  {tiers}" + ("  - N for next match" if m.done else "")
        viewer.draw(m, highlight=highlight, message=msg)
        viewer.tick(speed / m.cfg.dt)


if __name__ == "__main__":
    main()
