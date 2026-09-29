"""Measure a trained model against the scripted driver on identical matches.

    python scripts/evaluate.py runs/strategy/final_model.zip --tier strong --matches 100
    python scripts/evaluate.py runs/selfplay/latest.pt --tier elite --matches 100
    python scripts/evaluate.py --baseline-only --tier strong --matches 100

For each seed the model drives one seat; then the same match (same robots, same seed) is
replayed with the scripted driver for that robot type in the seat. The difference is what
your training bought.
"""

from __future__ import annotations

import argparse
import os
import statistics
from multiprocessing import Pool

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # before numpy loads: one BLAS thread per worker

import numpy as np  # noqa: E402

from rebuilt_sim.env import EnvConfig  # noqa: E402
from rebuilt_sim.robot import TIERS  # noqa: E402
from rebuilt_sim.runner import AGENTS, PolicyMatch, run_config_for  # noqa: E402

_POLICY = None


def _init(model: str | None) -> None:
    global _POLICY
    if model:
        from rebuilt_sim.policies import load_policy

        _POLICY = load_policy(model)


def _play(job) -> dict:
    seed, seat, tier, use_model, cfg_args = job
    cfg = EnvConfig(**cfg_args)
    tiers = {seat: tier}
    if use_model:
        fn, _ = _POLICY
        pm = PolicyMatch(cfg, {seat: fn}, tiers=tiers, seed=seed)
    else:  # the seat is driven by the regular scripted bot for its robot type
        pm = PolicyMatch(cfg, {}, tiers=tiers, seed=seed)
    pm.run()
    m = pm.match
    r = m.robots[seat]
    own, opp = m.scores[r.alliance], m.scores[r.alliance.other]
    rp = m.summary().rp[r.alliance]
    return {"win": float(own.total > opp.total), "own": own.total, "opp": opp.total, "margin": own.total - opp.total,
            "rp": rp.total, "robot_fuel": r.stats.fuel_scored, "wasted": r.stats.fuel_wasted,
            "fouls": r.stats.fouls_minor + r.stats.fouls_major, "tower": r.stats.tower_points}


def summarize(label: str, rows: list[dict]) -> None:
    def ms(k):
        xs = [row[k] for row in rows]
        return f"{statistics.mean(xs):7.2f} ±{statistics.pstdev(xs) / max(1, len(xs)) ** 0.5:5.2f}"
    print(f"{label:10s} win {ms('win')}  margin {ms('margin')}  own {ms('own')}  RP {ms('rp')}  "
          f"robot FUEL {ms('robot_fuel')}  wasted {ms('wasted')}  fouls {ms('fouls')}  tower {ms('tower')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help=".zip or .pt model")
    ap.add_argument("--tier", default="strong", choices=sorted(TIERS))
    ap.add_argument("--seat", default=None, choices=AGENTS, help="default: alternate blue_1 / red_1")
    ap.add_argument("--matches", type=int, default=60)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    if not args.model and not args.baseline_only:
        ap.error("give a model path or --baseline-only")

    if args.model:
        from rebuilt_sim.policies import load_policy

        _, mode = load_policy(args.model)
        cfg = run_config_for(args.model, mode)
    else:
        cfg = EnvConfig()
    cfg_args = {"action_mode": cfg.action_mode, "decision_dt": cfg.decision_dt}
    seats = [AGENTS.index(args.seat)] if args.seat else [1, 4]
    jobs = [(args.seed + k, seats[k % len(seats)], args.tier) for k in range(args.matches)]

    base_cfg = {"action_mode": "macro", "decision_dt": 0.25}
    with Pool(args.workers) as pool:
        baseline = pool.map(_play, [(s, seat, tier, False, base_cfg) for s, seat, tier in jobs])
    summarize("scripted", baseline)
    if args.model:
        with Pool(args.workers, initializer=_init, initargs=(args.model,)) as pool:
            model_rows = pool.map(_play, [(s, seat, tier, True, cfg_args) for s, seat, tier in jobs])
        summarize("model", model_rows)
        diff = [a["margin"] - b["margin"] for a, b in zip(model_rows, baseline)]
        print(f"\nmodel vs scripted driver, same {len(diff)} matches: margin change "
              f"{np.mean(diff):+.1f} (±{np.std(diff) / len(diff) ** 0.5:.1f}), "
              f"win rate {np.mean([r['win'] for r in model_rows]):.0%} vs {np.mean([r['win'] for r in baseline]):.0%}")


if __name__ == "__main__":
    main()
