"""Compare scripted-bot matches with the real 2026 Istanbul Regional Day 1 results.

Runs matches in parallel with random robot tiers drawn from the Day 1 skill mix and
prints per-tier robot output and the alliance FUEL distribution next to the real data
in data/istanbul2026_day1_match_results.csv.

    python scripts/calibrate.py --matches 120
    python scripts/calibrate.py --matches 40 --blue elite strong strong --red mid low low
    python scripts/calibrate.py --matches 96 --hifi              # the high-fidelity physics
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # before numpy loads: one BLAS thread per worker

import numpy as np  # noqa: E402

from rebuilt_sim.bots import ScriptedPolicy  # noqa: E402
from rebuilt_sim.robot import DAY1_TIER_WEIGHTS  # noqa: E402
from rebuilt_sim.sim import HiFiConfig, MatchConfig, make_match  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def draw_alliance(rng: np.random.Generator) -> list[str]:
    names = list(DAY1_TIER_WEIGHTS)
    p = np.array([DAY1_TIER_WEIGHTS[n] for n in names])
    return list(rng.choice(names, size=3, p=p / p.sum()))


def play(args: tuple[int, list[str] | None, list[str] | None, bool]) -> dict:
    seed, blue, red, hifi = args
    rng = np.random.default_rng(seed)
    blue = blue or draw_alliance(rng)
    red = red or draw_alliance(rng)
    m = make_match(blue, red, seed=seed, config=MatchConfig(hifi=HiFiConfig()) if hifi else None)
    s = m.run(ScriptedPolicy(m, seed=seed))
    return {
        "alliances": [
            {"fuel": sc.fuel, "auto": sc.auto_fuel, "total": sc.total, "tower": sc.tower,
             "fouls": sc.minor_fouls + sc.major_fouls, "wasted": sc.wasted_fuel}
            for sc in s.scores
        ],
        "robots": [(r.spec.name, r.stats.fuel_scored, r.stats.tower_points, r.stats.fuel_shot,
                    r.stats.fuel_scored + r.stats.fuel_wasted) for r in m.robots],
        "margin": abs(s.scores[0].total - s.scores[1].total),
    }


def real_alliance_fuel() -> list[int]:
    path = ROOT / "data" / "istanbul2026_day1_match_results.csv"
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for c in ("red", "blue"):
                out.append(int(row[f"{c}_auto_fuel"]) + int(row[f"{c}_teleop_fuel"]))
    return out


def describe(xs: list[int]) -> str:
    xs = sorted(xs)
    return (f"median {statistics.median(xs):6.1f}  mean {statistics.mean(xs):6.1f}  max {xs[-1]:4d}  "
            f">=100: {sum(x >= 100 for x in xs) / len(xs):5.1%}  >=360: {sum(x >= 360 for x in xs) / len(xs):5.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blue", nargs=3, help="fixed blue tiers, e.g. elite strong mid")
    ap.add_argument("--red", nargs=3, help="fixed red tiers")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--hifi", action="store_true", help="high-fidelity physics (docs/03-driving-and-aiming.md)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    jobs = [(args.seed + i, args.blue, args.red, args.hifi) for i in range(args.matches)]
    with Pool(args.workers) as pool:
        results = pool.map(play, jobs)
    wall = time.perf_counter() - t0

    per_tier = defaultdict(list)
    shots = defaultdict(lambda: [0, 0])  # tier -> [FUEL shot, FUEL that went into a HUB]
    for res in results:
        for name, scored, _, shot, hit in res["robots"]:
            per_tier[name].append(scored)
            shots[name][0] += shot
            shots[name][1] += hit
    fuel = [a["fuel"] for res in results for a in res["alliances"]]
    autos = [a["auto"] for res in results for a in res["alliances"]]
    fouls = [a["fouls"] for res in results for a in res["alliances"]]
    wasted = [a["wasted"] for res in results for a in res["alliances"]]
    towers = [a["tower"] for res in results for a in res["alliances"]]

    print(f"{len(results)} {'high-fidelity ' if args.hifi else ''}matches in {wall:.1f} s on {args.workers} workers\n")
    print("FUEL scored per robot, by tier (Day 1 targets: elite ~225-250, strong ~40-70, mid ~15-30, low <15)")
    for name, xs in sorted(per_tier.items(), key=lambda kv: -statistics.mean(kv[1])):
        shot, hit = shots[name]
        print(f"  {name:14s} n={len(xs):4d}  mean {statistics.mean(xs):6.1f}  sd {statistics.pstdev(xs):6.1f}"
              f"  shots in {hit / max(1, shot):4.0%}")
    print("\nAlliance FUEL per match")
    print("  simulated :", describe(fuel))
    if not (args.blue or args.red):
        print("  real Day 1:", describe(real_alliance_fuel()))
    print(f"\n  AUTO share of FUEL {sum(autos) / max(1, sum(fuel)):.1%} (real 19.8%)   "
          f"fouls/alliance {statistics.mean(fouls):.2f} (real 0.26 awards)   "
          f"wasted FUEL/alliance {statistics.mean(wasted):.1f}   tower pts/alliance {statistics.mean(towers):.2f} (real 0.48)")
    print(f"  winning margin median {statistics.median([r['margin'] for r in results]):.0f} (real 42)")


if __name__ == "__main__":
    main()
