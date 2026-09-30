"""Generate a labeled synthetic image dataset from simulated matches (docs/04-vision.md).

    python scripts/make_dataset.py --name smoke --matches 2 --frames 4            # a quick look
    python scripts/make_dataset.py --name v1 --matches 200 --workers 12 --dry-run # how long, how big?
    python scripts/make_dataset.py --name v1 --matches 200 --workers 12           # a real dataset
    python scripts/make_dataset.py --presets limelight4 --scale 1.0               # native 1280 x 800
    python scripts/make_dataset.py --no-randomize --effects 0 --png               # clean, lossless renders
    python scripts/make_dataset.py --randomization ranges.json                    # your own ranges
    python scripts/make_dataset.py --out D:/datasets ...                          # a drive with more space

The dataset goes to <out>/<name>/ (datasets/ is gitignored): images, per-match label records,
index.csv, manifest.json and a dataset card. Run it again with the same name to resume an
interrupted run. Then make training files with scripts/export_dataset.py and look at the data
with scripts/check_dataset.py. Rendering is CPU-only: about 1 s per 640 x 400 image per worker.
"""

from __future__ import annotations

import argparse
import json

from rebuilt_sim.vision.dataset import DatasetConfig, dry_run, generate
from rebuilt_sim.vision.randomize import PRESETS, RandomizationConfig


def _splits(items: list[str]) -> tuple[tuple[str, float], ...]:
    out = []
    for item in items:
        name, _, frac = item.partition("=")
        out.append((name, float(frac)))
    if abs(sum(f for _, f in out) - 1.0) > 1e-6:
        raise SystemExit(f"--splits fractions must add up to 1: {items}")
    return tuple(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = DatasetConfig()
    ap.add_argument("--name", default=d.name)
    ap.add_argument("--out", default=d.out_dir, help="parent folder (default: datasets)")
    ap.add_argument("--matches", type=int, default=d.matches)
    ap.add_argument("--frames", type=int, default=d.frames_per_match, help="moments sampled per match")
    ap.add_argument("--cameras", type=int, default=d.cameras_per_frame, help="cameras rendered per moment")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--presets", nargs="+", default=list(d.presets), choices=sorted(PRESETS))
    ap.add_argument("--scale", type=float, default=d.scale, help="fraction of each camera's native resolution")
    ap.add_argument("--supersample", type=int, default=d.supersample)
    ap.add_argument("--strategy-physics", action="store_true", help="faster matches, but FUEL flight heights are made up")
    ap.add_argument("--no-randomize", action="store_true",
                    help="default colors, lighting and lenses (camera effects are set by --effects)")
    ap.add_argument("--effects", type=float, default=d.effects_strength, help="camera effects strength (0 = clean)")
    ap.add_argument("--randomization", help="JSON file overriding RandomizationConfig ranges (vision/randomize.py)")
    ap.add_argument("--masks", action="store_true", help="also save per-pixel object ids")
    ap.add_argument("--depth", action="store_true", help="also save depth images")
    ap.add_argument("--png", action="store_true",
                    help="lossless PNG images (about 4x the disk space of JPEG: 2x for mono cameras, 7x for color)")
    ap.add_argument("--splits", nargs="+", default=[f"{n}={f:g}" for n, f in d.splits],
                    help="split fractions by match, e.g. train=0.8 val=0.1 test=0.1 (fixed when generating)")
    ap.add_argument("--workers", type=int, default=d.workers, help="parallel processes (one match each)")
    ap.add_argument("--dry-run", action="store_true", help="only estimate time, disk and memory")
    ap.add_argument("--overwrite", action="store_true", help="regenerate matches already on disk")
    args = ap.parse_args()

    rc = RandomizationConfig()
    if args.randomization:
        with open(args.randomization, encoding="utf-8") as f:
            rc = RandomizationConfig.from_dict(json.load(f))
    cfg = DatasetConfig(
        name=args.name, out_dir=args.out, matches=args.matches, frames_per_match=args.frames,
        cameras_per_frame=args.cameras, seed=args.seed, presets=tuple(args.presets), scale=args.scale,
        supersample=args.supersample, hifi=not args.strategy_physics, randomize=not args.no_randomize,
        effects_strength=args.effects, masks=args.masks, depth=args.depth, lossless=args.png,
        workers=args.workers, randomization=rc, splits=_splits(args.splits),
    )
    est = dry_run(cfg)
    print(f"about {est['images']} images, {est['hours']} h with {cfg.workers} worker(s), "
          f"{est['disk_gb']} GB of disk, {est['ram_gb']} GB of RAM")
    if args.dry_run:
        return
    st = generate(cfg, overwrite=args.overwrite)
    print(f"done: {sum(st['images'].values())} images, {st['disk_mb']} MB; see {cfg.out_dir}/{cfg.name}/dataset_card.md")


if __name__ == "__main__":
    main()
