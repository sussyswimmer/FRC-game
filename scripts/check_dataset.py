"""Check a generated dataset and look at it.

    python scripts/check_dataset.py datasets/v1                  # files and labels consistent?
    python scripts/check_dataset.py datasets/v1 --decode-tags 50 # the real AprilTag detector vs the labels
    python scripts/check_dataset.py datasets/v1 --sheet sheet.png --count 16   # images with labels drawn on
    python scripts/check_dataset.py datasets/v1 --stats

Looking at a sample of the data with its labels is the first thing to do with any dataset.
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--decode-tags", type=int, default=0, help="run the AprilTag detector on up to N clear tags")
    ap.add_argument("--sheet", help="write a contact sheet PNG here")
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--split", default=None)
    ap.add_argument("--amodal", action="store_true", help="draw whole-object boxes instead of visible ones")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    from rebuilt_sim.vision.check import contact_sheet, validate

    if args.stats:
        from rebuilt_sim.vision.card import stats

        print(json.dumps(stats(args.dataset), indent=2))
    report = validate(args.dataset, decode_tags=args.decode_tags)
    for p in report["problems"][:20]:
        print("PROBLEM:", p)
    if args.sheet:
        print("wrote", contact_sheet(args.dataset, args.sheet, args.count, split=args.split, seed=args.seed,
                                      amodal=args.amodal))


if __name__ == "__main__":
    main()
