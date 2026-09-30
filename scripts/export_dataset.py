"""Make training files (COCO or YOLO) from a generated dataset. Every choice here is a training
decision, so nothing is picked for you beyond neutral defaults; each export writes the settings it
used to export.json. Exports are quick: change your mind and export again.

    python scripts/export_dataset.py datasets/v1 --name fuel_only --format yolo --classes fuel
    python scripts/export_dataset.py datasets/v1 --name alliances --format yolo --classes fuel robot_blue robot_red
    python scripts/export_dataset.py datasets/v1 --name coco_all --format coco --classes fuel robot apriltag hub
    python scripts/export_dataset.py datasets/v1 --name tags --format yolo-pose --classes apriltag
    python scripts/export_dataset.py datasets/v1 --name big_only --format yolo --classes fuel \\
        --min-box-side 6 --min-visibility 0.3 --box amodal

Output: datasets/v1/exports/<name>/.
"""

from __future__ import annotations

import argparse

from rebuilt_sim.vision.export import CLASS_NAMES, ExportConfig, export


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = ExportConfig(name="x")
    ap.add_argument("dataset", help="the dataset folder, e.g. datasets/v1")
    ap.add_argument("--name", required=True, help="name of this export")
    ap.add_argument("--format", default=d.format, choices=("coco", "yolo", "yolo-pose"))
    ap.add_argument("--classes", nargs="+", default=list(d.classes), choices=CLASS_NAMES)
    ap.add_argument("--box", default=d.box, choices=("visible", "amodal"),
                    help="visible pixels only, or the whole object including hidden parts")
    ap.add_argument("--min-visible-px", type=float, default=d.min_visible_px)
    ap.add_argument("--min-box-side", type=float, default=d.min_box_side_px, help="pixels")
    ap.add_argument("--min-visibility", type=float, default=d.min_visibility, help="0-1: how much may be hidden")
    ap.add_argument("--max-truncation", type=float, default=d.max_truncation, help="0-1: how much may be off-image")
    ap.add_argument("--fuel-states", nargs="+", default=list(d.fuel_states), choices=("ground", "flight", "chute"))
    ap.add_argument("--splits", nargs="+", default=list(d.splits))
    args = ap.parse_args()
    ec = ExportConfig(name=args.name, format=args.format, classes=tuple(args.classes), box=args.box,
                      min_visible_px=args.min_visible_px, min_box_side_px=args.min_box_side,
                      min_visibility=args.min_visibility, max_truncation=args.max_truncation,
                      fuel_states=tuple(args.fuel_states), splits=tuple(args.splits))
    counts = export(args.dataset, ec)
    print(counts)


if __name__ == "__main__":
    main()
