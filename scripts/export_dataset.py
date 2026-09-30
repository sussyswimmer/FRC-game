"""Make training files (COCO or YOLO) from a generated dataset. Every choice here is a training
decision: the format, classes, box style and what to do with the camera's own robot must be given,
and the filters default to keeping everything. Each export writes the settings it used to
export.json. Exports are quick: change your mind and export again.

    python scripts/export_dataset.py datasets/v1 --name fuel_only --format yolo --classes fuel \\
        --box visible --camera-robot keep
    python scripts/export_dataset.py datasets/v1 --name alliances --format yolo \\
        --classes fuel robot_blue robot_red --box visible --camera-robot drop
    python scripts/export_dataset.py datasets/v1 --name coco_all --format coco \\
        --classes fuel robot apriltag hub --box amodal --camera-robot keep
    python scripts/export_dataset.py datasets/v1 --name tags --format yolo-pose --classes apriltag \\
        --box visible --camera-robot drop --min-box-side 6 --min-visibility 0.3

Output: datasets/v1/exports/<name>/.
"""

from __future__ import annotations

import argparse

from rebuilt_sim.vision.export import CLASS_NAMES, ExportConfig, export


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = ExportConfig("x", "coco", ("fuel",), "visible", True)  # only for the filters' defaults
    ap.add_argument("dataset", help="the dataset folder, e.g. datasets/v1")
    ap.add_argument("--name", required=True, help="name of this export")
    ap.add_argument("--format", required=True, choices=("coco", "yolo", "yolo-pose"))
    ap.add_argument("--classes", nargs="+", required=True, choices=CLASS_NAMES)
    ap.add_argument("--box", required=True, choices=("visible", "amodal"),
                    help="visible pixels only, or the whole object including hidden parts")
    ap.add_argument("--camera-robot", required=True, choices=("keep", "drop"),
                    help="the robot each camera is mounted on (its own bumper or intake in view): a 'robot' or not")
    ap.add_argument("--min-visible-px", type=float, default=d.min_visible_px)
    ap.add_argument("--min-box-side", type=float, default=d.min_box_side_px, help="pixels")
    ap.add_argument("--min-visibility", type=float, default=d.min_visibility, help="0-1: how much may be hidden")
    ap.add_argument("--max-truncation", type=float, default=d.max_truncation, help="0-1: how much may be off-image")
    ap.add_argument("--fuel-states", nargs="+", default=list(d.fuel_states), choices=("ground", "flight", "chute"))
    ap.add_argument("--splits", nargs="+", default=list(d.splits))
    args = ap.parse_args()
    ec = ExportConfig(name=args.name, format=args.format, classes=tuple(args.classes), box=args.box,
                      include_camera_robot=args.camera_robot == "keep", min_visible_px=args.min_visible_px, min_box_side_px=args.min_box_side,
                      min_visibility=args.min_visibility, max_truncation=args.max_truncation,
                      fuel_states=tuple(args.fuel_states), splits=tuple(args.splits))
    counts = export(args.dataset, ec)
    print(counts)


if __name__ == "__main__":
    main()
