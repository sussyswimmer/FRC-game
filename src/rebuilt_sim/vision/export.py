"""Turn a generated dataset into the formats training tools read: COCO and YOLO.

Generation keeps everything; exporting is where the training choices are made: which classes to
detect, visible or whole ("amodal") boxes, whether the camera's own robot counts, how small or
hidden an object may be and still count, which FUEL states to include. Those are yours to decide
(CLAUDE.md, Rule #1): the first ones have no default at all, and the filters default to keeping
everything. Each export records its settings in ``export.json``. Re-export as often as you like:
no re-rendering.

Layouts (under ``datasets/<name>/exports/<export name>/``):

- **COCO** (``--format coco``): ``annotations/<split>.json``. Boxes ``[x, y, w, h]`` in pixel edge
  coordinates. AprilTags also get their 4 corners as keypoints, in pixel-center coordinates (as in
  the records; detectron2 and mmpose read COCO keypoints that way).
- **YOLO** (``--format yolo``): ``images/<split>/`` (hard links to the dataset's images, or copies
  where links aren't possible), ``labels/<split>/*.txt`` with one ``class cx cy w h`` line per object
  (normalized to 0-1), and ``data.yaml``: a descriptor of paths and class names, no training settings.
- **YOLO pose** (``--format yolo-pose``): as YOLO, plus the 4 tag corners as keypoints (normalized
  like the boxes, so in edge coordinates).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .dataset import load_records
from .labels import CATEGORIES

# classes an export can ask for: a category, or a category split by alliance
CLASS_NAMES = tuple(CATEGORIES) + ("robot_blue", "robot_red", "hub_blue", "hub_red")


@dataclass
class ExportConfig:
    # the choices only you can make: no defaults
    name: str
    format: str  # "coco", "yolo" or "yolo-pose"
    classes: tuple[str, ...]  # from CLASS_NAMES
    box: str  # "visible" (only what's seen) or "amodal" (the whole object, hidden parts included)
    include_camera_robot: bool  # the robot carrying a camera often shows (its bumper or intake) in its images
    # filters: the defaults keep everything
    min_visible_px: float = 1.0  # objects with fewer visible pixels are left out (0 keeps fully hidden ones)
    min_box_side_px: float = 0.0
    min_visibility: float = 0.0  # fraction of the object that is not hidden
    max_truncation: float = 1.0  # fraction of the object outside the image
    fuel_states: tuple[str, ...] = ("ground", "flight", "chute")
    splits: tuple[str, ...] = ("train", "val", "test")

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name) or ".." in self.name:
            raise ValueError(f"export name {self.name!r}: use letters, digits, '_', '-' and '.' only "
                             "(it becomes a folder under <dataset>/exports/)")
        bad = [c for c in self.classes if c not in CLASS_NAMES]
        if bad:
            raise ValueError(f"unknown classes {bad}; choose from {CLASS_NAMES}")
        if self.format not in ("coco", "yolo", "yolo-pose"):
            raise ValueError("format must be coco, yolo or yolo-pose")
        if self.box not in ("visible", "amodal"):
            raise ValueError("box must be visible or amodal")


def class_of(obj: dict, classes: tuple[str, ...]) -> int | None:
    """The class index of a labeled object, or None when it isn't one of the classes wanted."""
    cat = obj["category"]
    for name in ([f"{cat}_{obj['alliance']}"] if "alliance" in obj else []) + [cat]:
        if name in classes:
            return classes.index(name)
    return None


def keep(obj: dict, ec: ExportConfig) -> bool:
    box = obj["bbox"] if ec.box == "visible" else obj["bbox_amodal"]
    if box is None or obj["visible_pixels"] < ec.min_visible_px:
        return False
    if obj.get("camera_robot") and not ec.include_camera_robot:
        return False
    if min(box[2], box[3]) < ec.min_box_side_px or obj["visibility"] < ec.min_visibility:
        return False
    if obj["truncation"] is not None and obj["truncation"] > ec.max_truncation:
        return False
    return obj["category"] != "fuel" or obj.get("state") in ec.fuel_states


def _keypoints(obj: dict, width: int, height: int, edge: bool) -> list[float]:
    """The 4 tag corners as (x, y, v): v = 2 visible, 1 in the image but hidden, 0 outside it.
    ``edge`` adds 0.5 (pixel-edge coordinates, like YOLO's boxes); otherwise pixel centers, as COCO
    keypoint tools (detectron2, mmpose) read them."""
    kp = []
    shift = 0.5 if edge else 0.0
    for p, seen in zip(obj["corners"], obj["corners_visible"]):
        inside = p[0] is not None and -0.5 <= p[0] <= width - 0.5 and -0.5 <= p[1] <= height - 0.5
        kp += [p[0] + shift, p[1] + shift, 2 if seen else 1] if inside else [0.0, 0.0, 0]
    return kp


def export(root, ec: ExportConfig, progress=print) -> dict:
    root = Path(root)
    out = root / "exports" / ec.name
    if out.exists():
        assert out.resolve().parent == (root / "exports").resolve()  # only ever delete an old export
        shutil.rmtree(out)
    out.mkdir(parents=True)
    counts: dict = {"images": {}, "objects": {c: 0 for c in ec.classes}}
    coco: dict[str, dict] = {}
    for rec in load_records(root):
        split = rec["split"]
        if split not in ec.splits:
            continue
        counts["images"][split] = counts["images"].get(split, 0) + 1
        w, h = rec["width"], rec["height"]
        kept = [(o, class_of(o, ec.classes)) for o in rec["objects"] if keep(o, ec)]
        kept = [(o, c) for o, c in kept if c is not None]
        for _, c in kept:
            counts["objects"][ec.classes[c]] += 1
        if ec.format == "coco":
            doc = coco.setdefault(split, {"images": [], "annotations": [], "categories": [
                {"id": i + 1, "name": n, **({"keypoints": ["bottom_left", "bottom_right", "top_right", "top_left"],
                                            "skeleton": [[1, 2], [2, 3], [3, 4], [4, 1]]} if n == "apriltag" else {})}
                for i, n in enumerate(ec.classes)]})
            img_id = len(doc["images"]) + 1
            doc["images"].append({"id": img_id, "file_name": rec["image"], "width": w, "height": h})
            for o, c in kept:
                box = o["bbox"] if ec.box == "visible" else o["bbox_amodal"]
                ann = {"id": len(doc["annotations"]) + 1, "image_id": img_id, "category_id": c + 1, "bbox": box,
                       "area": o["visible_pixels"] if ec.box == "visible" else o["covered_pixels"], "iscrowd": 0,
                       "visibility": o["visibility"], "truncation": o["truncation"], "object_id": o["id"]}
                if o["category"] == "apriltag":
                    ann["keypoints"] = _keypoints(o, w, h, edge=False)
                    ann["num_keypoints"] = sum(1 for v in ann["keypoints"][2::3] if v > 0)
                    ann["tag_id"] = o["tag_id"]
                doc["annotations"].append(ann)
        else:
            src = root / rec["image"]
            dst = out / "images" / split / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)  # no extra disk space (NTFS supports hard links too)
            except OSError:
                shutil.copy2(src, dst)
            lines = []
            for o, c in kept:
                x, y, bw, bh = o["bbox"] if ec.box == "visible" else o["bbox_amodal"]
                line = f"{c} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
                if ec.format == "yolo-pose":
                    kp = _keypoints(o, w, h, edge=True) if o["category"] == "apriltag" else [0.0, 0.0, 0] * 4
                    line += "".join(f" {kp[i] / w:.6f} {kp[i + 1] / h:.6f} {kp[i + 2]}" for i in range(0, 12, 3))
                lines.append(line)
            lab = out / "labels" / split / (src.stem + ".txt")
            lab.parent.mkdir(parents=True, exist_ok=True)
            lab.write_text("".join(s + "\n" for s in lines), encoding="utf-8")
    if ec.format == "coco":
        (out / "annotations").mkdir()
        for split, doc in coco.items():
            doc["info"] = {"description": f"{root.name} / {ec.name} ({split})", "images_root": str(root.resolve())}
            (out / "annotations" / f"{split}.json").write_text(json.dumps(doc), encoding="utf-8")
    else:
        for split in ec.splits:  # every split gets its folders, even an empty one
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        yaml = [f"# {ec.name}: exported from {root.name} by scripts/export_dataset.py (see export.json)",
                f"path: {out.resolve().as_posix()}",
                *[f"{s}: images/{s}" for s in ec.splits],
                "names:", *[f"  {i}: {n}" for i, n in enumerate(ec.classes)]]
        if ec.format == "yolo-pose":
            yaml += ["kpt_shape: [4, 3]  # the 4 AprilTag corners: bottom-left, bottom-right, top-right, top-left"]
        (out / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8")) if (root / "manifest.json").exists() else {}
    (out / "export.json").write_text(json.dumps({"settings": asdict(ec), "counts": counts,
                                                 "dataset_records_sha256": manifest.get("records_sha256")}, indent=2),
                                     encoding="utf-8")
    empty = [s for s in ec.splits if not counts["images"].get(s)]
    if empty:
        progress(f"WARNING: no images in {empty}: this dataset has too few matches for its splits "
                 "(training tools need a train and a val split); generate more matches")
    progress(f"exported {sum(counts['images'].values())} images to {out}")
    return counts
