"""Check a generated dataset: every file there, every label sane, and (optionally) the AprilTags
readable by the real detector. Plus a contact sheet: a grid of images with their labels drawn on,
for looking at the data, which is the most useful quality check there is."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .dataset import load_records


def validate(root, decode_tags: int = 0, progress=print) -> dict:
    """Problems found (an empty list is a pass), and how well the tag detector agrees with the labels
    on up to ``decode_tags`` clearly visible tags (needs pupil-apriltags)."""
    from PIL import Image

    root = Path(root)
    problems: list[str] = []
    images = 0
    detector = None
    if decode_tags:
        from pupil_apriltags import Detector

        detector = Detector(families="tag36h11")
    tags = {"checked": 0, "read": 0, "wrong_id": 0, "corner_err_px": []}
    for rec in load_records(root):
        images += 1
        name = rec["image"]
        for key in ("image", "masks", "depth"):
            if rec.get(key) and not (root / rec[key]).exists():
                problems.append(f"{name}: missing {rec[key]}")
        w, h = rec["width"], rec["height"]
        for o in rec["objects"]:
            if not 0.0 <= o["visibility"] <= 1.0 or o["visible_pixels"] > o["covered_pixels"] + 1e-6:
                problems.append(f"{name}: object {o['id']} has visibility {o['visibility']}")
            for key in ("bbox", "bbox_amodal"):
                b = o[key]
                if b is not None and (b[0] < -1e-6 or b[1] < -1e-6 or b[0] + b[2] > w + 1e-6 or b[1] + b[3] > h + 1e-6):
                    problems.append(f"{name}: object {o['id']} {key} {b} leaves the image")
        if detector is None or tags["checked"] >= decode_tags:
            continue
        clear = [o for o in rec["objects"] if o["category"] == "apriltag" and o["bbox"] and o["visibility"] > 0.999
                 and all(o["corners_visible"]) and o["truncation"] == 0.0 and (o["min_edge_px"] or 0) >= 24
                 and o["view_angle_deg"] < 60 and rec["render"]["effects"]["motion_px"] < 1]
        if not clear:
            continue
        img = np.asarray(Image.open(root / name).convert("L"))
        found = {d.tag_id: d for d in detector.detect(img)}
        labeled = {o["tag_id"] for o in rec["objects"] if o["category"] == "apriltag" and o["bbox"]}
        tags["wrong_id"] += sum(1 for t in found if t not in labeled)
        for o in clear:
            tags["checked"] += 1
            d = found.get(o["tag_id"])
            if d is not None:
                tags["read"] += 1
                err = np.abs(np.asarray(d.corners) - (np.asarray(o["corners"]) + 0.5)).max()
                tags["corner_err_px"].append(float(err))
    errs = tags.pop("corner_err_px")
    if errs:
        tags["corner_err_px_median_p95"] = [round(float(np.median(errs)), 2), round(float(np.percentile(errs, 95)), 2)]
    if tags["wrong_id"]:
        problems.append(f"the detector read {tags['wrong_id']} tag IDs that aren't labeled")
    progress(f"{images} images checked, {len(problems)} problems" + (f", tags {tags}" if detector else ""))
    return {"images": images, "problems": problems, "tags": tags if detector else None}


def contact_sheet(root, out, count: int = 16, columns: int = 4, split: str | None = None, seed: int = 0,
                  amodal: bool = False) -> Path:
    """A grid of random images from the dataset with their labels drawn on."""
    from PIL import Image

    from .overlay import draw_labels

    root = Path(root)
    records = list(load_records(root, split))
    if not records:
        raise ValueError(f"no records in {root}")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(records), min(count, len(records)), replace=False)
    tiles = []
    for i in sorted(int(i) for i in pick):
        rec = records[i]
        image = np.asarray(Image.open(root / rec["image"]).convert("RGB"))
        tiles.append(draw_labels(image, rec["objects"], amodal=amodal))
    tw, th = 480, 300
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * tw, rows * th), (20, 20, 20))
    for k, tile in enumerate(tiles):
        tile.thumbnail((tw, th))
        sheet.paste(tile, ((k % columns) * tw, (k // columns) * th))
    out = Path(out)
    sheet.save(out)
    return out
