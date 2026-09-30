"""What a generated dataset is: its statistics, a manifest (how exactly it was made) and a dataset
card (a README for the people who train on it).

Professional datasets ship with both. The manifest makes a dataset reproducible: config, code
version (git commit), library versions and file checksums. The card ("datasheets for datasets",
Gebru et al. 2018) tells whoever trains on it months later what is in it, how it was made, how to
read the labels, and what it can't be trusted for.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def stats(root) -> dict:
    """Counts and distributions over the whole dataset (everything, before any export filter)."""
    from .dataset import load_records

    root = Path(root)
    images: dict[str, int] = {}
    objects: dict[str, dict] = {}
    presets: dict[str, int] = {}
    empty = 0
    for r in load_records(root):
        images[r["split"]] = images.get(r["split"], 0) + 1
        presets[r["camera"]["preset"]] = presets.get(r["camera"]["preset"], 0) + 1
        seen = [o for o in r["objects"] if o["bbox"] is not None]
        empty += not seen
        for o in seen:
            s = objects.setdefault(o["category"], {"count": 0, "side": [], "visibility": []})
            s["count"] += 1
            s["side"].append(min(o["bbox"][2], o["bbox"][3]))
            s["visibility"].append(o["visibility"])
    summary = {}
    for c, s in sorted(objects.items()):
        side, vis = np.array(s["side"]), np.array(s["visibility"])
        summary[c] = {"count": s["count"], "per_image": round(s["count"] / max(sum(images.values()), 1), 2),
                      "box_side_px_p10_p50_p90": [round(float(np.percentile(side, q)), 1) for q in (10, 50, 90)],
                      "partly_hidden": round(float((vis < 0.99).mean()), 3)}
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and "exports" not in p.parts)
    return {"images": images, "empty_images": empty, "presets": presets, "objects": summary,
            "disk_mb": round(size / 1e6, 1)}


def git_info() -> dict:
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=here.parent, capture_output=True,
                               text=True, timeout=10)
        return {"commit": commit.stdout.strip() or None, "simulator_code_changed": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "simulator_code_changed": None}


def write_manifest(root, cfg, st: dict, seconds: float) -> None:
    import PIL

    from .dataset import SCHEMA

    root = Path(root)
    shards = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((root / "records").glob("*/m*.jsonl.gz"))}
    manifest = {
        "schema": SCHEMA,
        "name": cfg.name,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pillow": PIL.__version__,
                     "platform": platform.platform()},
        "config": json.loads(json.dumps(asdict(cfg))),
        "stats": st,
        "seconds": round(seconds, 1),
        "records_sha256": shards,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def dataset_card(cfg, st: dict) -> str:
    images = st.get("images", {})
    total = sum(images.values())
    rows = [f"| {c} | {s['count']} | {s['per_image']} | {' / '.join(str(v) for v in s['box_side_px_p10_p50_p90'])} |"
            f" {100 * s['partly_hidden']:.0f}% |" for c, s in st.get("objects", {}).items()]
    lines = [
        f"# Dataset card: {cfg.name}",
        "",
        "Synthetic camera images of FRC 2026 REBUILT matches, rendered by `rebuilt_sim.vision` (step 4 of",
        "this project; see docs/04-vision.md) with `scripts/make_dataset.py`. `manifest.json` has the exact",
        "config, code version and checksums.",
        "",
        "## Contents",
        "",
        f"- {total} images from {cfg.matches} simulated matches ({cfg.frames_per_match} moments per match,"
        f" {cfg.cameras_per_frame} camera(s) per moment); {st.get('empty_images', 0)} show no labeled object.",
        "- Images per split: " + ", ".join(f"{s} {n}" for s, n in sorted(images.items())) + ". Splits are by match:",
        "  no match appears in two splits.",
        "- Cameras: " + ", ".join(f"{p} {n}" for p, n in sorted(st.get("presets", {}).items())) +
        f"; at {cfg.scale:g} x native resolution, rendered {cfg.supersample} x {cfg.supersample} supersampled.",
        f"- Physics: {'high fidelity (real FUEL flight heights and robot motion)' if cfg.hifi else 'strategy (made-up FUEL arcs)'};"
        f" domain randomization {'on' if cfg.randomize else 'off'}; camera effects strength {cfg.effects_strength:g}.",
        f"- Disk: {st.get('disk_mb', '?')} MB.",
        "",
        "Objects with at least one visible pixel (every object is in the records, filtering happens at export):",
        "",
        "| Category | Count | Per image | Box side px (10th / 50th / 90th percentile) | Partly hidden |",
        "|---|---|---|---|---|",
        *rows,
        "",
        "## Files",
        "",
        f"- `images/<split>/`: {'PNG (lossless)' if cfg.lossless else 'JPEG (quality 95, lower when compression was drawn as a camera effect)'}."
        " Monochrome cameras give single-channel images.",
        "- `records/<split>/m<match>.jsonl.gz`: one JSON object per image with everything: camera intrinsics",
        "  and distortion, camera and robot poses (also as WPILib `Pose3d`/`Transform3d`), match state,",
        "  lighting, camera effects, and every object in view, even fully hidden ones (visibility 0). Read",
        "  them with `rebuilt_sim.vision.dataset.load_records(path)`.",
        "- `index.csv`: one row per image (time, camera, counts), for browsing and filtering.",
        "- Training formats: make them with `scripts/export_dataset.py` (COCO, YOLO); you choose the format,",
        "  classes, box style and filters there.",
    ]
    if cfg.masks:
        lines.append("- `masks/<split>/*.png`: 16-bit object id + 1 per pixel (0 = backdrop); ids in `vision/ids.py`.")
    if cfg.depth:
        lines.append("- `depth/<split>/*.png`: 16-bit depth along the optical axis in millimeters (0 = nothing).")
    lines += [
        "",
        "## Conventions",
        "",
        "- Field coordinates: WPILib blue-origin field frame, meters. Camera frame: OpenCV (x right, y down,",
        "  z forward). Quaternions are [w, x, y, z].",
        "- Boxes: `[x, y, w, h]` in pixel edge coordinates (the image spans `[0, width] x [0, height]`).",
        "  `bbox` covers the visible pixels; `bbox_amodal` covers the whole object as if nothing were in",
        "  front of it (still clipped to the image).",
        "- Points (tag corners) use OpenCV pixel-center coordinates: the top-left pixel's center is (0, 0).",
        "  The AprilTag library (WPILib, PhotonVision, pupil-apriltags) reports the same corners 0.5 px",
        "  further right and down.",
        "- `visibility` = visible pixels / pixels the object would cover with nothing in front of it.",
        "  `truncation` = fraction of it outside the image (or behind the camera).",
        "- Robot labels carry `camera_robot`: true for the robot carrying the camera (its own bumper or",
        "  intake at the image's edge). Whether to train on those is an export choice.",
        "- Distortion: OpenCV `k1 k2 p1 p2 k3`.",
        "- AprilTags: family 36h11, black-square size 0.1651 m. Corners are listed like WPILib and the",
        "  AprilTag library: counter-clockwise from the printed bottom-left. OpenCV's ArUco AprilTag",
        "  dictionary lists them in the order `[1, 0, 3, 2]` of these.",
        "",
        "## Known limits",
        "",
        "- **Tags are chiral.** A mirrored AprilTag is a different (invalid) tag, and its corners swap.",
        "  Flip augmentations (Ultralytics' default `fliplr=0.5`, for example) corrupt tag IDs and corners.",
        "  Whether to turn flips off is your training decision.",
        "- The 3D field follows the official drawings. The strategy simulator still uses older positions",
        "  for the TOWER, OUTPOST and DEPOT (docs/04-vision.md, 'Simulator vs drawings'), so DEPOT FUEL and",
        "  climbing robots can sit away from the drawn structures.",
        "- Simple shapes and simple lighting: no mesh-level detail, soft shadows only under FUEL and robots,",
        "  invisible polycarbonate (no tint or reflections), no rolling-shutter skew.",
        "- Robots are random boxes in bumpers, not real robot designs; FUEL has no printed logos.",
        "- A synthetic-only dataset has a sim-to-real gap. The standard way to measure it is a small set",
        "  of real, hand-labeled images held out for testing.",
        "",
    ]
    return "\n".join(lines)
