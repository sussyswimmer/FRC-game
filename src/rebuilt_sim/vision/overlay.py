"""Draw labels on top of an image, to check them by eye (quality assurance)."""

from __future__ import annotations

import numpy as np

COLORS = {"fuel": (255, 220, 0), "robot": (255, 0, 255), "apriltag": (0, 255, 0), "hub": (0, 200, 255),
          "tower": (255, 128, 0), "trench": (160, 160, 255), "bump": (255, 90, 90), "outpost": (200, 255, 200),
          "depot": (255, 255, 255)}
CORNER_COLORS = ((255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 255, 0))  # BL, BR, TR, TL


def draw_labels(image: np.ndarray, objects: list[dict], min_pixels: float = 1.0, amodal: bool = False):
    """A PIL image with every object's box (and each tag's corners and id) drawn on ``image``."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(image if image.ndim == 3 else np.repeat(image[..., None], 3, axis=-1)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for o in objects:
        box = o.get("bbox_amodal" if amodal else "bbox")
        if box is None or o["visible_pixels"] < min_pixels:
            continue
        x, y, w, h = box
        color = COLORS.get(o["category"], (255, 255, 255))
        draw.rectangle([x, y, x + max(w - 1, 0), y + max(h - 1, 0)], outline=color, width=1)
        if o["category"] == "apriltag":
            draw.text((x + 1, y - 11), f"{o['tag_id']}", fill=color)
            for k, (p, seen) in enumerate(zip(o["corners"], o["corners_visible"])):
                if p[0] is None:
                    continue
                r = 2 if seen else 1
                draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], outline=CORNER_COLORS[k])
        elif o["category"] == "robot":
            draw.text((x + 1, y + 1), f"{o['alliance'][0].upper()}{o['team_number']}", fill=color)
    return img
