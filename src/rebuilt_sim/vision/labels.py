"""Labels from a rendered frame: what each object is, where it is in the image, and how much of it
the camera actually sees.

The renderer knows exactly which object every pixel shows (the id buffer) and how many pixels each
object would cover if nothing were in front of it (its coverage). From those:

- **visible box:** the tight box around the pixels that show the object. This is what detectors
  are usually trained on.
- **amodal box:** the box around everything the object would cover with nothing in front of it
  (still clipped to the image). "Amodal" is the vision term for "including the hidden parts".
- **visibility:** visible pixels / covered pixels. 1 = in full view, 0.2 = mostly hidden.
- **truncation:** the fraction of the object that falls outside the image (cut off by the image
  edge): exact for FUEL (its round silhouette is sampled), from the box around its projected 3D
  bounding points for everything else. None when part of it is behind the camera.

Conventions (see docs/04-vision.md):

- Boxes are ``[x, y, w, h]`` in pixel *edge* coordinates like COCO: the image spans
  ``[0, width] x [0, height]``.
- Points (tag corners, FUEL centers) are in OpenCV's pixel *center* coordinates: the top-left pixel's
  center is (0, 0). Add 0.5 to convert to edge coordinates.
- Frames can be rendered ``k`` times finer than the output image (supersampling, to smooth edges);
  labels are always given at the output resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .camera import Camera, rodrigues
from .render import Frame

CATEGORIES = ("fuel", "robot", "apriltag", "hub", "tower", "trench", "bump", "outpost", "depot")


@dataclass
class ObjectInfo:
    """What the scene builder knows about one labeled object."""

    category: str
    attributes: dict = field(default_factory=dict)
    points: np.ndarray | None = None  # (m, 3) field-frame points that bound it (for truncation)
    corners: np.ndarray | None = None  # (4, 3) AprilTag black-square corners (BL, BR, TR, TL)
    sphere: tuple[np.ndarray, float] | None = None  # FUEL: center and radius


_GRID = (np.arange(16) + 0.5) / 16  # stratified sample positions in [0, 1]


def _outside_fraction(camera: Camera, x: np.ndarray, y: np.ndarray) -> float:
    """Fraction of normalized image points (X/Z, Y/Z) that land outside the image. Points beyond the
    lens model's valid radius count as outside (they would fold back in if pushed through it)."""
    intr = camera.intrinsics
    w, h = intr.width, intr.height
    inside = np.ones(x.shape, dtype=bool)
    if any(intr.dist):
        cu, cv = np.array([-0.5, w - 0.5, -0.5, w - 0.5]), np.array([-0.5, -0.5, h - 0.5, h - 0.5])
        ux, uy = intr.undistort(cu, cv)
        inside &= x * x + y * y <= 1.2 ** 2 * float(np.max(ux * ux + uy * uy))
    u, v = intr.distort(x, y)
    inside &= (u >= -0.5) & (u <= w - 0.5) & (v >= -0.5) & (v <= h - 0.5)
    return float(1.0 - inside.mean())


def _truncation(camera: Camera, info: "ObjectInfo") -> float | None:
    if info.sphere is not None:  # FUEL: sample its silhouette, the disk the tangent rays touch
        center, r = info.sphere
        c = camera.to_camera(np.asarray(center)[None])[0]
        d2 = float(c @ c)
        if d2 <= r * r:
            return None
        base = c * (1.0 - r * r / d2)
        rad = r * np.sqrt(1.0 - r * r / d2)
        a = np.cross(c, (1.0, 0.0, 0.0) if abs(c[0]) < 0.9 * np.sqrt(d2) else (0.0, 1.0, 0.0))
        a /= np.linalg.norm(a)
        b = np.cross(c, a) / np.sqrt(d2)
        rr, th = np.meshgrid(np.sqrt(_GRID), _GRID * 2 * np.pi)
        pts = base + rad * (rr * np.cos(th)).reshape(-1, 1) * a + rad * (rr * np.sin(th)).reshape(-1, 1) * b
        if (pts[:, 2] <= 1e-6).any():
            return None
        return _outside_fraction(camera, pts[:, 0] / pts[:, 2], pts[:, 1] / pts[:, 2])
    if info.points is None:
        return None
    pc = camera.to_camera(info.points)
    if (pc[:, 2] <= 1e-6).any():
        return None  # partly behind the camera: its projection is unbounded
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    gx, gy = np.meshgrid(x.min() + _GRID * (x.max() - x.min()), y.min() + _GRID * (y.max() - y.min()))
    return _outside_fraction(camera, gx.ravel(), gy.ravel())


def _box(u0: int, u1: int, v0: int, v1: int, k: int) -> list[float]:
    """Inclusive fine-pixel range -> [x, y, w, h] in output edge coordinates."""
    return [u0 / k, v0 / k, (u1 + 1 - u0) / k, (v1 + 1 - v0) / k]


def annotate(frame: Frame, camera: Camera, objects: dict[int, ObjectInfo], k: int = 1) -> list[dict]:
    """One label per object that appears in the frame (any visible or hidden-but-covered pixel).
    ``camera`` is at the output resolution; ``frame`` was rendered ``k`` times finer."""
    out = []
    kk = k * k
    for obj, covered in sorted(frame.coverage.items()):
        info = objects.get(obj)
        if info is None:
            continue
        u0, u1, v0, v1 = frame.extent[obj]
        win = frame.obj[v0:v1 + 1, u0:u1 + 1] == obj
        visible = int(win.sum())
        label = {
            "id": obj,
            "category": info.category,
            **info.attributes,
            "visible_pixels": visible / kk,
            "covered_pixels": covered / kk,
            "visibility": visible / covered,
            "bbox_amodal": _box(u0, u1, v0, v1, k),
            "bbox": None,
            "truncation": _truncation(camera, info),
        }
        if visible:
            cols, rows = np.flatnonzero(win.any(axis=0)), np.flatnonzero(win.any(axis=1))
            label["bbox"] = _box(u0 + cols[0], u0 + cols[-1], v0 + rows[0], v0 + rows[-1], k)
        if info.points is not None:
            center = info.points.mean(axis=0)
            label["distance"] = float(np.linalg.norm(center - camera.position))
        if info.corners is not None:
            label.update(_tag_corners(frame, camera, info.corners, obj, k))
        out.append(label)
    return out


def _tag_corners(frame: Frame, camera: Camera, corners: np.ndarray, obj: int, k: int) -> dict:
    """Pixel corners of a tag's black square, and whether each is in the image and unhidden."""
    uv, z = camera.project(corners)
    h, w = frame.obj.shape
    seen = []
    for (u, v), depth in zip(uv, z):
        ok = bool(depth > 0 and np.isfinite(u) and np.isfinite(v))
        if ok:
            fu, fv = int(round(u * k + (k - 1) / 2)), int(round(v * k + (k - 1) / 2))
            ok = 0 <= fu < w and 0 <= fv < h and frame.obj[fv, fu] == obj
        seen.append(ok)
    center = corners.mean(axis=0)
    right = corners[1] - corners[0]
    up = corners[3] - corners[0]
    right, up = right / np.linalg.norm(right), up / np.linalg.norm(up)
    normal = np.cross(right, up)  # out of the printed face
    to_cam = camera.position - center
    angle = math.degrees(math.acos(float(np.clip(normal @ to_cam / np.linalg.norm(to_cam), -1.0, 1.0))))
    # the tag's pose in the camera frame, as cv2.solvePnP reports it with tags.tag_object_points():
    # tag frame x right, y down, z into the tag
    rot = camera.rotation.T @ np.column_stack([right, -up, -normal])
    edges = [float(np.linalg.norm(uv[(i + 1) % 4] - uv[i])) for i in range(4)] if np.isfinite(uv).all() else []
    return {
        "corners": [[float(u), float(v)] for u, v in uv],  # BL, BR, TR, TL as printed
        "corners_visible": seen,
        "corners_field": corners.tolist(),
        "min_edge_px": min(edges) if edges else None,  # the shortest side of the black square in the image
        "view_angle_deg": angle,  # 0 = seen head-on
        "pose_in_camera": {"rotation": rot.tolist(), "translation": camera.to_camera(center[None])[0].tolist(),
                           "rvec": rodrigues(rot).tolist()},
    }
