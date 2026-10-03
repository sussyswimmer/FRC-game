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
  edge, or behind the camera): exact for FUEL (its round silhouette is sampled); for everything
  else, the share of points spread through its 3D box that the camera can't see.

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
from .render import NEAR, Frame

CATEGORIES = ("fuel", "robot", "apriltag", "hub", "tower", "trench", "bump", "outpost", "depot")


@dataclass
class ObjectInfo:
    """What the scene builder knows about one labeled object."""

    category: str
    attributes: dict = field(default_factory=dict)
    points: np.ndarray | None = None  # (m, 3) field-frame points that bound it (for truncation)
    corners: np.ndarray | None = None  # (4, 3) AprilTag black-square corners (BL, BR, TR, TL)
    sphere: tuple[np.ndarray, float] | None = None  # FUEL: center and radius
    children: tuple[int, ...] = ()  # ids of objects that are part of it (the tags on a structure)


_GRID = (np.arange(16) + 0.5) / 16  # stratified sample positions in [0, 1]


def _lens_ok(camera: Camera, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Which normalized image points (X/Z, Y/Z) the lens model can map to pixels. Beyond about 1.2
    times the image corners' radius a distortion polynomial may fold back into the image, so points
    out there are treated as outside it (Intrinsics.lens_error checks the model up to that radius)."""
    intr = camera.intrinsics
    if not any(intr.dist):
        return np.ones(np.shape(x), dtype=bool)
    w, h = intr.width, intr.height
    ux, uy = intr.undistort(np.array([-0.5, w - 0.5, -0.5, w - 0.5]), np.array([-0.5, -0.5, h - 0.5, h - 0.5]))
    return x * x + y * y <= 1.2 ** 2 * float(np.max(ux * ux + uy * uy))


def _inside(camera: Camera, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Which normalized image points (X/Z, Y/Z) land inside the image."""
    intr = camera.intrinsics
    u, v = intr.distort(x, y)
    return _lens_ok(camera, x, y) & (u >= -0.5) & (u <= intr.width - 0.5) & (v >= -0.5) & (v <= intr.height - 0.5)


def _outside_fraction(camera: Camera, x: np.ndarray, y: np.ndarray) -> float:
    """Fraction of normalized image points (X/Z, Y/Z) that land outside the image."""
    return float(1.0 - _inside(camera, x, y).mean())


def _truncation(camera: Camera, info: "ObjectInfo") -> float | None:
    """The fraction of an object outside the image (1 = all of it); parts behind the camera count as
    outside. For FUEL: the share of its round silhouette. For everything else: the share of points
    spread through its 3D box (or across a tag's face) that the camera can't see, whatever is in
    front of them. None only for objects without geometry."""
    if info.sphere is not None:  # FUEL: sample its silhouette, the disk the tangent rays touch
        center, r = info.sphere
        c = camera.to_camera(np.asarray(center)[None])[0]
        d2 = float(c @ c)
        if d2 <= r * r:
            return 1.0  # the lens is inside the ball
        base = c * (1.0 - r * r / d2)
        rad = r * np.sqrt(1.0 - r * r / d2)
        a = np.cross(c, (1.0, 0.0, 0.0) if abs(c[0]) < 0.9 * np.sqrt(d2) else (0.0, 1.0, 0.0))
        a /= np.linalg.norm(a)
        b = np.cross(c, a) / np.sqrt(d2)
        rr, th = np.meshgrid(np.sqrt(_GRID), _GRID * 2 * np.pi)
        pts = base + rad * (rr * np.cos(th)).reshape(-1, 1) * a + rad * (rr * np.sin(th)).reshape(-1, 1) * b
        front = pts[:, 2] > NEAR
        if not front.any():
            return 1.0
        inside = 1.0 - _outside_fraction(camera, pts[front, 0] / pts[front, 2], pts[front, 1] / pts[front, 2])
        return float(1.0 - inside * front.mean())
    if info.points is None:
        return None
    pc = camera.to_camera(_spread(np.asarray(info.points, dtype=np.float64)))
    front = pc[:, 2] > NEAR
    inside = np.zeros(len(pc), dtype=bool)
    if front.any():
        x, y = pc[front, 0] / pc[front, 2], pc[front, 1] / pc[front, 2]
        inside[front] = _inside(camera, x, y)
    return float(1.0 - inside.mean())


_T8 = (np.arange(8) + 0.5) / 8


def _spread(points: np.ndarray) -> np.ndarray:
    """Sample points spread through an object: through its box when given its 8 corners (in x, y, z
    bit order), across its face when given 4 (top-left, top-right, bottom-right, bottom-left)."""
    if len(points) == 8:
        u, v, w = (g.ravel()[:, None] for g in np.meshgrid(_T8, _T8, _T8, indexing="ij"))
        c = points.reshape(2, 2, 2, 3)
        return ((1 - u) * ((1 - v) * ((1 - w) * c[0, 0, 0] + w * c[0, 0, 1]) + v * ((1 - w) * c[0, 1, 0] + w * c[0, 1, 1]))
                + u * ((1 - v) * ((1 - w) * c[1, 0, 0] + w * c[1, 0, 1]) + v * ((1 - w) * c[1, 1, 0] + w * c[1, 1, 1])))
    if len(points) == 4:
        u, v = (g.ravel()[:, None] for g in np.meshgrid(_GRID, _GRID, indexing="ij"))
        return points[0] + u * (points[1] - points[0]) + v * (points[3] - points[0])
    lo, hi = points.min(axis=0), points.max(axis=0)
    u, v, w = (g.ravel()[:, None] for g in np.meshgrid(_T8, _T8, _T8, indexing="ij"))
    return lo + np.hstack([u, v, w]) * (hi - lo)


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
        block = frame.obj[v0:v1 + 1, u0:u1 + 1]
        # a structure's own tags are part of it, not something in front of it
        win = np.isin(block, (obj, *info.children)) if info.children else block == obj
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
    pc = camera.to_camera(corners)
    with np.errstate(divide="ignore", invalid="ignore"):
        valid = (z > 1e-9) & _lens_ok(camera, pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2])
    uv[~valid] = np.nan  # behind the camera, or so far out that the lens model would fold it back in
    h, w = frame.obj.shape
    seen = []
    for (u, v), ok in zip(uv, valid):
        ok = bool(ok)
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
