"""A small CPU renderer: exact ray tests per pixel, a z-buffer, and an object-id buffer.

Every piece of geometry is drawn the same way:

1. Project its corners to find the pixels it could cover (its screen bounding box). If part of it
   is behind the camera, it could cover the whole image.
2. Intersect those pixels' rays with it exactly: ray-sphere for FUEL, the slab test for boxes,
   ray-plane plus an inside test for flat polygons.
3. Keep a hit where it is nearer than what the z-buffer holds, and record the object's id.

Rays come from the camera model, so lens distortion needs nothing special. Shading is deferred:
drawing only records each pixel's surface color (albedo), normal and material, and one pass at the
end lights every pixel at once: an ambient term, a sky term that brightens upward faces, one
directional light, a highlight on shiny materials (FUEL), and soft "blob" shadows on the carpet
under FUEL and robots. The id buffer gives exact, occlusion-aware labels (see ``labels.py``).

Everything is numpy: about 1.35 s per million rendered pixels on one core for a full match (so
0.3-1 s for 640 x 480), less for simple scenes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import Camera, Intrinsics

NEAR = 0.02  # m, nothing nearer to the lens is drawn
MATTE, SHINY, CARPET = 0, 1, 2  # materials in the per-pixel material buffer


@dataclass
class Light:
    direction: tuple[float, float, float] = (0.3, 0.2, 1.0)  # toward the light, field frame
    ambient: float = 0.35
    sky: float = 0.25  # extra light on faces pointing up
    diffuse: float = 0.55
    specular: float = 0.25  # highlight strength on shiny materials (FUEL)
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class Sphere:
    center: tuple[float, float, float]
    radius: float
    color: tuple[float, float, float]
    obj: int = -1
    shiny: bool = False


@dataclass
class Box:
    """An oriented box: ``rotation`` columns are its local axes in the field frame, ``half`` its half sizes."""

    center: tuple[float, float, float]
    half: tuple[float, float, float]
    color: tuple[float, float, float]
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    obj: int = -1
    emission: tuple[float, float, float] | None = None  # light it gives off (lamps, LEDs)


@dataclass
class Cylinder:
    """A capped cylinder around the local z axis (``rotation``'s third column) through ``center``."""

    center: tuple[float, float, float]
    radius: float
    half_length: float
    color: tuple[float, float, float]
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    obj: int = -1
    emission: tuple[float, float, float] | None = None


@dataclass
class Polygon:
    """A flat convex polygon (vertices in order around it). With a ``texture`` (h, w, 3 or 4 array,
    row 0 = top), vertices 0, 1, 3 are the texture's top-left, top-right and bottom-left corners. A
    4th texture channel is a cut-out mask: where it is below 0.5 the polygon has a hole (nets, frames)."""

    vertices: np.ndarray
    color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    obj: int = -1
    texture: np.ndarray | None = None
    emission: tuple[float, float, float] | None = None
    # one-sided polygons (printed tags) are only seen from the side where their vertices run
    # clockwise, like reading a page; from behind they are not drawn at all
    two_sided: bool = True


@dataclass
class Scene:
    spheres: list[Sphere] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)
    polygons: list[Polygon] = field(default_factory=list)
    cylinders: list[Cylinder] = field(default_factory=list)
    floor: object | None = None  # callable (x, y) -> (n, 3) albedo, drawn on the plane z = 0
    floor_obj: int = -1
    background: object | None = None  # callable (directions (n, 3) field frame) -> (n, 3) color
    multipart: set[int] = field(default_factory=set)  # objects built from several primitives (robots)
    # soft shadows on the floor: rows of (x, y, radius, strength), darkest (by ``strength``) at the center
    blobs: np.ndarray | None = None


@dataclass
class Frame:
    """A rendered image and its per-pixel buffers (all at the rendered resolution)."""

    color: np.ndarray  # (h, w, 3) float32, linear 0..1 (can exceed 1 in highlights)
    depth: np.ndarray  # (h, w) float32, meters along the optical axis (inf: nothing)
    obj: np.ndarray  # (h, w) int32 object id (-1: none)
    coverage: dict[int, int]  # object id -> pixels it covers ignoring everything in front of it
    # object id -> (u0, u1, v0, v1), the inclusive pixel range of its coverage: its "amodal" box,
    # as if nothing were in front of it (still clipped to the image)
    extent: dict[int, tuple[int, int, int, int]]
    intrinsics: Intrinsics


_CUBE = np.array([(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=np.float64)


class Renderer:
    def __init__(self, intrinsics: Intrinsics, full_windows: bool = False) -> None:
        self.full_windows = full_windows  # test every pixel against every shape (slow; a reference for tests)
        if not intrinsics.lens_ok(tolerance=0.01):
            raise ValueError(f"lens distortion {intrinsics.dist} folds over inside the image")
        self.intrinsics = intrinsics
        self.rays = intrinsics.rays().astype(np.float32)  # (h, w, 3), camera frame, z = 1
        # Which pixel columns can see a given normalized x = X/Z (and rows a given y): the range of x
        # over each column, made monotonic so a binary search finds the columns. Working from the
        # rays avoids pushing far-off points through the lens model, where it folds back.
        x, y = self.rays[..., 0], self.rays[..., 1]
        self._col_hi = np.maximum.accumulate(x.max(axis=0))
        self._col_lo = np.minimum.accumulate(x.min(axis=0)[::-1])[::-1]
        self._row_hi = np.maximum.accumulate(y.max(axis=1))
        self._row_lo = np.minimum.accumulate(y.min(axis=1)[::-1])[::-1]

    # ---------------------------------------------------------------- public
    def render(self, scene: Scene, camera: Camera, light: Light | None = None) -> Frame:
        light = light or Light()
        h, w = self.intrinsics.height, self.intrinsics.width
        self._cam = camera
        rot = camera.rotation  # field-frame ray directions (h, w, 3); each ray's camera z is 1
        rays = self.rays.astype(np.float64)
        self._dirs = rays[..., :1] * rot[:, 0] + rays[..., 1:2] * rot[:, 1] + rot[:, 2]
        del rays
        self.depth = np.full((h, w), np.inf, dtype=np.float32)
        self.obj = np.full((h, w), -1, dtype=np.int32)
        self.albedo = np.zeros((h, w, 3), dtype=np.float32)
        self.normal = np.zeros((h, w, 3), dtype=np.float32)
        self.material = np.zeros((h, w), dtype=np.uint8)
        self.emit: np.ndarray | None = None  # (h, w, 3), allocated when something glows
        self.cover_count: dict[int, int] = {}
        self.cover_box: dict[int, list[int]] = {}
        self.cover_mask: dict[int, np.ndarray] = {}
        self._multipart = scene.multipart

        if scene.floor is not None:
            self._floor(scene.floor, scene.floor_obj)
        for b in scene.boxes:
            self._box(b)
        for p in scene.polygons:
            self._polygon(p)
        for c in scene.cylinders:
            self._cylinder(c)
        self._spheres(scene.spheres)
        color = self._light_pass(light, scene)
        coverage = dict(self.cover_count)
        extent = {k: tuple(b) for k, b in self.cover_box.items()}
        for k, m in self.cover_mask.items():
            coverage[k] = int(m.sum())
            if coverage[k]:
                cols, rows = np.flatnonzero(m.any(axis=0)), np.flatnonzero(m.any(axis=1))
                extent[k] = (int(cols[0]), int(cols[-1]), int(rows[0]), int(rows[-1]))
        coverage = {k: n for k, n in coverage.items() if n > 0}
        extent = {k: e for k, e in extent.items() if k in coverage}
        return Frame(color, self.depth, self.obj, coverage, extent, self.intrinsics)

    # ---------------------------------------------------------------- lighting
    def _light_pass(self, lt: Light, scene: Scene) -> np.ndarray:
        """Light every drawn pixel from its albedo, normal and material, then fill the background."""
        ldir = np.asarray(lt.direction, dtype=np.float32)
        ldir /= np.linalg.norm(ldir)
        n = self.normal
        k = (lt.ambient + lt.sky * 0.5) + (lt.sky * 0.5) * n[..., 2] + lt.diffuse * np.maximum(n @ ldir, 0.0)
        if scene.blobs is not None and len(scene.blobs):
            k *= self._blob_shade(scene.blobs, self.material == CARPET)
        color = self.albedo * (k[..., None] * np.asarray(lt.color, dtype=np.float32))
        sel = self.material == SHINY
        if lt.specular > 0 and sel.any():
            view = self._dirs[sel]
            view /= np.linalg.norm(view, axis=1, keepdims=True)
            half = ldir - view
            half /= np.linalg.norm(half, axis=1, keepdims=True) + 1e-12
            spec = lt.specular * np.maximum((n[sel] * half).sum(axis=1), 0.0) ** 24
            color[sel] += (spec[:, None] * np.asarray(lt.color)).astype(np.float32)
        if self.emit is not None:
            color += self.emit
        empty = ~np.isfinite(self.depth)
        if empty.any():
            if scene.background is not None:
                color[empty] = scene.background(self._dirs[empty])
            else:
                color[empty] = (0.1, 0.1, 0.12)
        return color

    def _blob_shade(self, blobs: np.ndarray, carpet: np.ndarray) -> np.ndarray:
        """A (h, w) light factor, below 1 where visible carpet lies under a blob."""
        h, w = self.depth.shape
        factor = np.ones((h, w), dtype=np.float32)
        blobs = np.asarray(blobs, dtype=np.float64)
        square = np.array([(-1, -1), (1, -1), (-1, 1), (1, 1)], dtype=np.float64)
        corners = np.zeros((len(blobs), 4, 3))
        corners[..., :2] = blobs[:, None, :2] + square[None] * blobs[:, 2, None, None]
        u0, u1, v0, v1, ok = self._windows(self._cam.to_camera(corners.reshape(-1, 3)).reshape(-1, 4, 3))
        cam = self._cam.position
        for i in np.flatnonzero(ok):
            x, y, r, strength = blobs[i]
            win = (slice(v0[i], v1[i]), slice(u0[i], u1[i]))
            sel = carpet[win]
            if not sel.any():
                continue
            t = self.depth[win]
            d = self._dirs[win]
            px = cam[0] + t * d[..., 0] - x
            py = cam[1] + t * d[..., 1] - y
            q = np.clip(1.0 - (px * px + py * py) / (r * r), 0.0, 1.0)
            factor[win] = np.minimum(factor[win], np.where(sel, 1.0 - strength * q, 1.0))
        return factor

    # ---------------------------------------------------------------- helpers
    def _span(self, xmin, xmax, ymin, ymax):
        """Pixel windows (u0, u1, v0, v1), end-exclusive, that can see normalized ranges (arrays)."""
        u0 = np.searchsorted(self._col_hi, xmin, "left")
        u1 = np.searchsorted(self._col_lo, xmax, "right")
        v0 = np.searchsorted(self._row_hi, ymin, "left")
        v1 = np.searchsorted(self._row_lo, ymax, "right")
        return u0, u1, v0, v1

    def _bbox(self, corners: np.ndarray) -> tuple[int, int, int, int] | None:
        """Pixel window (u0, u1, v0, v1) that may contain a convex shape with these field-frame corners.
        Parts behind the camera are clipped off at the near plane first."""
        pc = self._cam.to_camera(corners)
        front = pc[:, 2] > NEAR
        if not front.any():
            return None
        pts = pc[front]
        if not front.all():
            # where the segments between front and back corners cross the near plane
            a, b = pc[front][:, None, :], pc[~front][None, :, :]
            s = (NEAR - a[..., 2]) / (b[..., 2] - a[..., 2])
            cut = (a + s[..., None] * (b - a)).reshape(-1, 3)
            cut[:, 2] = NEAR
            pts = np.concatenate([pts, cut])
        if self.full_windows:
            return 0, self.intrinsics.width, 0, self.intrinsics.height
        x, y = pts[:, 0] / pts[:, 2], pts[:, 1] / pts[:, 2]
        u0, u1, v0, v1 = (int(v) for v in self._span(x.min(), x.max(), y.min(), y.max()))
        if u0 >= u1 or v0 >= v1:
            return None
        return u0, u1, v0, v1

    def _windows(self, corners: np.ndarray):
        """Pixel windows for many shapes at once from their camera-frame corners (n, m, 3). Returns
        u0, u1, v0, v1 (int arrays) and which shapes may be visible at all. A shape that crosses the
        near plane gets the whole image (rare for small shapes)."""
        h, w = self.intrinsics.height, self.intrinsics.width
        zc = corners[..., 2]
        straddle = (zc <= NEAR).any(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = corners[..., 0] / zc, corners[..., 1] / zc
        u0, u1, v0, v1 = self._span(x.min(axis=1), x.max(axis=1), y.min(axis=1), y.max(axis=1))
        every = straddle | self.full_windows
        u0 = np.where(every, 0, u0)
        u1 = np.where(every, w, u1)
        v0 = np.where(every, 0, v0)
        v1 = np.where(every, h, v1)
        ok = (zc > NEAR).any(axis=1) & (u0 < u1) & (v0 < v1)
        return u0, u1, v0, v1, ok

    def _write(self, window, t: np.ndarray, hit: np.ndarray, obj: int, surface, material: int = MATTE,
               emission=None) -> None:
        """Z-test the hits in a pixel window and record depth, id, albedo and normal where they win.
        ``surface(win)`` returns (albedo, normal), each (n, 3) or (3,), for the winning pixels."""
        u0, u1, v0, v1 = window
        self._cover(window, hit, obj)
        dblock = self.depth[v0:v1, u0:u1]
        win = hit & (t < dblock) & (t > NEAR)
        if not win.any():
            return
        dblock[win] = t[win]
        self.obj[v0:v1, u0:u1][win] = obj
        albedo, normal = surface(win)
        self.albedo[v0:v1, u0:u1][win] = albedo
        self.normal[v0:v1, u0:u1][win] = normal
        self.material[v0:v1, u0:u1][win] = material
        if emission is not None or self.emit is not None:
            if self.emit is None:
                self.emit = np.zeros(self.albedo.shape, dtype=np.float32)
            self.emit[v0:v1, u0:u1][win] = 0.0 if emission is None else emission

    def _cover(self, window, hit: np.ndarray, obj: int) -> None:
        if obj < 0:
            return
        if obj in self._multipart:
            mask = self.cover_mask.get(obj)
            if mask is None:
                mask = self.cover_mask[obj] = np.zeros(self.depth.shape, dtype=bool)
            u0, u1, v0, v1 = window
            mask[v0:v1, u0:u1] |= hit
        else:
            n = int(hit.sum())
            if not n:
                return
            self.cover_count[obj] = self.cover_count.get(obj, 0) + n
            u0, v0 = int(window[0]), int(window[2])
            cols, rows = np.flatnonzero(hit.any(axis=0)), np.flatnonzero(hit.any(axis=1))
            box = [u0 + int(cols[0]), u0 + int(cols[-1]), v0 + int(rows[0]), v0 + int(rows[-1])]
            old = self.cover_box.get(obj)
            if old is not None:
                box = [min(old[0], box[0]), max(old[1], box[1]), min(old[2], box[2]), max(old[3], box[3])]
            self.cover_box[obj] = box

    # ---------------------------------------------------------------- primitives
    def _floor(self, shader, obj: int) -> None:
        """The carpet: the plane z = 0, colored by ``shader(x, y)``."""
        d = self._dirs
        cz = self._cam.position[2]
        down = d[..., 2] < -1e-9
        t = np.where(down, -cz / np.where(down, d[..., 2], -1.0), np.inf)
        hit = down & (t > NEAR)
        cam = self._cam.position

        def surface(win):
            px = cam[0] + t[win] * d[..., 0][win]
            py = cam[1] + t[win] * d[..., 1][win]
            return shader(px, py), (0.0, 0.0, 1.0)
        self._write((0, d.shape[1], 0, d.shape[0]), t, hit, obj, surface, CARPET)

    def _box(self, b: Box) -> None:
        rot = np.asarray(b.rotation, dtype=np.float64)
        half = np.asarray(b.half, dtype=np.float64)
        center = np.asarray(b.center, dtype=np.float64)
        window = self._bbox(center + (_CUBE * half) @ rot.T)
        if window is None:
            return
        u0, u1, v0, v1 = window
        d = self._dirs[v0:v1, u0:u1] @ rot  # ray directions in the box frame
        o = (self._cam.position - center) @ rot  # camera in the box frame
        if (np.abs(o) < half).all():
            return  # the camera is inside this box: don't draw it
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
            t1 = (-half - o) * inv
            t2 = (half - o) * inv
        tnear = np.minimum(t1, t2)
        tfar = np.maximum(t1, t2)
        tnear = np.where(np.isnan(tnear), -np.inf, tnear)
        tfar = np.where(np.isnan(tfar), np.inf, tfar)
        enter = tnear.max(axis=-1)
        axis = tnear.argmax(axis=-1)
        leave = tfar.min(axis=-1)
        hit = (enter <= leave) & (enter > NEAR)

        def surface(win):
            ax = axis[win]
            rows = np.arange(ax.size)
            n_local = np.zeros((ax.size, 3))
            n_local[rows, ax] = -np.sign(d[win][rows, ax])
            return b.color, n_local @ rot.T
        self._write(window, enter, hit, b.obj, surface, emission=b.emission)

    def _polygon(self, p: Polygon) -> None:
        verts = np.asarray(p.vertices, dtype=np.float64)
        if not p.two_sided:
            n = np.cross(verts[1] - verts[0], verts[2] - verts[0])  # points away from the readable side
            if (self._cam.position - verts[0]) @ n >= 0:
                return
        window = self._bbox(verts)
        if window is None:
            return
        u0, u1, v0, v1 = window
        d = self._dirs[v0:v1, u0:u1]
        cam = self._cam.position
        n = np.cross(verts[1] - verts[0], verts[2] - verts[0])
        n /= np.linalg.norm(n)
        denom = d @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((verts[0] - cam) @ n) / denom
        hit = np.isfinite(t) & (t > NEAR)
        pts = cam + t[..., None] * d
        k = len(verts)
        for i in range(k):
            a, b = verts[i], verts[(i + 1) % k]
            hit &= (np.cross(b - a, pts - a) @ n) >= -1e-12
        facing = n if (cam - verts[0]) @ n >= 0 else -n  # two-sided: light the side we see
        texels = None
        if p.texture is not None and hit.any():
            e1, e3 = verts[1] - verts[0], verts[3] - verts[0]
            rel = pts[hit] - verts[0]
            th, tw = p.texture.shape[:2]
            col = (np.clip(rel @ e1 / (e1 @ e1), 0.0, 1.0 - 1e-9) * tw).astype(np.int64)
            row = (np.clip(rel @ e3 / (e3 @ e3), 0.0, 1.0 - 1e-9) * th).astype(np.int64)
            texels = np.zeros((*hit.shape, 3), dtype=np.float32)
            texels[hit] = p.texture[row, col, :3]
            if p.texture.shape[2] == 4:  # cut-out: drop the hits that land in holes
                keep = p.texture[row, col, 3] >= 0.5
                hit[hit] = keep

        def surface(win):
            return (p.color if texels is None else texels[win]), facing
        self._write(window, t, hit, p.obj, surface, emission=p.emission)

    def _cylinder(self, c: Cylinder) -> None:
        rot = np.asarray(c.rotation, dtype=np.float64)
        center = np.asarray(c.center, dtype=np.float64)
        half = np.array([c.radius, c.radius, c.half_length])
        window = self._bbox(center + (_CUBE * half) @ rot.T)
        if window is None:
            return
        u0, u1, v0, v1 = window
        d = self._dirs[v0:v1, u0:u1] @ rot  # the cylinder's frame: its axis is z
        o = (self._cam.position - center) @ rot
        r2, hl = c.radius ** 2, c.half_length
        # side: (ox + t dx)^2 + (oy + t dy)^2 = r^2
        a = d[..., 0] ** 2 + d[..., 1] ** 2
        b = o[0] * d[..., 0] + o[1] * d[..., 1]
        cc = o[0] ** 2 + o[1] ** 2 - r2
        disc = b * b - a * cc
        with np.errstate(divide="ignore", invalid="ignore"):
            t_side = (-b - np.sqrt(np.maximum(disc, 0.0))) / a
        z_side = o[2] + t_side * d[..., 2]
        side = (disc >= 0) & (a > 1e-12) & (np.abs(z_side) <= hl) & (t_side > NEAR)
        # caps: the planes z = +-hl, inside the circle
        with np.errstate(divide="ignore", invalid="ignore"):
            cap_z = np.where(o[2] > 0, hl, -hl)  # only the cap facing the camera can be hit first
            t_cap = (cap_z - o[2]) / d[..., 2]
        px, py = o[0] + t_cap * d[..., 0], o[1] + t_cap * d[..., 1]
        cap = np.isfinite(t_cap) & (t_cap > NEAR) & (px * px + py * py <= r2) & (abs(o[2]) > hl)
        t = np.where(side & (~cap | (t_side < t_cap)), t_side, np.where(cap, t_cap, np.inf))
        on_side = side & (t == t_side)
        hit = side | cap

        def surface(win):
            nl = np.zeros((int(win.sum()), 3))
            s = on_side[win]
            tw = t[win][s][:, None]
            p = o + tw * d[win][s]
            nl[s, 0], nl[s, 1] = p[:, 0] / c.radius, p[:, 1] / c.radius
            nl[~s, 2] = np.sign(o[2])
            return c.color, nl @ rot.T
        self._write(window, t, hit, c.obj, surface, emission=c.emission)

    def _spheres(self, spheres: list[Sphere]) -> None:
        if not spheres:
            return
        centers = np.array([s.center for s in spheres], dtype=np.float64)
        radii = np.array([s.radius for s in spheres], dtype=np.float64)
        pc = self._cam.to_camera(centers)
        # screen windows from each sphere's bounding cube (8 corners), all spheres at once
        u0, u1, v0, v1, ok = self._windows(pc[:, None, :] + _CUBE[None] * radii[:, None, None])
        todo = np.flatnonzero(ok & (pc[:, 2] + radii > NEAR))
        rays = self.rays
        rot = self._cam.rotation
        for i in todo:
            s = spheres[i]
            window = (u0[i], u1[i], v0[i], v1[i])
            dd = rays[v0[i]:v1[i], u0[i]:u1[i]]
            c = pc[i]
            a = (dd * dd).sum(axis=-1)
            b = dd @ c
            disc = b * b - a * ((c @ c) - radii[i] ** 2)
            t = (b - np.sqrt(np.maximum(disc, 0.0))) / a
            hit = (disc >= 0.0) & (t > NEAR)  # the near side, in front of the camera

            def surface(win, dd=dd, t=t, c=c, r=radii[i], color=s.color):
                p = t[win][:, None] * dd[win]  # camera-frame hit points
                return color, ((p - c) / r) @ rot.T
            self._write(window, t, hit, s.obj, surface, SHINY if s.shiny else MATTE)
