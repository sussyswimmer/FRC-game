"""The 2026 field in 3D, built from simple shapes.

Dimensions come from the official 2026 Field Dimension Drawings (FE-2026 rev B and the element
sheets GE-26000 to GE-26900), the Game Manual (section 5) and the Field Manual, in inches as the
drawings give them. Values marked APPROX were measured off a drawing at scale or estimated from
renders; colors are all estimates (randomize them). See docs/04-vision.md for the sources.

**The render follows the drawings, not the strategy simulator.** Four simulator constants
disagree with the drawings (TOWER, OUTPOST and DEPOT positions and the HUB hexagon's rotation;
see docs/04-vision.md). The simulator is left as it is, so its physics stays calibrated, and the
3D field here uses the official positions. FUEL and robots are drawn wherever the simulator puts
them.

Blue elements are built first. Most red elements are the point mirror of blue ones (rotated 180
degrees about the field center). The TRENCHes are the exception: the fixed TRENCHes are on the
scoring-table side (low y) for both alliances, so red TRENCHes mirror blue ones across the center
line only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .. import constants as C
from ..apriltags import TAGS
from ..constants import Alliance
from . import ids
from . import tags as T
from .labels import ObjectInfo
from .render import Box, Cylinder, Polygon

IN = 0.0254
L, W = C.FIELD_LENGTH, C.FIELD_WIDTH


def srgb(r: float, g: float, b: float) -> tuple[float, float, float]:
    """An sRGB color (0-255, as a color picker shows it) as linear albedo for the renderer."""
    return tuple(float((c / 255.0) ** 2.2) for c in (r, g, b))


# ---------------------------------------------------------------------------- layout (inches)
HUB_X, HUB_Y = 182.11 * IN, 158.84 * IN  # blue HUB center
HUB_HALF = 23.5 * IN  # 47 in square body
HUB_BODY_TOP = 49.75 * IN
HUB_CAP_TOP = 59.4 * IN  # APPROX
HUB_CAP_HALF_TOP = 18.0 * IN  # APPROX: the cap narrows to about 36 in square
HUB_FUNNEL_TOP = 72.0 * IN
HUB_FUNNEL_APOTHEM_TOP = 41.932 / 2 * IN  # outside, across the flats
HUB_FUNNEL_APOTHEM_BOTTOM = 28.7 / 2 * IN  # APPROX
HUB_NET_HALF_WIDTH = 58.41 / 2 * IN
HUB_NET_TOP = 120.36 * IN
HUB_NET_REACH = 10.26 * IN  # past the neutral face
BUMP_HALF_X = 22.2 * IN
BUMP_APEX = 6.51 * IN
BUMP_LIP = 0.61 * IN
BUMP_Y = ((62.34 * IN, 135.34 * IN), (182.34 * IN, 255.34 * IN))
TRENCH_OPENING = 50.34 * IN
TRENCH_COLUMN = 12.0 * IN
TRENCH_ARM_BOTTOM = 22.25 * IN
TRENCH_ARM_TOP = 26.3 * IN  # APPROX
TRENCH_COLUMN_TOP_HALF = 7.0 * IN  # APPROX: the column's trapezoid narrows to about 14 in on top
TAG_PANEL_Z = (29.75 * IN, 40.25 * IN)
TOWER_Y = 147.47 * IN  # blue TOWER center (tag 31), NOT constants.TOWER_CENTER_Y_BLUE
TOWER_WALL_Y = (122.85 * IN, 172.10 * IN)
TOWER_WALL_TOP = 78.25 * IN
TOWER_UPRIGHT_X = (40.0 * IN, 43.51 * IN)
TOWER_UPRIGHT_GAP = 32.25 * IN  # between the uprights' inner faces
TOWER_RUNG_X = 41.75 * IN
TOWER_RUNG_HALF = 23.5 * IN
DEPOT_Y = (213.85 * IN, 255.85 * IN)  # NOT constants.DEPOT_CENTER_Y_BLUE
DEPOT_DEPTH = 27.0 * IN
OUTPOST_Y = (0.0, 49.84 * IN)
OUTPOST_CENTER_Y = 26.22 * IN  # tag 29, NOT constants.OUTPOST_CENTER_Y_BLUE
CHUTE_Z = (28.13 * IN, 35.13 * IN)
CHUTE_HALF_WIDTH = 31.8 / 2 * IN
CORRAL_Z = (1.88 * IN, 8.88 * IN)
DS_Y = (49.84 * IN, 122.85 * IN, TOWER_WALL_Y[1], TOWER_WALL_Y[1] + (317.69 - 172.10) / 2 * IN, W)
DS_BASE_TOP = 36.8 * IN
WALL_TOP = 78.8 * IN
GUARDRAIL_TOP = 20.0 * IN
CENTER_LINE = (324.61 * IN, 326.61 * IN)
STARTING_LINE = (156.61 * IN, 158.61 * IN)  # blue ROBOT STARTING LINE, against the HUB and TRENCH faces

# 3 x 5 letters for the "REBUILT" graphics
_LETTERS = {"R": "110101110101101", "E": "111100110100111", "B": "110101110101110", "U": "101101101101111",
            "I": "111010010010111", "L": "100100100100111", "T": "111010010010010"}


@dataclass
class FieldAppearance:
    """Colors of the field (linear albedo). ``random`` varies them the way venues and cameras do."""

    carpet: tuple = srgb(100, 100, 104)
    carpet_fleck: float = 0.5  # how dark the carpet's flecks are (fraction of the base color)
    arena_floor: tuple = srgb(40, 40, 42)
    tape_white: tuple = srgb(215, 215, 210)
    alliance: dict = field(default_factory=lambda: {Alliance.BLUE: srgb(10, 75, 190), Alliance.RED: srgb(205, 20, 25)})
    aluminum: tuple = srgb(165, 168, 172)
    steel: tuple = srgb(150, 150, 155)
    black: tuple = srgb(30, 30, 32)
    teal: tuple = srgb(68, 129, 148)
    diffuser: tuple = srgb(235, 232, 232)
    tag_white: float = 0.80
    tag_black: float = 0.03
    net: tuple = srgb(25, 25, 28)
    trip_guard: tuple = srgb(230, 190, 20)
    seed: int = 0  # carpet mottling and the arena backdrop

    @classmethod
    def random(cls, rng: np.random.Generator) -> "FieldAppearance":
        def jitter(c, s=0.15):
            g = rng.uniform(1 - s, 1 + s)
            return tuple(float(np.clip(x * g * rng.uniform(0.95, 1.05), 0.0, 1.0)) for x in c)

        base = cls()
        return cls(
            carpet=jitter(srgb(*(rng.uniform(80, 125) + rng.uniform(-6, 6, 3))), 0.0),
            carpet_fleck=float(rng.uniform(0.35, 0.75)),
            arena_floor=jitter(base.arena_floor, 0.5),
            tape_white=jitter(base.tape_white, 0.1),
            alliance={a: jitter(c) for a, c in base.alliance.items()},
            aluminum=jitter(base.aluminum), steel=jitter(base.steel), black=jitter(base.black, 0.4),
            teal=jitter(base.teal), diffuser=jitter(base.diffuser, 0.08),
            tag_white=float(rng.uniform(0.65, 0.9)), tag_black=float(rng.uniform(0.01, 0.06)),
            net=jitter(base.net, 0.4), trip_guard=jitter(base.trip_guard),
            seed=int(rng.integers(2 ** 31)),
        )


# ---------------------------------------------------------------------------- building blocks
class Parts:
    """Shapes collected while building, plus the labeled objects among them."""

    def __init__(self) -> None:
        self.boxes: list[Box] = []
        self.polygons: list[Polygon] = []
        self.cylinders: list[Cylinder] = []

    def box(self, lo, hi, color, obj, emission=None) -> None:
        lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
        self.boxes.append(Box(tuple((lo + hi) / 2), tuple(np.abs(hi - lo) / 2), color, np.eye(3), obj, emission))

    def beam(self, p0, p1, thick, color, obj) -> None:
        """A square-section bar from p0 to p1."""
        p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
        x = p1 - p0
        length = float(np.linalg.norm(x))
        x /= length
        y = np.cross((0.0, 0.0, 1.0), x)
        y = y / np.linalg.norm(y) if np.linalg.norm(y) > 1e-9 else np.array([0.0, 1.0, 0.0])
        rot = np.column_stack([x, y, np.cross(x, y)])
        self.boxes.append(Box(tuple((p0 + p1) / 2), (length / 2, thick / 2, thick / 2), color, rot, obj))

    def poly(self, verts, color, obj, texture=None, emission=None, two_sided=True) -> None:
        self.polygons.append(Polygon(np.asarray(verts, dtype=np.float64), color, obj, texture, emission, two_sided))

    def cyl(self, center, radius, half_length, axis: str, color, obj) -> None:
        rot = {"x": np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]]), "y": np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]),
               "z": np.eye(3)}[axis]
        self.cylinders.append(Cylinder(tuple(center), radius, half_length, color, rot, obj))

    def extend(self, other: "Parts") -> None:
        self.boxes += other.boxes
        self.polygons += other.polygons
        self.cylinders += other.cylinders

    def mirrored(self, point: bool) -> "Parts":
        """This geometry rotated about the field center (``point``) or reflected across the center line."""
        m = np.diag([-1.0, -1.0 if point else 1.0, 1.0])
        off = np.array([L, W if point else 0.0, 0.0])
        out = Parts()
        out.boxes = [Box(tuple(m @ b.center + off), b.half, b.color, m @ b.rotation @ m, b.obj, b.emission) for b in self.boxes]
        out.cylinders = [Cylinder(tuple(m @ c.center + off), c.radius, c.half_length, c.color, m @ c.rotation @ m, c.obj,
                                  c.emission) for c in self.cylinders]
        # a point mirror keeps a polygon's handedness; a reflection reverses it (so its readable side)
        out.polygons = [Polygon(p.vertices @ m + off if point else (p.vertices @ m + off)[::-1], p.color, p.obj,
                                None if (p.texture is not None and not point) else p.texture, p.emission, p.two_sided)
                        for p in self.polygons]
        return out


def _mirror_point(p, point: bool = True) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.array([L - p[0], (W - p[1]) if point else p[1], p[2]])


def _letters_texture(text: str, color, ink, border=None) -> np.ndarray:
    """Blocky text on a colored panel, with an optional dashed border."""
    cols = 4 * len(text) + 5
    tex = np.empty((11, cols, 3), dtype=np.float32)
    tex[:] = color
    for k, ch in enumerate(text):
        bits = np.array([int(b) for b in _LETTERS[ch]], dtype=bool).reshape(5, 3)
        tex[3:8, 3 + 4 * k:6 + 4 * k][bits] = ink
    if border is not None:
        dash = (np.arange(cols) // 2) % 2 == 0
        tex[0, dash] = tex[-1, dash] = border
        tex[::2, 0] = tex[::2, -1] = border
    return tex


def _diamond_plate(color) -> np.ndarray:
    tex = np.empty((24, 96, 3), dtype=np.float32)
    tex[:] = np.asarray(color) * 0.8
    y, x = np.mgrid[0:24, 0:96]
    raised = ((x + 2 * y) % 8 < 2) & ((y // 2 + x // 4) % 2 == 0)
    tex[raised] = np.minimum(np.asarray(color) * 1.25, 1.0)
    return tex


def _net_texture(rng: np.random.Generator, color) -> np.ndarray:
    """A dark mesh: mostly holes, so the HUB shows through it like it does through the real net."""
    h, w = 96, 64
    tex = np.zeros((h, w, 4), dtype=np.float32)
    tex[..., :3] = color
    tex[::3, :, 3] = 1.0
    tex[:, ::3, 3] = 1.0
    tex[rng.random((h, w)) < 0.03, 3] = 0.0  # a few broken strands
    return tex


# ---------------------------------------------------------------------------- the elements
def _hub(a: Alliance, ap: FieldAppearance) -> Parts:
    """A HUB in blue coordinates (the neutral-zone side is +x). Its cap lights are added per frame."""
    p = Parts()
    obj = ids.HUB + a
    x0, x1 = HUB_X - HUB_HALF, HUB_X + HUB_HALF
    y0, y1 = HUB_Y - HUB_HALF, HUB_Y + HUB_HALF
    band = 37.5 * IN  # APPROX: bottom of the black band that carries the tags
    p.box((x0, y0, 0.0), (x1, y1, 0.10), ap.steel, obj)  # grey steel base
    p.box((x0 + 0.06, y0 + 0.02, 0.10), (x1 - 0.06, y1 - 0.02, band), srgb(18, 18, 20), obj)  # inside, seen through clear panels
    p.box((x0, y0, band), (x1, y1, HUB_BODY_TOP), ap.black, obj)  # black band
    for yy in (y0, y1 - 0.012):  # black side panels toward the guardrails, with the teal graphic
        p.box((x0, yy, 0.10), (x1, yy + 0.012, band), ap.black, obj)
    graphic = _letters_texture("REBUILT", ap.teal, srgb(235, 235, 235), border=ap.black)
    gw, gz0, gz1 = 0.95 * HUB_HALF, 0.22, 0.90
    for sign, yy in ((-1, y0 - 0.002), (1, y1 + 0.002)):
        r = -sign  # the reader's right when facing this side: +x on the low-y side, -x on the high-y side
        p.poly([(HUB_X - r * gw, yy, gz1), (HUB_X + r * gw, yy, gz1), (HUB_X + r * gw, yy, gz0), (HUB_X - r * gw, yy, gz0)],
               ap.teal, obj, graphic)
    for cx in (x0 + 0.03, x1 - 0.03):  # frame behind the clear panels: corner posts, rails, X braces
        for cy in (y0 + 0.045, y1 - 0.045):
            p.box((cx - 0.025, cy - 0.025, 0.10), (cx + 0.025, cy + 0.025, band), ap.aluminum, obj)
        p.box((cx - 0.02, y0 + 0.02, 0.48), (cx + 0.02, y1 - 0.02, 0.52), ap.aluminum, obj)
        p.beam((cx, y0 + 0.07, 0.12), (cx, y1 - 0.07, band - 0.02), 0.03, ap.aluminum, obj)
        p.beam((cx, y1 - 0.07, 0.12), (cx, y0 + 0.07, band - 0.02), 0.03, ap.aluminum, obj)
    for dy in (-0.42, -0.14, 0.14, 0.42):  # FUEL exits at the base of the neutral face (APPROX sizes)
        p.poly([(x1 + 0.002, HUB_Y + dy - 0.1, 0.19), (x1 + 0.002, HUB_Y + dy + 0.1, 0.19),
                (x1 + 0.002, HUB_Y + dy + 0.1, 0.02), (x1 + 0.002, HUB_Y + dy - 0.1, 0.02)], srgb(8, 8, 8), obj)
    # roof panel in the alliance color, then the clear funnel's frame
    h = HUB_CAP_HALF_TOP
    p.box((HUB_X - h, HUB_Y - h, HUB_CAP_TOP - 0.01), (HUB_X + h, HUB_Y + h, HUB_CAP_TOP), ap.alliance[a], obj)
    rb = HUB_FUNNEL_APOTHEM_BOTTOM / math.cos(math.pi / 6)
    rt = HUB_FUNNEL_APOTHEM_TOP / math.cos(math.pi / 6)
    angles = [k * math.pi / 3 for k in range(6)]  # corners toward +-x: the flats face the guardrails
    bottom = [(HUB_X + rb * math.cos(t), HUB_Y + rb * math.sin(t), HUB_CAP_TOP) for t in angles]
    top = [(HUB_X + rt * math.cos(t), HUB_Y + rt * math.sin(t), HUB_FUNNEL_TOP) for t in angles]
    for k in range(6):
        p.beam(top[k], top[(k + 1) % 6], 0.03, srgb(20, 20, 22), obj)
        p.beam(bottom[k], top[k], 0.02, ap.aluminum, obj)
    # the net on the neutral side: two sloped poles, a top bar and the mesh between them
    nx0, nx1 = x1 + 0.01, x1 + HUB_NET_REACH
    ny0, ny1 = HUB_Y - HUB_NET_HALF_WIDTH, HUB_Y + HUB_NET_HALF_WIDTH
    for ny in (ny0, ny1):
        p.beam((nx0, ny, HUB_BODY_TOP), (nx1, ny, HUB_NET_TOP), 0.04, ap.steel, obj)
    p.beam((nx1, ny0 - 0.02, HUB_NET_TOP), (nx1, ny1 + 0.02, HUB_NET_TOP), 0.045, ap.steel, obj)
    net_rng = np.random.default_rng(ap.seed + 17 + a)
    p.poly([(nx1, ny0, HUB_NET_TOP), (nx1, ny1, HUB_NET_TOP), (nx0, ny1, HUB_BODY_TOP + 0.05), (nx0, ny0, HUB_BODY_TOP + 0.05)],
           ap.net, obj, _net_texture(net_rng, ap.net))
    return p


def hub_cap(a: Alliance, ap: FieldAppearance, glow: tuple[float, float, float] | None) -> Parts:
    """The HUB's sloped cap: white diffusers, lit from inside in the alliance color while the HUB counts."""
    p = Parts()
    obj = ids.HUB + a
    b, t, z0, z1 = HUB_HALF, HUB_CAP_HALF_TOP, HUB_BODY_TOP, HUB_CAP_TOP
    cx, cy = HUB_X, HUB_Y
    faces = (
        [(cx + b, cy - b, z0), (cx + b, cy + b, z0), (cx + t, cy + t, z1), (cx + t, cy - t, z1)],
        [(cx - b, cy + b, z0), (cx - b, cy - b, z0), (cx - t, cy - t, z1), (cx - t, cy + t, z1)],
        [(cx + b, cy + b, z0), (cx - b, cy + b, z0), (cx - t, cy + t, z1), (cx + t, cy + t, z1)],
        [(cx - b, cy - b, z0), (cx + b, cy - b, z0), (cx + t, cy - t, z1), (cx - t, cy - t, z1)],
    )
    # lit from inside, the translucent panels show the LEDs' color more than the arena light they reflect
    albedo = ap.diffuser if glow is None else tuple(0.35 * c for c in ap.diffuser)
    for f in faces:
        p.poly(f, albedo, obj, emission=glow)
    return p if a == Alliance.BLUE else p.mirrored(point=True)


def _bumps(a: Alliance, ap: FieldAppearance) -> tuple[Parts, dict[int, ObjectInfo]]:
    p = Parts()
    infos = {}
    x0, x1, xc = HUB_X - BUMP_HALF_X, HUB_X + BUMP_HALF_X, HUB_X
    # alliance-colored HDPE with rows of bolt heads: one texel is about 1.2 x 1.8 cm, a bolt head one texel
    tex = np.empty((160, 32, 3), dtype=np.float32)
    tex[:] = ap.alliance[a]
    for col in (3, 16, 29):  # near the ridge, midway, near the edge
        tex[4::12, col] = srgb(25, 25, 25)
    dark = tuple(0.35 * c for c in ap.steel)
    for side, (ya, yb) in enumerate(BUMP_Y):
        # side 0 is the low-y BUMP; red is built in blue coordinates and then turned around the field
        # center, which swaps its sides
        real_side = side if a == Alliance.BLUE else 1 - side
        obj = ids.BUMP + 2 * a + real_side
        # ramps: rows of the texture run along y, so vertices 0, 1, 3 = (y low, apex), (y low, edge), (y high, apex)
        for xe in (x0, x1):
            p.poly([(xc, ya, BUMP_APEX), (xe, ya, BUMP_LIP), (xe, yb, BUMP_LIP), (xc, yb, BUMP_APEX)], ap.alliance[a], obj, tex)
        for yy in (ya, yb):  # end caps and the lips along both edges
            p.poly([(x0, yy, 0.0), (x0, yy, BUMP_LIP), (xc, yy, BUMP_APEX), (x1, yy, BUMP_LIP), (x1, yy, 0.0)], dark, obj)
        for xe in (x0, x1):
            p.poly([(xe, ya, 0.0), (xe, yb, 0.0), (xe, yb, BUMP_LIP), (xe, ya, BUMP_LIP)], dark, obj)
        corners = np.array([(x, y, z) for x in (x0, x1) for y in (ya, yb) for z in (0.0, BUMP_APEX)])
        infos[obj] = ObjectInfo("bump", {"alliance": a.name.lower(), "side": ("low_y", "high_y")[real_side]}, corners)
    return p, infos


def bump_height(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Height of the BUMP surface under field points (0 off the BUMPs), for resting FUEL on them."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    out = np.zeros(np.broadcast(x, y).shape)
    for hx, flip in ((HUB_X, False), (L - HUB_X, True)):
        dx = np.abs(x - hx)
        for ya, yb in BUMP_Y:
            if flip:
                ya, yb = W - yb, W - ya
            on = (dx <= BUMP_HALF_X) & (y >= ya) & (y <= yb)
            z = BUMP_APEX - (BUMP_APEX - BUMP_LIP) * dx / BUMP_HALF_X
            out = np.where(on, z, out)
    return out


def _trench(a: Alliance, side: int, ap: FieldAppearance) -> Parts:
    """A TRENCH in blue coordinates. side 0 is the fixed one at low y (scoring-table side), 1 the
    hinged one at high y."""
    p = Parts()
    obj = ids.TRENCH + 2 * a + side
    col = ap.alliance[a]
    x0, x1, xc = HUB_X - HUB_HALF, HUB_X + HUB_HALF, HUB_X

    def y(v):  # distances from the near guardrail -> field y
        return v if side == 0 else W - v

    # column: a trapezoid in side view, alliance-colored skirts
    ya, yb = sorted((y(TRENCH_OPENING), y(TRENCH_OPENING + TRENCH_COLUMN)))
    th = TRENCH_COLUMN_TOP_HALF
    zt = TRENCH_ARM_TOP
    for yy in (ya, yb):
        p.poly([(x0, yy, 0.0), (x1, yy, 0.0), (xc + th, yy, zt), (xc - th, yy, zt)], col, obj)
    p.poly([(x0, ya, 0.0), (x0, yb, 0.0), (xc - th, yb, zt), (xc - th, ya, zt)], col, obj)
    p.poly([(x1, ya, 0.0), (x1, yb, 0.0), (xc + th, yb, zt), (xc + th, ya, zt)], col, obj)
    p.poly([(xc - th, ya, zt), (xc + th, ya, zt), (xc + th, yb, zt), (xc - th, yb, zt)], col, obj)
    # arm across the opening; the fixed arm overhangs the guardrail and rests on a leg
    half = (6.0 if side == 0 else 4.5) / 2 * IN
    reach = -8.0 * IN if side == 0 else -0.02
    lo, hi = sorted((y(reach), y(TRENCH_OPENING + 0.02)))
    p.box((xc - half, lo, TRENCH_ARM_BOTTOM), (xc + half, hi, TRENCH_ARM_TOP), col, obj)
    if side == 0:
        ly = y(reach + 0.03)
        p.box((xc - 0.03, ly - 0.03, 0.0), (xc + 0.03, ly + 0.03, TRENCH_ARM_BOTTOM), ap.steel, obj)
    else:
        hy = y(TRENCH_OPENING + 0.05)
        p.box((xc - 0.06, hy - 0.05, zt), (xc + 0.06, hy + 0.05, zt + 0.08), srgb(60, 60, 62), obj)  # hinge block
    # grey bracket on top of the arm carrying two tags back to back
    ty = y(25.37 * IN)
    bx = 1.475 * IN - 0.002
    p.box((xc - bx, ty - 0.03, TRENCH_ARM_TOP), (xc + bx, ty + 0.03, TAG_PANEL_Z[0]), ap.steel, obj)
    p.box((xc - bx, ty - T.PANEL / 2, TAG_PANEL_Z[0]), (xc + bx, ty + T.PANEL / 2, TAG_PANEL_Z[1]), ap.steel, obj)
    return p


def _tower(a: Alliance, ap: FieldAppearance) -> Parts:
    p = Parts()
    obj = ids.TOWER + a
    col = ap.alliance[a]
    yc = TOWER_Y
    p.box((0.0, yc - 19.5 * IN, 0.0), (45.18 * IN, yc + 19.5 * IN, 0.0064), srgb(22, 22, 24), obj)  # base plate
    gap, t = TOWER_UPRIGHT_GAP / 2, 1.5 * IN
    ux0, ux1 = TOWER_UPRIGHT_X
    for s in (-1, 1):
        yi = yc + s * gap
        lo, hi = sorted((yi, yi + s * t))
        p.box((ux0, lo, 0.0064), (ux1, hi, 72.125 * IN), col, obj)
        ym = (lo + hi) / 2
        # grey supports back to the wall: a horizontal tube, a diagonal brace and a gusset (APPROX)
        p.box((0.0, ym - 0.022, 35.125 * IN), (ux0, ym + 0.022, 36.875 * IN), ap.steel, obj)
        p.beam((0.0, ym, 28.4 * IN), (ux0 * 0.55, ym, 35.2 * IN), 0.035, ap.steel, obj)
        p.beam((ux0 * 0.7, ym, 36.8 * IN), (ux0, ym, 43.375 * IN), 0.03, ap.steel, obj)
    for z in C.RUNG_HEIGHTS:
        p.cyl((TOWER_RUNG_X, yc, z), 1.66 / 2 * IN, TOWER_RUNG_HALF, "y", col, obj)
    return p


def _depot(a: Alliance, ap: FieldAppearance) -> Parts:
    p = Parts()
    obj = ids.DEPOT + a
    y0, y1 = DEPOT_Y
    bar, z = 3.0 * IN, 1.125 * IN
    col = ap.alliance[a]
    p.box((DEPOT_DEPTH - bar, y0, 0.0), (DEPOT_DEPTH, y1, z), col, obj)
    p.box((0.0, y0, 0.0), (DEPOT_DEPTH - bar, y0 + bar, z), col, obj)
    p.box((0.0, y1 - bar, 0.0), (DEPOT_DEPTH - bar, y1, z), col, obj)
    return p


def _outpost(a: Alliance, ap: FieldAppearance) -> Parts:
    """The OUTPOST's clear wall is invisible here; its aluminum frame, the black CHUTE door and the
    CHUTE's sloped floor behind it are drawn."""
    p = Parts()
    obj = ids.OUTPOST + a
    y0, y1 = OUTPOST_Y
    yc, hw = OUTPOST_CENTER_Y, CHUTE_HALF_WIDTH
    al = ap.aluminum
    top = 78.0 * IN
    for ya in (y0, y1 - 0.04):
        p.box((-0.045, ya, 0.0), (0.0, ya + 0.04, top), al, obj)
    p.box((-0.045, y0, top - 0.04), (0.0, y1, top), al, obj)
    for z in (CORRAL_Z[1], CHUTE_Z[0] - 0.035, CHUTE_Z[1]):
        p.box((-0.04, y0 + 0.04, z), (0.0, y1 - 0.04, z + 0.035), al, obj)
    p.cyl((-0.02, yc, sum(CORRAL_Z) / 2), 1.66 / 2 * IN, (CORRAL_Z[1] - CORRAL_Z[0]) / 2, "z", al, obj)
    # CHUTE door (black HDPE, drawn closed) and the chute's floor sloping up and away at 15 degrees
    p.box((-0.03, yc - hw, CHUTE_Z[0]), (-0.015, yc + hw, CHUTE_Z[0] + 0.6 * (CHUTE_Z[1] - CHUTE_Z[0])), srgb(15, 15, 15), obj)
    depth = 1.4
    rise = depth * math.tan(math.radians(15.0))
    p.poly([(-0.03, yc - hw, CHUTE_Z[0]), (-0.03, yc + hw, CHUTE_Z[0]), (-0.03 - depth, yc + hw, CHUTE_Z[0] + rise),
            (-0.03 - depth, yc - hw, CHUTE_Z[0] + rise)], srgb(60, 60, 62), obj)
    return p


def chute_fuel_positions(a: Alliance, count: int) -> np.ndarray:
    """(count, 3) centers of FUEL waiting in an OUTPOST CHUTE, 5 across, nearest the door first."""
    r = C.FUEL_RADIUS
    slope = math.tan(math.radians(15.0))
    out = []
    for k in range(count):
        row, col = divmod(k, 5)
        x = -0.03 - r - row * 2.02 * r
        out.append((x, OUTPOST_CENTER_Y + (col - 2) * 2.05 * r, CHUTE_Z[0] + r + 0.01 - x * slope))
    pts = np.array(out).reshape(-1, 3)
    return pts if a == Alliance.BLUE else np.array([_mirror_point(q) for q in pts]).reshape(-1, 3)


def _alliance_wall(a: Alliance, ap: FieldAppearance) -> Parts:
    """Driver-station walls and the TOWER wall behind the TOWER (blue coordinates, wall face x = 0).
    Team signs are added per match (they show the robots' numbers)."""
    p = Parts()
    obj = ids.ALLIANCE_WALL + a
    plate = _diamond_plate(ap.aluminum)
    for ya, yb in ((DS_Y[0], DS_Y[1]), (DS_Y[2], DS_Y[3]), (DS_Y[3], DS_Y[4])):  # DS3, DS2, DS1
        p.box((-0.08, ya, 0.0), (0.0, yb, DS_BASE_TOP), ap.aluminum, obj)
        p.poly([(0.002, ya, DS_BASE_TOP), (0.002, yb, DS_BASE_TOP), (0.002, yb, 0.0), (0.002, ya, 0.0)], ap.aluminum, obj, plate)
        p.box((-0.05, ya, DS_BASE_TOP), (-0.02, yb, 42.9 * IN), srgb(225, 225, 225), obj)  # sponsor panel
        p.box((-0.06, ya, WALL_TOP - 0.04), (0.0, yb, WALL_TOP), ap.aluminum, obj)  # top rail
        for yy in (ya, yb - 0.05):
            p.box((-0.06, yy, DS_BASE_TOP), (0.0, yy + 0.05, WALL_TOP), ap.aluminum, obj)
    ya, yb = TOWER_WALL_Y
    p.box((-0.08, ya, 0.0), (0.0, yb, 36.0 * IN), ap.black, obj)
    p.box((-0.08, ya, 36.0 * IN), (-0.004, yb, TOWER_WALL_TOP - 0.04), ap.black, obj)
    p.box((-0.08, ya, TOWER_WALL_TOP - 0.04), (0.0, yb, TOWER_WALL_TOP), ap.aluminum, obj)
    g = _letters_texture("REBUILT", ap.teal, srgb(235, 235, 235), border=srgb(235, 235, 235))
    m = 0.1
    # facing +x, the reader's right is +y
    p.poly([(0.0, ya + m, TOWER_WALL_TOP - 0.12), (0.0, yb - m, TOWER_WALL_TOP - 0.12),
            (0.0, yb - m, 40.0 * IN), (0.0, ya + m, 40.0 * IN)], ap.teal, obj, g)
    return p


def team_signs(a: Alliance, numbers: list[str], ap: FieldAppearance) -> Parts:
    """Team number signs on top of each driver station, with the team's LED stack."""
    from .robot_model import number_texture

    p = Parts()
    obj = ids.ALLIANCE_WALL + a
    spans = ((DS_Y[3], DS_Y[4]), (DS_Y[2], DS_Y[3]), (DS_Y[0], DS_Y[1]))  # DS1, DS2, DS3
    for number, (ya, yb) in zip(numbers, spans):
        yc = (ya + yb) / 2
        tex = number_texture(number, srgb(10, 10, 10), ap.alliance[a])
        hh = 0.11
        hw = min(hh * tex.shape[1] / tex.shape[0], 0.45)
        z0 = WALL_TOP + 0.01
        p.box((-0.12, yc - hw, z0), (-0.01, yc + hw, z0 + 2 * hh), srgb(10, 10, 10), obj)
        p.poly([(-0.008, yc - hw, z0 + 2 * hh), (-0.008, yc + hw, z0 + 2 * hh), (-0.008, yc + hw, z0), (-0.008, yc - hw, z0)],
               srgb(10, 10, 10), obj, tex)
        for k, color in enumerate((ap.alliance[a], ap.alliance[a], srgb(255, 150, 0))):
            p.cyl((-0.06, yc, z0 + 2 * hh + 0.035 + 0.06 * k), 0.025, 0.028, "z", color, obj)
    return p if a == Alliance.BLUE else p.mirrored(point=True)


def _guardrails(ap: FieldAppearance) -> Parts:
    p = Parts()
    obj = ids.GUARDRAIL
    t = 1.5 * IN
    n = 12
    for side in (0, 1):
        ya, yb = (-t, 0.0) if side == 0 else (W, W + t)
        p.box((0.0, ya, 0.0), (L, yb, t), ap.aluminum, obj)
        p.box((0.0, ya, GUARDRAIL_TOP - t), (L, yb, GUARDRAIL_TOP), ap.aluminum, obj)
        out = -1 if side == 0 else 1
        for k in range(1, n):
            x = L * k / n
            p.box((x - t / 2, ya, t), (x + t / 2, yb, GUARDRAIL_TOP - t), ap.aluminum, obj)
            yo = (ya if side == 0 else yb)
            lo, hi = sorted((yo, yo + out * 0.35))
            p.box((x - 0.05, lo, 0.0), (x + 0.05, hi, 0.035), ap.trip_guard, obj)  # yellow trip guards outside
    return p


def _tag_panels(ap: FieldAppearance) -> tuple[Parts, dict[int, ObjectInfo]]:
    """Every AprilTag: its white backing panel (part of the structure it is on) and the printed tag."""
    p = Parts()
    infos = {}
    for tid, x, y, z, _ in TAGS:
        host = _tag_host(tid, x, y)
        p.poly(T.square(tid, T.PANEL, 0.001), (ap.tag_white,) * 3, host)
        p.poly(T.square(tid, T.TAG_OUTER, 0.002), (ap.tag_white,) * 3, ids.TAG + tid,
               T.tag_texture(tid, ap.tag_white, ap.tag_black), two_sided=False)
        infos[ids.TAG + tid] = ObjectInfo("apriltag", {"tag_id": tid}, T.square(tid, T.TAG_OUTER), T.tag_corners(tid))
    return p, infos


def _extent_points(parts: Parts, obj: int) -> np.ndarray:
    """The 8 corners of the axis-aligned box around every shape drawn with id ``obj``."""
    cube = np.array([(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=np.float64)
    pts = [np.asarray(b.center) + (cube * b.half) @ np.asarray(b.rotation).T for b in parts.boxes if b.obj == obj]
    pts += [np.asarray(p.vertices) for p in parts.polygons if p.obj == obj]
    pts += [np.asarray(c.center) + (cube * (c.radius, c.radius, c.half_length)) @ np.asarray(c.rotation).T
            for c in parts.cylinders if c.obj == obj]
    allp = np.concatenate(pts)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    return np.array([(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _tag_host(tid: int, x: float, y: float) -> int:
    red = x > L / 2
    a = int(red)
    if z_band(tid) == "trench":
        low = y < W / 2
        return ids.TRENCH + 2 * a + (0 if low else 1)
    if z_band(tid) == "hub":
        return ids.HUB + a
    near_outpost = (y < 1.5) if not red else (y > W - 1.5)
    return (ids.OUTPOST if near_outpost else ids.TOWER) + a


def z_band(tid: int) -> str:
    z = {t[0]: t[3] for t in TAGS}[tid]
    return "hub" if z > 1.0 else ("trench" if z > 0.8 else "wall")


# ---------------------------------------------------------------------------- the whole field
class FieldModel:
    """All the static field geometry for one appearance, plus the labeled field objects."""

    def __init__(self, appearance: FieldAppearance | None = None) -> None:
        ap = self.appearance = appearance or FieldAppearance()
        self.parts = Parts()
        self.objects: dict[int, ObjectInfo] = {}
        for a in Alliance:
            blue = Parts()
            for build in (_hub, _tower, _depot, _outpost, _alliance_wall):
                blue.extend(build(a, ap))
            bumps, infos = _bumps(a, ap)
            blue.extend(bumps)
            self.parts.extend(blue if a == Alliance.BLUE else blue.mirrored(point=True))
            trenches = Parts()
            for side in (0, 1):
                trenches.extend(_trench(a, side, ap))
            self.parts.extend(trenches if a == Alliance.BLUE else trenches.mirrored(point=False))
            for obj, info in infos.items():
                if a == Alliance.RED:
                    info.points = np.array([_mirror_point(q) for q in info.points])
                self.objects[obj] = info
            self._structure_infos(a)
        self.parts.extend(_guardrails(ap))
        tag_parts, tag_infos = _tag_panels(ap)
        self.parts.extend(tag_parts)
        self.objects.update(tag_infos)
        for obj, info in self.objects.items():
            if obj >= ids.HUB:  # bound each structure by everything drawn for it (net, rungs, chute...)
                info.points = _extent_points(self.parts, obj)
        for tid, x, y, _, _ in TAGS:  # a structure's tags are part of it (see labels.annotate)
            host = self.objects[_tag_host(tid, x, y)]
            host.children = host.children + (ids.TAG + tid,)
        self.multipart = {obj for obj in self.objects if obj >= ids.HUB} | {ids.ALLIANCE_WALL, ids.ALLIANCE_WALL + 1,
                                                                            ids.GUARDRAIL}
        self._backdrop = _backdrop_texture(np.random.default_rng(ap.seed))

    def _structure_infos(self, a: Alliance) -> None:
        red = a == Alliance.RED

        def box_points(lo, hi, point=True):
            pts = np.array([(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            return np.array([_mirror_point(q, point) for q in pts]) if red else pts

        name = a.name.lower()
        h = HUB_HALF
        self.objects[ids.HUB + a] = ObjectInfo("hub", {"alliance": name}, box_points(
            (HUB_X - h, HUB_Y - h, 0.0), (HUB_X + h + HUB_NET_REACH, HUB_Y + h, HUB_NET_TOP)))
        self.objects[ids.TOWER + a] = ObjectInfo("tower", {"alliance": name}, box_points(
            (0.0, TOWER_Y - 0.45, 0.0), (TOWER_UPRIGHT_X[1], TOWER_Y + 0.45, 72.125 * IN)))
        self.objects[ids.DEPOT + a] = ObjectInfo("depot", {"alliance": name}, box_points(
            (0.0, DEPOT_Y[0], 0.0), (DEPOT_DEPTH, DEPOT_Y[1], 1.125 * IN)))
        self.objects[ids.OUTPOST + a] = ObjectInfo("outpost", {"alliance": name}, box_points(
            (-0.05, OUTPOST_Y[0], 0.0), (0.0, OUTPOST_Y[1], 78.0 * IN)))
        for side in (0, 1):
            y0, y1 = (-8.0 * IN, TRENCH_OPENING + TRENCH_COLUMN) if side == 0 else (W - TRENCH_OPENING - TRENCH_COLUMN, W)
            self.objects[ids.TRENCH + 2 * a + side] = ObjectInfo("trench", {"alliance": name, "side": ("low_y", "high_y")[side]},
                                                                 box_points((HUB_X - HUB_HALF, y0, 0.0),
                                                                            (HUB_X + HUB_HALF, y1, TAG_PANEL_Z[1]), point=False))

    # ------------------------------------------------------------------ carpet and backdrop
    def floor(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Carpet albedo at field points: gray with dark flecks, tape lines, arena floor beyond."""
        ap = self.appearance
        n = x.size
        out = np.empty((n, 3), dtype=np.float32)
        ix = np.floor(x / 0.006).astype(np.int64)
        iy = np.floor(y / 0.006).astype(np.int64)
        hsh = _hash(ix + ap.seed % 997, iy)
        tone = 0.93 + 0.14 * _hash(iy + 7, ix)
        tone = np.where(hsh > 0.82, ap.carpet_fleck, tone)
        mottle = 1.0 + 0.04 * np.sin(x * 1.3 + ap.seed % 13) * np.sin(y * 1.7 + ap.seed % 7)
        out[:] = np.asarray(ap.carpet, dtype=np.float32)
        out *= (tone * mottle)[:, None]
        tape = [(CENTER_LINE, ap.tape_white), (STARTING_LINE, ap.alliance[Alliance.BLUE]),
                ((L - STARTING_LINE[1], L - STARTING_LINE[0]), ap.alliance[Alliance.RED])]
        on_field = (y >= 0) & (y <= W)
        for (x0, x1), color in tape:
            sel = (x >= x0) & (x <= x1) & on_field
            out[sel] = color
        outside = (x < -2.5) | (x > L + 2.5) | (y < -1.5) | (y > W + 1.5)
        out[outside] = ap.arena_floor
        return out

    def background(self, dirs: np.ndarray) -> np.ndarray:
        """The arena beyond the field: lit ceiling, stands and banners (a panorama at infinity)."""
        d = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        az = (np.arctan2(d[:, 1], d[:, 0]) / (2 * math.pi)) % 1.0
        el = np.arcsin(np.clip(d[:, 2], -1.0, 1.0))
        tex = self._backdrop
        h, w = tex.shape[:2]
        ceiling = int(h * _CEILING_ROWS)
        # the ceiling spans elevations from straight up to 0.35 rad, the stands 0.35 rad down to -0.3 rad
        frac = np.where(el > 0.35, (math.pi / 2 - el) / (math.pi / 2 - 0.35) * ceiling,
                        ceiling + (0.35 - el) / 0.65 * (h - ceiling))
        row = np.clip(frac.astype(np.int64), 0, h - 1)
        col = (az * w).astype(np.int64) % w
        return tex[row, col]


_CEILING_ROWS = 0.35


def _hash(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    h = (ix * 374761393 + iy * 668265263) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h ^= h >> 16
    return (h & 0xFFFFFF).astype(np.float32) / float(0x1000000)


def _backdrop_texture(rng: np.random.Generator) -> np.ndarray:
    """A 1024 x 192 panorama: a dark ceiling with bright lamps on top; below it the stands, with
    muted, blurry crowds, aisles and a few banners; a dark arena wall at the bottom."""
    h, w = 192, 1024
    tex = np.empty((h, w, 3), dtype=np.float32)
    tex[:] = srgb(*(rng.uniform(8, 35) + rng.uniform(-4, 4, 3)))
    ceiling = int(h * _CEILING_ROWS)
    for _ in range(int(rng.integers(10, 30))):  # lamps: brighter than white, so exposure matters
        r, c = int(rng.integers(0, ceiling - 4)), int(rng.integers(0, w - 8))
        tex[r:r + int(rng.integers(1, 4)), c:c + int(rng.integers(2, 9))] = rng.uniform(2.0, 8.0)
    stands = tex[ceiling:]
    n = h - ceiling
    seat = np.array(srgb(*(rng.uniform(25, 80) + rng.uniform(-10, 10, 3))), dtype=np.float32)
    stands[:] = seat
    # people: small blobs of muted clothing colors, denser in some sections than others
    density = np.repeat(rng.uniform(0.15, 0.7, w // 64 + 1), 64)[:w]
    for row in range(0, n - 4, 3):
        cols = np.flatnonzero(rng.random(w) < density)
        grey = rng.uniform(20, 200, (cols.size, 1))
        tint = rng.uniform(-40, 40, (cols.size, 3)) * rng.uniform(0.2, 1.0)
        colors = np.clip(grey + tint, 0, 255)
        for dr in range(3):
            stands[row + dr, cols] = (colors / 255.0) ** 2.2 * rng.uniform(0.7, 1.0)
    for c in rng.integers(0, w, int(rng.integers(4, 12))):  # aisles
        stands[:, c:c + int(rng.integers(2, 6))] = seat * 0.6
    for _ in range(int(rng.integers(3, 10))):  # banners
        r, c = int(rng.integers(ceiling, h - 30)), int(rng.integers(0, w - 60))
        tex[r:r + int(rng.integers(6, 16)), c:c + int(rng.integers(20, 80))] = srgb(*rng.uniform(0, 255, 3))
    tex[h - int(h * 0.12):] = srgb(*(rng.uniform(10, 40, 3)))  # arena wall behind the field
    return tex
