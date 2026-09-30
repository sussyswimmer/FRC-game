"""What a robot looks like in 3D: bumpers in its alliance color with a team number, a chassis,
and a few boxes for its mechanisms.

Real robots all look different, and a detector trained on one look learns that look instead of
"robot". So each robot gets a random ``RobotStyle`` (colors, mechanism layout, team number),
drawn once per match so a robot keeps its look from frame to frame. The simulator only knows a
robot's footprint (``spec.length`` x ``spec.width``) and height, so those are exact; the rest is
made up. Bumper dimensions are APPROX: about 5 in tall, riding just above the carpet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import constants as C
from ..constants import Alliance
from ..robot import ClimbState, Robot
from .render import Box, Polygon

BUMPER_BOTTOM = 0.025  # m above the carpet (APPROX)
BUMPER_HEIGHT = 0.127  # m, 5 in (APPROX)
CHASSIS_BOTTOM = 0.03

# bumper fabric (linear albedo), before per-robot jitter
BUMPER_COLORS = {Alliance.BLUE: (0.03, 0.10, 0.55), Alliance.RED: (0.62, 0.04, 0.04)}
_BODY_COLORS = (
    (0.55, 0.56, 0.58),  # bare aluminum
    (0.04, 0.04, 0.045),  # black powder coat
    (0.30, 0.31, 0.33),  # gray
    (0.60, 0.30, 0.03),  # orange
    (0.05, 0.30, 0.08),  # green
    (0.45, 0.02, 0.35),  # purple
    (0.70, 0.62, 0.03),  # gold
    (0.75, 0.75, 0.78),  # white-ish polycarbonate
)

# 3 x 5 bitmap digits for bumper numbers
_DIGITS = {
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111",
    "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001010010010",
    "8": "111101111101111", "9": "111101111001111",
}


@dataclass(frozen=True)
class Part:
    """One box of a robot in its own frame (x forward, y left, z up, origin on the carpet under its
    center). ``turret`` parts turn with the robot's turret."""

    center: tuple[float, float, float]
    half: tuple[float, float, float]
    color: tuple[float, float, float]
    turret: bool = False


@dataclass(frozen=True)
class RobotStyle:
    number: str
    bumper: tuple[float, float, float]
    parts: tuple[Part, ...]


def number_texture(number: str, background, ink=(0.85, 0.85, 0.85)) -> np.ndarray:
    """A bumper number plate: white digits on the bumper fabric, one cell of margin all round."""
    cols = 4 * len(number) + 1
    tex = np.empty((7, cols, 3), dtype=np.float32)
    tex[:] = background
    for k, ch in enumerate(number):
        bits = np.array([int(b) for b in _DIGITS[ch]], dtype=bool).reshape(5, 3)
        block = tex[1:6, 1 + 4 * k:4 + 4 * k]
        block[bits] = ink
    return tex


def random_style(robot: Robot, rng: np.random.Generator) -> RobotStyle:
    """A plausible-looking robot that fills the simulated footprint and height."""
    s = robot.spec
    bd = C.BUMPER_DEPTH
    fl, fw = s.length / 2 - bd, s.width / 2 - bd  # frame half sizes inside the bumpers
    height = max(s.height, 0.30)
    bumper = tuple(float(np.clip(c * rng.uniform(0.8, 1.2) + rng.uniform(-0.02, 0.02), 0.0, 1.0))
                   for c in BUMPER_COLORS[robot.alliance])
    main, accent = (_BODY_COLORS[i] for i in rng.choice(len(_BODY_COLORS), 2, replace=False))
    parts = []
    chassis_top = CHASSIS_BOTTOM + rng.uniform(0.10, 0.18)
    parts.append(Part((0.0, 0.0, (CHASSIS_BOTTOM + chassis_top) / 2), (fl, fw, (chassis_top - CHASSIS_BOTTOM) / 2), main))
    # a hopper over the back of the frame, up to most of the robot's height
    hop_len = rng.uniform(0.45, 0.8) * 2 * fl
    hop_top = chassis_top + (height - chassis_top) * rng.uniform(0.55, 0.85)
    hop_w = fw * rng.uniform(0.75, 1.0)
    parts.append(Part((-fl + hop_len / 2, 0.0, (chassis_top + hop_top) / 2),
                      (hop_len / 2, hop_w, (hop_top - chassis_top) / 2), accent if rng.random() < 0.5 else main))
    # the shooter on top, reaching the full height; it turns with the turret on robots that have one
    sh_len = rng.uniform(0.18, 0.32)
    sh_w = rng.uniform(0.15, 0.3)
    sx = float(np.clip(-fl + hop_len * rng.uniform(0.3, 0.8), -fl + sh_len, fl - sh_len))
    parts.append(Part((sx, 0.0, (hop_top + height) / 2), (sh_len, sh_w, (height - hop_top) / 2 + 1e-3),
                      accent, turret=s.turret))
    # uprights and an intake at the front
    if rng.random() < 0.7:
        post = rng.uniform(0.015, 0.03)
        for side in (-1, 1):
            parts.append(Part((sx - sh_len * 0.6, side * (hop_w - post), (chassis_top + height) / 2),
                              (post, post, (height - chassis_top) / 2), main))
    if rng.random() < 0.8:  # an over-the-bumper intake
        in_top = chassis_top + rng.uniform(0.03, 0.10)
        x0, x1 = fl - 0.10, fl + bd * rng.uniform(0.3, 1.2)
        parts.append(Part(((x0 + x1) / 2, 0.0, (chassis_top - 0.02 + in_top) / 2),
                          ((x1 - x0) / 2, fw * rng.uniform(0.6, 0.95), (in_top - chassis_top + 0.02) / 2),
                          _BODY_COLORS[int(rng.integers(len(_BODY_COLORS)))]))
    # shrink everything on top of the chassis a little so no two faces share a plane (they would
    # flicker between the two colors: "z-fighting")
    parts[1:] = [Part(p.center, (p.half[0] - 0.004 * (k + 1), p.half[1] - 0.003 * (k + 1), p.half[2]), p.color, p.turret)
                 for k, p in enumerate(parts[1:])]
    number = str(int(rng.integers(1, 11000)))
    return RobotStyle(number, bumper, tuple(parts))


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def lift(robot: Robot) -> float:
    """How high a climbing robot's frame is off the carpet (APPROX: hanging just under its rung)."""
    level = robot.climb_level or robot.climb_target
    if robot.climb_state == ClimbState.GROUND or level <= 0:
        return 0.0
    full = max(0.0, C.RUNG_HEIGHTS[level - 1] - max(robot.spec.height, 0.30) + 0.05)
    if robot.climb_state == ClimbState.CLIMBED:
        return full
    total = robot.spec.climb_duration(level) if robot.climb_state == ClimbState.CLIMBING else 1.5
    done = 1.0 - max(0.0, robot.climb_timer) / max(total, 1e-6)
    return full * (done if robot.climb_state == ClimbState.CLIMBING else 1.0 - done)


def robot_frame(robot: Robot) -> tuple[np.ndarray, np.ndarray]:
    """Where the robot's frame is in 3D: the origin (on the carpet under its center, or higher when
    it climbs or rides a BUMP) and its axes (forward, left, up) as columns. On a BUMP the robot
    rests on its four wheels, so it rises and tilts with the ramps. The strategy simulator is flat;
    this is only for drawing (and for the cameras riding on the robot)."""
    from .field_model import bump_height

    s = robot.spec
    yaw = _rot_z(robot.heading)
    hx, hy = s.length / 2 - C.MODULE_INSET, s.width / 2 - C.MODULE_INSET
    wheels = np.array([(hx, hy), (hx, -hy), (-hx, hy), (-hx, -hy)])
    field_xy = wheels @ yaw[:2, :2].T + (robot.x, robot.y)
    z = bump_height(field_xy[:, 0], field_xy[:, 1])
    base = np.array([robot.x, robot.y, lift(robot)])
    if not z.any():
        return base, yaw
    # the plane through the wheels' contact heights: z = a + b * forward + c * left (least squares)
    a, b, c = np.linalg.lstsq(np.column_stack([np.ones(4), wheels]), z, rcond=None)[0]
    fwd = np.array([1.0, 0.0, b])
    left = np.array([0.0, 1.0, c])
    up = np.cross(fwd, left)
    up /= np.linalg.norm(up)
    fwd /= np.linalg.norm(fwd)
    left = np.cross(up, fwd)
    base[2] += max(a, 0.0)
    return base, yaw @ np.column_stack([fwd, left, up])


def robot_primitives(robot: Robot, style: RobotStyle, obj: int) -> tuple[list[Box], list[Polygon]]:
    """Boxes and number plates of a robot at its current pose, all tagged with id ``obj``."""
    s = robot.spec
    base, rot = robot_frame(robot)
    bd = C.BUMPER_DEPTH
    hl, hw = s.length / 2, s.width / 2
    bz = BUMPER_BOTTOM + BUMPER_HEIGHT / 2
    hz = BUMPER_HEIGHT / 2
    boxes = []

    def add(center, half, color, extra=None):
        r = rot if extra is None else rot @ extra
        boxes.append(Box(tuple(base + rot @ np.asarray(center)), half, color, r, obj))

    add((hl - bd / 2, 0.0, bz), (bd / 2, hw, hz), style.bumper)  # front
    add((-hl + bd / 2, 0.0, bz), (bd / 2, hw, hz), style.bumper)  # back
    add((0.0, hw - bd / 2, bz), (hl - bd, bd / 2, hz), style.bumper)  # left
    add((0.0, -hw + bd / 2, bz), (hl - bd, bd / 2, hz), style.bumper)  # right
    turret = _rot_z(robot.turret) if s.turret else None
    for p in style.parts:
        if p.turret and turret is not None:
            c = np.array(p.center)
            # the turret spins about a vertical axis through its own center
            boxes.append(Box(tuple(base + rot @ c), p.half, p.color, rot @ turret, obj))
        else:
            add(p.center, p.half, p.color)
    # number plates on the four bumper faces, 2 mm proud of the fabric
    tex = number_texture(style.number, style.bumper)
    plate_h = BUMPER_HEIGHT * 0.8
    plate_w = min(plate_h * tex.shape[1] / tex.shape[0], 0.9 * min(s.length, s.width) - 2 * bd)
    polys = []
    for nx, ny in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        n = np.array([nx, ny, 0.0])
        a = np.array([-ny, nx, 0.0])  # the reader's right when facing this bumper
        reach = hl if nx else hw
        c = np.array([0.0, 0.0, bz]) + (reach + 0.002) * n
        up = np.array([0.0, 0.0, plate_h / 2])
        side = a * plate_w / 2
        local = [c - side + up, c + side + up, c + side - up, c - side - up]
        polys.append(Polygon(np.array([base + rot @ v for v in local]), style.bumper, obj, tex))
    return boxes, polys
