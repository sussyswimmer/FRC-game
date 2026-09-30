"""Build a 3D scene from a match: the field, the robots and every visible FUEL, plus the label
information for each object.

Which FUEL are drawn:

- **on the ground:** resting on the carpet, or on a BUMP's ramp;
- **in flight:** at the height the high-fidelity ballistics computes; with the strategy physics,
  along a made-up arc from the shooter to its target, because that physics has no heights;
- **in an OUTPOST CHUTE:** stacked behind the clear OUTPOST wall.

FUEL held inside robots or being processed inside a HUB are not drawn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import constants as C
from ..constants import Alliance
from ..robot import ClimbState
from ..sim import Match
from . import ids
from .camera import Camera, Intrinsics, Mount
from .field_model import FieldAppearance, FieldModel, bump_height, chute_fuel_positions, hub_cap, srgb, team_signs
from .labels import ObjectInfo
from .render import Scene, Sphere
from .robot_model import RobotStyle, lift, random_style, robot_frame, robot_primitives

FUEL_COLOR = srgb(235, 195, 45)


@dataclass
class MatchLook:
    """Everything about a match's appearance that stays fixed from frame to frame."""

    field: FieldModel
    styles: list[RobotStyle]
    fuel_colors: np.ndarray  # (504, 3) per-FUEL albedo (a little wear variation)
    hub_glow: float = 1.0  # brightness of the HUB lights


def match_look(match: Match, rng: np.random.Generator, randomize: bool = True, field: FieldModel | None = None) -> MatchLook:
    """Pick the field colors, each robot's look and the FUEL colors for a match."""
    if field is None:
        field = FieldModel(FieldAppearance.random(rng) if randomize else FieldAppearance())
    styles = [random_style(r, rng) for r in match.robots]
    base = np.array(FUEL_COLOR)
    if randomize:
        base = np.clip(base * rng.uniform(0.8, 1.15) * rng.uniform(0.95, 1.05, 3), 0.0, 1.0)
    wear = rng.uniform(0.88, 1.05, (C.FUEL_TOTAL, 1)) if randomize else np.ones((C.FUEL_TOTAL, 1))
    glow = float(rng.uniform(0.3, 1.5)) if randomize else 1.0
    return MatchLook(field, styles, np.clip(base * wear, 0.0, 1.0), glow)


def hub_light(match: Match, a: Alliance, look: MatchLook) -> tuple[float, float, float] | None:
    """The HUB's light: its alliance color while it counts, pulsing in the last 3 s before it
    turns off, dark otherwise (Game Manual table 5-3)."""
    if match.t >= C.MATCH_END or not match.hub_active(a):
        return None
    level = 1.0
    ttt = match.time_to_toggle(a)
    if ttt is not None and ttt < 3.0:
        level = 0.5 + 0.5 * math.cos(2 * math.pi * 2.0 * match.t)
    color = np.asarray(look.field.appearance.alliance[a])
    color = color / max(color.max(), 1e-6)  # a saturated light, not the paint
    return tuple(float(c) for c in color * 1.3 * look.hub_glow * level)


def _flight_positions(match: Match, idx: np.ndarray) -> np.ndarray:
    if match.shooters is not None:
        return np.column_stack([match.f_pos[idx], match.f_z[idx]])
    # the strategy physics has no heights: draw a parabola from the shooter to the HUB or the landing spot
    t0, t1 = match.f_t0[idx], match.f_time[idx]
    frac = np.clip((match.t - t0) / np.maximum(t1 - t0, 1e-6), 0.0, 1.0)
    target = np.where(match.f_hit[idx, None], np.array(match.field.hub_centers)[match.f_owner[idx].clip(0)], match.f_land[idx])
    xy = match.f_pos[idx] + (target - match.f_pos[idx]) * frac[:, None]
    z1 = np.where(match.f_hit[idx], C.HUB_OPENING_HEIGHT, C.FUEL_RADIUS)
    z = 0.55 + (z1 - 0.55) * frac + 4 * frac * (1 - frac) * 1.4
    return np.column_stack([xy, z])


def fuel_positions(match: Match) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Indices, (n, 3) centers and states of every FUEL to draw."""
    r = C.FUEL_RADIUS
    idx, pts, states = [], [], []
    ground = np.flatnonzero(match.f_state == C.GROUND)
    if ground.size:
        xy = match.f_pos[ground]
        z = bump_height(xy[:, 0], xy[:, 1])
        z = np.where(z > 0, z + r / math.cos(math.radians(15.0)), r)
        idx.append(ground)
        pts.append(np.column_stack([xy, z]))
        states += ["ground"] * ground.size
    flight = np.flatnonzero(match.f_state == C.FLIGHT)
    if flight.size:
        idx.append(flight)
        pts.append(_flight_positions(match, flight))
        states += ["flight"] * flight.size
    for a in Alliance:
        chute = np.flatnonzero((match.f_state == C.CHUTE) & (match.f_owner == a))
        if chute.size:
            idx.append(chute)
            pts.append(chute_fuel_positions(a, chute.size))
            states += ["chute"] * chute.size
    if not idx:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3)), []
    return np.concatenate(idx), np.concatenate(pts), states


def _robot_points(robot, height: float) -> np.ndarray:
    s = robot.spec
    base, rot = robot_frame(robot)
    local = np.array([(dx, dy, z) for dx in (-s.length / 2, s.length / 2) for dy in (-s.width / 2, s.width / 2)
                      for z in (0.0, height)])
    return base + local @ rot.T


def build_scene(match: Match, look: MatchLook, hide: set[int] | None = None) -> tuple[Scene, dict[int, ObjectInfo]]:
    """The match as it is right now. Robots listed in ``hide`` are left out."""
    fm = look.field
    parts = fm.parts
    scene = Scene(boxes=list(parts.boxes), polygons=list(parts.polygons), cylinders=list(parts.cylinders),
                  floor=fm.floor, floor_obj=ids.CARPET, background=fm.background, multipart=set(fm.multipart))
    objects = dict(fm.objects)
    for a in Alliance:
        extra = hub_cap(a, fm.appearance, hub_light(match, a, look))
        signs = team_signs(a, [look.styles[r.index].number for r in match.robots if r.alliance == a], fm.appearance)
        for p in (extra, signs):
            scene.boxes += p.boxes
            scene.polygons += p.polygons
            scene.cylinders += p.cylinders
    blobs = []
    for r in match.robots:
        if hide and r.index in hide:
            continue
        obj = ids.ROBOT + r.index
        boxes, polys = robot_primitives(r, look.styles[r.index], obj)
        scene.boxes += boxes
        scene.polygons += polys
        scene.multipart.add(obj)
        height = max(r.spec.height, 0.30)
        objects[obj] = ObjectInfo("robot", {
            "alliance": r.alliance.name.lower(), "robot_index": r.index, "team_number": look.styles[r.index].number,
            "pose": {"x": r.x, "y": r.y, "heading_deg": math.degrees(r.heading)},
            "climbing": r.climb_state != ClimbState.GROUND, "fuel_held": r.fuel,
        }, _robot_points(r, height))
        if lift(r) < 0.05:
            blobs.append((r.x, r.y, 0.62 * max(r.spec.length, r.spec.width), 0.55))
    idx, pts, states = fuel_positions(match)
    rad = C.FUEL_RADIUS
    axes = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=np.float64) * rad
    for i, p, state in zip(idx, pts, states):
        scene.spheres.append(Sphere(tuple(p), rad, tuple(look.fuel_colors[i]), int(i), shiny=True))
        objects[int(i)] = ObjectInfo("fuel", {"state": state, "center": [float(v) for v in p]}, p + axes, sphere=(p, rad))
        if state == "ground":
            blobs.append((p[0], p[1], 1.7 * rad, 0.6))
    scene.blobs = np.array(blobs) if blobs else None
    return scene, objects


def robot_camera(match: Match, index: int, intrinsics: Intrinsics, mount: Mount) -> Camera:
    """A camera on a robot. It rides up when the robot climbs and tilts with it on a BUMP."""
    base, rot = robot_frame(match.robots[index])
    local = Camera.looking(intrinsics, (mount.x, mount.y, mount.z), mount.yaw, mount.pitch_up, mount.roll)
    return Camera(intrinsics, base + rot @ local.position, rot @ local.rotation)
