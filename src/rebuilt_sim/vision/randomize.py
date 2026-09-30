"""Domain randomization: the random choices behind every synthetic image.

**Domain randomization** means varying everything that doesn't matter (colors, lighting, lenses,
camera placement, noise) so widely that a model trained on the images can only succeed by learning
what does matter (the shapes of FUEL, robots and tags). It is the standard way to make models
trained on synthetic images work on real ones (Tobin et al. 2017).

Every range is a field of ``RandomizationConfig`` so it can be changed without editing code
(``scripts/make_dataset.py --randomization my_ranges.json``). Every value drawn is written into the
image's labels. The defaults come from what FRC teams actually run (docs/04-vision.md section 5);
they are data-generation settings, not training choices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

from .camera import Intrinsics, Mount
from .render import Light


@dataclass(frozen=True)
class CameraPreset:
    """A real FRC camera: its native resolution, horizontal field of view and sensor type."""

    name: str
    width: int
    height: int
    hfov_deg: float
    mono: bool
    note: str = ""


PRESETS = {
    "limelight4": CameraPreset("limelight4", 1280, 800, 82.0, True, "OV9281 mono global shutter, 82 x 56.2 deg"),
    "limelight3g": CameraPreset("limelight3g", 1280, 800, 82.0, True, "OV9281 mono global shutter, 82 x 56.2 deg"),
    "limelight3": CameraPreset("limelight3", 640, 480, 62.5, False, "OV5647 color rolling shutter (skew not modeled)"),
    "ov9281": CameraPreset("ov9281", 1280, 800, 70.0, True, "Arducam OV9281 USB mono, 70 deg low-distortion lens"),
    "ov9782": CameraPreset("ov9782", 1280, 800, 70.0, False, "Arducam OV9782 USB color, 70 deg lens"),
    "ov2311": CameraPreset("ov2311", 1600, 1300, 78.0, True, "OV2311 / ThriftyCam class mono, about 75-80 deg"),
}


@dataclass(frozen=True)
class RandomizationConfig:
    """Ranges for everything drawn at random. (lo, hi) pairs are uniform ranges."""

    # lenses (per camera, fixed for a match)
    hfov_scale: tuple[float, float] = (0.95, 1.05)
    aspect_scale: tuple[float, float] = (0.99, 1.01)  # fy / fx
    principal_point_sd: float = 0.01  # fraction of the image size
    k1: tuple[float, float] = (-0.30, 0.10)
    k2: tuple[float, float] = (-0.10, 0.15)
    tangential_sd: float = 0.001  # p1, p2
    k3: tuple[float, float] = (-0.02, 0.02)
    lens_tries: int = 50  # redraws before falling back to no distortion
    # mounts (per camera): AprilTag cameras sit low and tilt up, game-piece cameras sit high and tilt down
    apriltag_role: float = 0.5  # chance a camera is an AprilTag camera
    apriltag_height: tuple[float, float] = (0.17, 0.40)
    apriltag_pitch_up_deg: tuple[float, float] = (10.0, 35.0)
    object_height: tuple[float, float] = (0.35, 0.70)
    object_pitch_up_deg: tuple[float, float] = (-30.0, -8.0)
    yaw_choices_deg: tuple[float, ...] = (0.0, 0.0, 0.0, 25.0, -25.0, 70.0, -70.0, 180.0)
    yaw_sd_deg: float = 3.0
    mount_error_sd_deg: float = 1.5  # pitch; roll gets 1 degree. Recorded as the truth
    lateral: tuple[float, float] = (-0.15, 0.15)  # m, sideways from the middle of the frame edge
    second_camera: float = 0.5  # chance a robot carries a second camera
    # lighting (per match)
    ambient: tuple[float, float] = (0.22, 0.50)
    sky: tuple[float, float] = (0.10, 0.35)
    diffuse: tuple[float, float] = (0.30, 0.70)
    specular: tuple[float, float] = (0.10, 0.40)
    light_tilt: tuple[float, float] = (0.0, 0.8)  # horizontal part of the light direction (0 = overhead)
    warmth: tuple[float, float] = (-0.08, 0.08)  # color temperature: + is warm (reddish), - is cool
    # the camera itself (per frame)
    exposure_ms_mono: tuple[float, float] = (1.0, 5.0)  # teams keep tag cameras' exposure short
    exposure_ms_color: tuple[float, float] = (3.0, 10.0)
    gain_db: tuple[float, float] = (0.0, 18.0)
    brightness_sd: float = 0.35  # log-normal spread of the overall exposure
    white_balance_sd: float = 0.06
    vignette: tuple[float, float] = (0.0, 0.35)
    defocus_px: tuple[float, float] = (0.0, 1.2)
    defocus_chance: float = 0.4
    shot_noise: tuple[float, float] = (0.0, 0.03)
    jpeg_chance: float = 0.7
    jpeg_quality: tuple[int, int] = (35, 95)

    @classmethod
    def from_dict(cls, d: dict) -> "RandomizationConfig":
        known = {f.name for f in fields(cls)}
        bad = set(d) - known
        if bad:
            raise ValueError(f"unknown randomization settings: {sorted(bad)}")
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def random_intrinsics(preset: CameraPreset, scale: float, rng: np.random.Generator, rc: RandomizationConfig | None,
                      ) -> tuple[Intrinsics, int]:
    """The preset's lens at the output resolution, varied like real units and calibrations vary.
    Distortion is redrawn until the lens model is valid (see ``Intrinsics.lens_error``); redrawing
    keeps the accepted lenses spread over the full range, where shrinking bad draws would bias them.
    Returns the lens and how many draws it took (0: no distortion)."""
    w, h = max(16, round(preset.width * scale)), max(16, round(preset.height * scale))
    if rc is None:
        return Intrinsics.from_fov(w, h, preset.hfov_deg), 0
    base = Intrinsics.from_fov(w, h, preset.hfov_deg * rng.uniform(*rc.hfov_scale))
    fy = base.fy * rng.uniform(*rc.aspect_scale)
    cx = base.cx + rng.normal(0.0, rc.principal_point_sd * w)
    cy = base.cy + rng.normal(0.0, rc.principal_point_sd * h)
    for tries in range(1, rc.lens_tries + 1):
        dist = (rng.uniform(*rc.k1), rng.uniform(*rc.k2), rng.normal(0.0, rc.tangential_sd),
                rng.normal(0.0, rc.tangential_sd), rng.uniform(*rc.k3))
        k = Intrinsics(w, h, base.fx, fy, cx, cy, tuple(float(d) for d in dist))
        if k.lens_error(step=max(4, w // 80)) < 1e-6:
            return k, tries
    return Intrinsics(w, h, base.fx, fy, cx, cy), 0


def random_mount(robot, rng: np.random.Generator, rc: RandomizationConfig) -> tuple[Mount, str]:
    """A camera placed the way FRC teams place them, at the robot's edge on the side it faces."""
    role = "apriltag" if rng.random() < rc.apriltag_role else "object"
    yaw = math.radians(float(rng.choice(rc.yaw_choices_deg)) + rng.normal(0.0, rc.yaw_sd_deg))
    height = max(robot.spec.height, 0.30)
    if role == "apriltag":
        z = rng.uniform(rc.apriltag_height[0], max(rc.apriltag_height[0], min(rc.apriltag_height[1], height)))
        pitch = rng.uniform(*rc.apriltag_pitch_up_deg)
    else:
        lo = min(rc.object_height[0], height)
        z = rng.uniform(lo, max(lo, min(rc.object_height[1], height + 0.05)))
        pitch = rng.uniform(*rc.object_pitch_up_deg)
    reach_x, reach_y = robot.spec.length / 2 - 0.10, robot.spec.width / 2 - 0.10
    t = min(reach_x / max(abs(math.cos(yaw)), 1e-6), reach_y / max(abs(math.sin(yaw)), 1e-6))
    side = rng.uniform(*rc.lateral)
    x = t * math.cos(yaw) - side * math.sin(yaw)
    y = t * math.sin(yaw) + side * math.cos(yaw)
    return Mount(float(np.clip(x, -reach_x, reach_x)), float(np.clip(y, -reach_y, reach_y)), float(z), yaw,
                 math.radians(pitch + rng.normal(0.0, rc.mount_error_sd_deg)),
                 math.radians(rng.normal(0.0, rc.mount_error_sd_deg * 2 / 3))), role


def random_light(rng: np.random.Generator, rc: RandomizationConfig | None) -> Light:
    if rc is None:
        return Light()
    tilt = rng.uniform(*rc.light_tilt)
    az = rng.uniform(0, 2 * math.pi)
    warm = rng.uniform(*rc.warmth)
    return Light(direction=(tilt * math.cos(az), tilt * math.sin(az), 1.0), ambient=rng.uniform(*rc.ambient),
                 sky=rng.uniform(*rc.sky), diffuse=rng.uniform(*rc.diffuse), specular=rng.uniform(*rc.specular),
                 color=(1.0 + warm, 1.0, 1.0 - warm))
