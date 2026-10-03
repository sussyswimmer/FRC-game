"""Camera effects: turn a clean rendered frame into something that looks like it came off a real
robot camera.

A detector trained only on perfect renders learns the renderer's quirks (flat colors, razor-sharp
edges, no noise) and then fails on real images: the "sim-to-real gap". **Domain randomization**
narrows it by varying everything a real camera varies, so the only thing that stays the same from
image to image is what we want the model to learn. The effects here, in the order a camera applies
them:

1. **Exposure and white balance**: overall brightness and color cast (arena lighting, auto-exposure).
2. **Vignetting**: corners darker than the center (cheap lenses).
3. **Blur**: motion blur from the robot moving or turning during the exposure, and defocus.
4. **Tone curve**: linear light -> the gamma-encoded 8-bit values images store.
5. **Sensor noise**: shot noise that grows with brightness plus a constant read-noise floor.
6. **JPEG compression**: blocky artifacts from the camera stream's compression (applied when saving).

``Effects`` holds one draw of all the settings, so a dataset can record exactly what each image got.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Effects:
    exposure: float = 1.0  # multiplies linear light
    white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0)  # per-channel gain
    vignette: float = 0.0  # fraction of brightness lost in the far corners
    motion_px: float = 0.0  # motion blur length in pixels
    motion_angle: float = 0.0  # rad, direction of the blur in the image
    defocus_px: float = 0.0  # blur radius in pixels
    gamma: float = 2.2
    noise: float = 0.0  # shot noise strength (std at full brightness, 0..1 scale)
    read_noise: float = 0.0  # constant noise floor std (0..1 scale)
    jpeg_quality: int | None = None  # None: save losslessly
    grayscale: bool = False  # monochrome sensor (most AprilTag cameras)
    # how a monochrome sensor weighs red, green and blue light. Sensors without an infrared filter
    # see red (and near-infrared) more strongly than the eye does, so this is not plain luminance.
    mono_weights: tuple[float, float, float] = (0.299, 0.587, 0.114)
    # recorded for reference (the blur and noise above already include their effect)
    exposure_ms: float | None = None
    gain_db: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def random_effects(rng: np.random.Generator, strength: float = 1.0, mono: bool | None = None, rc=None) -> Effects:
    """One random draw from the ranges in ``rc`` (a ``RandomizationConfig``; the defaults if None).
    ``strength`` scales them: 0 gives a clean image, 1 the full ranges. ``mono`` forces a
    monochrome or color sensor (default: 15% monochrome)."""
    from .randomize import RandomizationConfig

    rc = rc or RandomizationConfig()
    s = strength
    wb = np.exp(rng.normal(0.0, rc.white_balance_sd * s, 3))
    if mono is None:
        mono = bool(rng.random() < 0.15 * s)
    weights = np.array([0.299, 0.587, 0.114]) + rng.uniform([-0.05, -0.15, -0.05], [0.25, 0.05, 0.05]) * s
    weights = np.clip(weights, 0.02, None)
    q_lo, q_hi = rc.jpeg_quality
    return Effects(
        exposure=float(np.exp(rng.normal(0.0, rc.brightness_sd * s))),
        white_balance=tuple(float(x) for x in wb / wb.mean()),
        vignette=float(rng.uniform(rc.vignette[0], rc.vignette[1]) * s),
        motion_px=float(rng.exponential(1.5 * s)) if rng.random() < 0.5 else 0.0,
        motion_angle=float(rng.uniform(0, math.pi)),
        defocus_px=float(rng.uniform(*rc.defocus_px) * s) if rng.random() < rc.defocus_chance else 0.0,
        gamma=float(rng.uniform(2.0, 2.4)),
        noise=float(rng.uniform(*rc.shot_noise) * s),
        read_noise=float(rng.uniform(0.0, 0.01 * s)),
        jpeg_quality=int(rng.integers(max(10, int(q_hi - (q_hi - q_lo) * s)), q_hi + 1))
        if rng.random() < rc.jpeg_chance * s else None,
        grayscale=mono,
        mono_weights=tuple(float(w) for w in weights / weights.sum()),
    )


def _kernel_line(length: float, angle: float) -> np.ndarray:
    n = max(1, int(math.ceil(length)))
    size = 2 * n + 1
    k = np.zeros((size, size), dtype=np.float32)
    steps = max(2, 4 * n)
    for t in np.linspace(-length / 2, length / 2, steps):
        x, y = n + t * math.cos(angle), n + t * math.sin(angle)
        k[int(round(y)), int(round(x))] += 1.0
    return k / k.sum()


def _kernel_disk(radius: float) -> np.ndarray:
    n = max(1, int(math.ceil(radius)))
    y, x = np.mgrid[-n:n + 1, -n:n + 1]
    k = np.clip(radius + 0.5 - np.hypot(x, y), 0.0, 1.0).astype(np.float32)
    return k / k.sum()


def _convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Small-kernel 2D convolution of an (h, w, c) image, edges repeated."""
    kh, kw = kernel.shape
    py, px = kh // 2, kw // 2
    pad = np.pad(img, ((py, py), (px, px), (0, 0)), mode="edge")
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    for dy, dx in zip(*np.nonzero(kernel)):
        out += kernel[dy, dx] * pad[dy:dy + h, dx:dx + w]
    return out


def apply(color: np.ndarray, fx: Effects, rng: np.random.Generator) -> np.ndarray:
    """Linear float (h, w, 3) render -> uint8 (h, w, 3) camera image (JPEG is applied when saving)."""
    img = color.astype(np.float32) * fx.exposure * np.asarray(fx.white_balance, dtype=np.float32)
    h, w = img.shape[:2]
    if fx.vignette > 0:
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((x - (w - 1) / 2) ** 2 + (y - (h - 1) / 2) ** 2) / (((w - 1) / 2) ** 2 + ((h - 1) / 2) ** 2)
        img *= (1.0 - fx.vignette * r2)[..., None]
    if fx.motion_px >= 0.5:
        img = _convolve(img, _kernel_line(fx.motion_px, fx.motion_angle))
    if fx.defocus_px >= 0.5:
        img = _convolve(img, _kernel_disk(fx.defocus_px))
    img = np.clip(img, 0.0, 1.0) ** (1.0 / fx.gamma)
    if fx.grayscale:
        img = np.repeat((img @ np.asarray(fx.mono_weights, dtype=np.float32))[..., None], 3, axis=-1)
    if fx.noise > 0 or fx.read_noise > 0:
        std = np.sqrt(fx.noise ** 2 * img + fx.read_noise ** 2)
        img = img + std * rng.standard_normal(img.shape, dtype=np.float32)
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def downsample(color: np.ndarray, k: int) -> np.ndarray:
    """Average k x k blocks: turns a supersampled render into smooth-edged output pixels."""
    if k == 1:
        return color
    h, w = color.shape[0] // k, color.shape[1] // k
    return color[:h * k, :w * k].reshape(h, k, w, k, -1).mean(axis=(1, 3))
