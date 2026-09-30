"""Cameras: a pinhole model with optional lens distortion, mounted on a robot.

Conventions (the same as OpenCV, so labels line up with ``cv2`` tools):

- **Camera frame:** x right, y down, z forward (out of the lens).
- **Pixels:** (u, v) = (column, row), with pixel *centers* at integer coordinates. The top-left
  pixel covers u, v in [-0.5, 0.5].
- **Distortion:** OpenCV's Brown-Conrady model (k1, k2, p1, p2, k3).

A robot's camera is placed with a ``Mount`` in WPILib's robot frame (x forward, y left, z up,
meters from the robot's center on the carpet). ``pitch_up`` is positive for a camera tilted
**up**. WPILib's ``Rotation3d`` pitch has the opposite sign (positive is nose down), so the same
mount in WPILib is ``Transform3d(x, y, z, Rotation3d(roll, -pitch_up, yaw))``; ``Mount.to_wpilib``
gives it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)  # k1, k2, p1, p2, k3

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float, dist=(0.0, 0.0, 0.0, 0.0, 0.0)) -> "Intrinsics":
        """Square pixels, principal point at the image center, horizontal field of view in degrees."""
        f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        return cls(width, height, f, f, (width - 1) / 2, (height - 1) / 2, tuple(dist))

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan((self.width / 2) / self.fx))

    def scaled(self, k: int) -> "Intrinsics":
        """The same camera sampled k times finer in each direction (for supersampling). Pixel
        centers stay consistent: coarse pixel u covers fine pixels k*u .. k*u + k - 1."""
        return replace(self, width=self.width * k, height=self.height * k, fx=self.fx * k, fy=self.fy * k,
                       cx=self.cx * k + (k - 1) / 2, cy=self.cy * k + (k - 1) / 2)

    # ---------------------------------------------------------------- distortion
    def _distort_normalized(self, x, y):
        k1, k2, p1, p2, k3 = self.dist
        r2 = x * x + y * y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        return (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)

    def distort(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Normalized image coordinates (X/Z, Y/Z) -> pixels."""
        if any(self.dist):
            x, y = self._distort_normalized(x, y)
        return self.fx * x + self.cx, self.fy * y + self.cy

    def undistort(self, u: np.ndarray, v: np.ndarray, iterations: int = 30) -> tuple[np.ndarray, np.ndarray]:
        """Pixels -> normalized image coordinates: the inverse of ``distort``, by Newton's method
        (it converges in a few steps even where simple fixed-point iteration, as in
        ``cv2.undistortPoints``, drifts)."""
        xd = (np.asarray(u, dtype=np.float64) - self.cx) / self.fx
        yd = (np.asarray(v, dtype=np.float64) - self.cy) / self.fy
        k1, k2, p1, p2, k3 = self.dist
        if not (k1 or k2 or p1 or p2 or k3):
            return xd, yd
        x, y = xd.copy(), yd.copy()
        for _ in range(iterations):
            r2 = x * x + y * y
            radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
            dradial = k1 + r2 * (2 * k2 + 3 * k3 * r2)  # d radial / d r2
            px, py = self._distort_normalized(x, y)
            ex, ey = px - xd, py - yd
            a = radial + 2 * x * x * dradial + 2 * p1 * y + 6 * p2 * x  # the Jacobian is [[a, b], [b, d]]
            b = 2 * x * y * dradial + 2 * p1 * x + 2 * p2 * y
            d = radial + 2 * y * y * dradial + 6 * p1 * y + 2 * p2 * x
            det = a * d - b * b
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)
            x = x - (d * ex - b * ey) / det
            y = y - (a * ey - b * ex) / det
            if max(float(np.abs(ex).max(initial=0.0)), float(np.abs(ey).max(initial=0.0))) < 1e-15:
                break
        return x, y

    def lens_error(self, step: int = 4, guard: float = 1.2) -> float:
        """How far (pixels) undistorting then re-distorting pixels misses, worst case over a grid that
        includes the image border; ``inf`` if the lens model folds over. A distortion polynomial
        stops being a valid lens where it bends back on itself (then some pixels have no ray, or
        two). So the radial part must keep growing out to ``guard`` times the image corners'
        radius, and the rays must run in order across the image. Real calibrated lenses pass easily."""
        us = np.unique(np.r_[np.arange(0, self.width, step), self.width - 1]).astype(np.float64)
        vs = np.unique(np.r_[np.arange(0, self.height, step), self.height - 1]).astype(np.float64)
        u, v = np.meshgrid(us, vs)
        x, y = self.undistort(u, v)
        u2, v2 = self.distort(x, y)
        err = np.hypot(u2 - u, v2 - v)
        if not np.isfinite(err).all() or (np.diff(x, axis=1) <= 0).any() or (np.diff(y, axis=0) <= 0).any():
            return float("inf")
        k1, k2, _, _, k3 = self.dist
        r = np.linspace(0.0, guard * float(np.hypot(x, y).max()), 400)
        slope = 1 + r * r * (3 * k1 + r * r * (5 * k2 + 7 * k3 * r * r))  # d(r * radial) / dr
        return float("inf") if (slope <= 0).any() else float(err.max())

    def rays(self) -> np.ndarray:
        """(height, width, 3) ray through each pixel center in the camera frame, scaled so z = 1
        (a hit at ray parameter t is t meters in front of the camera)."""
        u, v = np.meshgrid(np.arange(self.width, dtype=np.float64), np.arange(self.height, dtype=np.float64))
        x, y = self.undistort(u, v)
        return np.stack([x, y, np.ones_like(x)], axis=-1)


@dataclass(frozen=True)
class Mount:
    """Where a camera sits on a robot, in WPILib's robot frame (x forward, y left, z up)."""

    x: float = 0.25
    y: float = 0.0
    z: float = 0.5
    yaw: float = 0.0  # rad, positive turns the camera to the robot's left
    pitch_up: float = 0.0  # rad, positive tilts the camera up (WPILib's pitch is the negative of this)
    roll: float = 0.0  # rad, positive rolls clockwise as seen from behind the camera (as in WPILib)

    def to_wpilib(self) -> dict:
        """The mount as WPILib's robot-to-camera ``Transform3d``: translation and quaternion [w, x, y, z]."""
        rot = _rot_z(self.yaw) @ _rot_y(-self.pitch_up) @ _rot_x(self.roll)
        return {"translation": [self.x, self.y, self.z], "quaternion": quaternion(rot).tolist()}


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def quaternion(rot: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion [w, x, y, z] (WPILib's order), with w >= 0."""
    m = np.asarray(rot, dtype=np.float64)
    t = float(np.trace(m))
    if t > 0:
        s = 2.0 * math.sqrt(1.0 + t)
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(max(1.0 + m[i, i] - m[j, j] - m[k, k], 1e-300))
        q = np.zeros(4)
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def rodrigues(rot: np.ndarray) -> np.ndarray:
    """Rotation matrix -> OpenCV rotation vector (axis times angle, radians), like ``cv2.Rodrigues``."""
    q = quaternion(rot)
    s = float(np.linalg.norm(q[1:]))
    if s < 1e-12:
        return np.zeros(3)
    return q[1:] / s * 2.0 * math.atan2(s, q[0])


# columns: the OpenCV camera axes (right, down, forward) written in a forward-left-up frame
_CV_FROM_FLU = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


class Camera:
    """A camera at ``position`` (field frame, meters) whose axes, as columns of ``rotation``, are the
    OpenCV camera axes (right, down, forward) expressed in the field frame."""

    def __init__(self, intrinsics: Intrinsics, position, rotation: np.ndarray) -> None:
        self.intrinsics = intrinsics
        self.position = np.asarray(position, dtype=np.float64)
        self.rotation = np.asarray(rotation, dtype=np.float64)  # field <- camera

    @classmethod
    def looking(cls, intrinsics: Intrinsics, position, yaw: float, pitch_up: float = 0.0, roll: float = 0.0) -> "Camera":
        """A camera at a field position facing ``yaw`` (rad, field frame), tilted up by ``pitch_up``."""
        flu = _rot_z(yaw) @ _rot_y(-pitch_up) @ _rot_x(roll)
        return cls(intrinsics, position, flu @ _CV_FROM_FLU)

    @classmethod
    def on_robot(cls, intrinsics: Intrinsics, x: float, y: float, heading: float, mount: Mount) -> "Camera":
        """The camera of a robot at (x, y, heading), flat on the carpet."""
        c, s = math.cos(heading), math.sin(heading)
        pos = (x + c * mount.x - s * mount.y, y + s * mount.x + c * mount.y, mount.z)
        return cls.looking(intrinsics, pos, heading + mount.yaw, mount.pitch_up, mount.roll)

    @property
    def forward(self) -> np.ndarray:
        return self.rotation[:, 2]

    def to_camera(self, points) -> np.ndarray:
        """(n, 3) field points -> (n, 3) camera-frame points."""
        return (np.asarray(points, dtype=np.float64) - self.position) @ self.rotation

    def project(self, points) -> tuple[np.ndarray, np.ndarray]:
        """(n, 3) field points -> (n, 2) pixels and (n,) depth along the optical axis. Points at or
        behind the camera get nan pixels."""
        pc = self.to_camera(points)
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            front = z > 1e-9
            x = np.where(front, pc[:, 0] / np.where(front, z, 1.0), np.nan)
            y = np.where(front, pc[:, 1] / np.where(front, z, 1.0), np.nan)
        u, v = self.intrinsics.distort(x, y)
        return np.column_stack([u, v]), z

    def pose(self) -> dict:
        """Where the camera is, three ways:

        - plain numbers: position, and yaw / pitch_up / roll in degrees;
        - ``wpilib``: the camera's ``Pose3d`` in the field (x forward, y left, z up axes), quaternion [w, x, y, z];
        - ``opencv``: the extrinsics ``R``, ``t`` and ``rvec`` that map field points into the camera
          frame (X_cam = R X_field + t), as ``cv2.solvePnP`` and ``cv2.projectPoints`` use them.
        """
        f, left, up = self.rotation[:, 2], -self.rotation[:, 0], -self.rotation[:, 1]
        yaw = math.atan2(f[1], f[0])
        pitch = math.asin(max(-1.0, min(1.0, f[2])))  # up is positive
        roll = math.atan2(left[2], up[2])
        r_cv = self.rotation.T
        t_cv = -r_cv @ self.position
        return {
            "x": float(self.position[0]), "y": float(self.position[1]), "z": float(self.position[2]),
            "yaw_deg": math.degrees(yaw), "pitch_up_deg": math.degrees(pitch), "roll_deg": math.degrees(roll),
            "wpilib": {"translation": self.position.tolist(),
                       "quaternion": quaternion(np.column_stack([f, left, up])).tolist()},
            "opencv": {"R": r_cv.tolist(), "t": t_cv.tolist(), "rvec": rodrigues(r_cv).tolist()},
        }
