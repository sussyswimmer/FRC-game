"""What a robot's software knows about itself (high fidelity): a pose estimate, not the truth.

It works like WPILib's ``SwerveDrivePoseEstimator``. Every step the estimate moves by swerve
odometry: the wheel encoders and module angles, through the drive kinematics, turned onto the
field with the gyro heading. Several times a second the cameras see AprilTags and pull the
estimate toward a vision fix, weighted by a Kalman filter on the position uncertainty.

The pieces fail the way real ones do:

- encoders are slightly noisy, and a wheel that slips or a robot shoved sideways moves without
  the encoders noticing, so odometry drifts during pushing matches;
- the gyro drifts;
- a vision fix gets worse with distance to the tags and with motion blur, and is only possible
  when a tag faces the robot within range.

``HiFiConfig.sensor_noise`` scales every noise below (1 = these values). Robots use the estimate
to drive field-relative and to aim; the learner's observation is built from it.
"""

from __future__ import annotations

import math

import numpy as np

from .apriltags import TAG_NORMAL, TAG_XY
from .drivetrain import module_positions

ENCODER_SD = 0.01  # wheel speed noise, fraction of the reading
STEER_SD = 0.005  # rad, module angle noise
GYRO_DRIFT = 0.002  # rad per sqrt(s): random walk of the gyro's bias
ODOMETRY_SD = 0.03  # position uncertainty added per meter driven, as a fraction
PROCESS_SD = 0.05  # m per sqrt(s) of uncertainty added even at rest (bumps, unseen pushes)
VISION_RATE = 20.0  # Hz, pose fixes from the cameras
VISION_RANGE = 5.0  # m, tags farther away are not used
VISION_SD = (0.02, 0.006)  # m: a + b * d^2 with d the distance to the nearest tag seen
VISION_HEADING_SD = (0.01, 0.003)  # rad: a + b * d^2
OTHER_SD = 0.10  # m, other robots' positions as a robot perceives them
OTHER_VEL_SD = 0.25  # m/s, other robots' velocities
FUEL_SD = 0.05  # m, FUEL positions from object detection


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _kinematics(pos: np.ndarray) -> np.ndarray:
    """(8, 3) matrix taking chassis (vx, vy, omega) to every module's (vx, vy), robot frame."""
    rows = []
    for x, y in pos:
        rows += [(1.0, 0.0, -y), (0.0, 1.0, x)]
    return np.array(rows)


class PoseEstimator:
    """Pose estimates of all robots in a match."""

    def __init__(self, match, rng: np.random.Generator, noise: float) -> None:
        self.match = match
        self.rng = rng
        self.noise = noise
        robots = match.robots
        n = len(robots)
        self.est = np.array([(r.x, r.y, r.heading) for r in robots])  # starts at the known starting pose
        self.vel = np.zeros((n, 2))  # field-relative velocity from odometry
        self.var = np.full(n, 1e-4)  # position variance the filter believes, m^2
        self.hvar = np.full(n, 1e-5)  # heading variance, rad^2
        self.fixes = np.zeros(n, dtype=int)  # vision fixes used
        self._gyro = np.array([r.heading for r in robots])  # true heading at the last update
        self._fk = np.array([np.linalg.pinv(_kinematics(module_positions(r.spec))) for r in robots])
        self._next_fix = np.full(n, match.t + rng.uniform(0.0, 1.0 / VISION_RATE))

    def pose(self, i: int) -> tuple[float, float, float, float, float]:
        e, v = self.est[i], self.vel[i]
        return float(e[0]), float(e[1]), float(e[2]), float(v[0]), float(v[1])

    def update(self, dt: float) -> None:
        m, rng, k = self.match, self.rng, self.noise
        robots = m.robots
        n = len(robots)
        heading = np.array([r.heading for r in robots])
        turn = _wrap(heading - self._gyro) + rng.normal(0.0, GYRO_DRIFT * k * math.sqrt(dt), n)
        self._gyro = heading
        h0 = self.est[:, 2]
        h1 = _wrap(h0 + turn)
        if m.drive is not None:  # odometry from what the module sensors read
            w = m.drive.wheel_speed * (1.0 + rng.normal(0.0, ENCODER_SD * k, (n, 4)))
            a = m.drive.angle + rng.normal(0.0, STEER_SD * k, (n, 4))
            chassis = np.einsum("nij,nj->ni", self._fk, np.stack([w * np.cos(a), w * np.sin(a)], axis=2).reshape(n, 8))
            fwd, left = chassis[:, 0], chassis[:, 1]
        else:  # no module model: the true motion, with encoder noise
            v = np.array([(r.vx, r.vy) for r in robots])
            scale = 1.0 + rng.normal(0.0, ENCODER_SD * k, n)
            c, s = np.cos(heading), np.sin(heading)
            fwd = (c * v[:, 0] + s * v[:, 1]) * scale
            left = (-s * v[:, 0] + c * v[:, 1]) * scale
        mid = h0 + 0.5 * _wrap(h1 - h0)
        c, s = np.cos(mid), np.sin(mid)
        self.vel[:, 0] = c * fwd - s * left
        self.vel[:, 1] = s * fwd + c * left
        self.est[:, 0] += self.vel[:, 0] * dt
        self.est[:, 1] += self.vel[:, 1] * dt
        self.est[:, 2] = h1
        moved = np.hypot(self.vel[:, 0], self.vel[:, 1]) * dt
        self.var += (ODOMETRY_SD * k * moved) ** 2 + (PROCESS_SD * k) ** 2 * dt
        self.hvar += (GYRO_DRIFT * k) ** 2 * dt
        due = np.flatnonzero(m.t + dt >= self._next_fix)
        if due.size:
            self._next_fix[due] += 1.0 / VISION_RATE
            self._vision(due, [robots[i] for i in due])

    def _vision(self, idx: np.ndarray, robots) -> None:
        """Camera fixes for robots ``idx``: tags facing a robot within range give a noisy pose,
        blended into its estimate."""
        pos = np.array([(r.x, r.y, r.heading, r.speed, abs(r.omega)) for r in robots])
        dx = pos[:, 0:1] - TAG_XY[:, 0]
        dy = pos[:, 1:2] - TAG_XY[:, 1]
        d = np.hypot(dx, dy)
        seen = (d < VISION_RANGE) & (dx * TAG_NORMAL[:, 0] + dy * TAG_NORMAL[:, 1] > 0.25 * d)
        count = seen.sum(axis=1)
        ok = count > 0
        if not ok.any():
            return
        idx, pos, count = idx[ok], pos[ok], count[ok]
        near = np.where(seen[ok], d[ok], np.inf).min(axis=1)
        # motion blur, and several tags in view pin the pose down better than one
        scale = self.noise * (1.0 + 0.25 * pos[:, 3] + 0.2 * pos[:, 4]) / np.sqrt(np.minimum(count, 4))
        sd = (VISION_SD[0] + VISION_SD[1] * near * near) * scale
        hsd = (VISION_HEADING_SD[0] + VISION_HEADING_SD[1] * near * near) * scale
        noise = self.rng.normal(0.0, 1.0, (len(idx), 3))
        est, var, hvar = self.est[idx], self.var[idx], self.hvar[idx]
        g = var / (var + sd * sd)
        est[:, 0] += g * (pos[:, 0] + noise[:, 0] * sd - est[:, 0])
        est[:, 1] += g * (pos[:, 1] + noise[:, 1] * sd - est[:, 1])
        gh = hvar / (hvar + hsd * hsd)
        est[:, 2] = _wrap(est[:, 2] + gh * _wrap(pos[:, 2] + noise[:, 2] * hsd - est[:, 2]))
        self.est[idx] = est
        self.var[idx] = var * (1.0 - g)
        self.hvar[idx] = hvar * (1.0 - gh)
        self.fixes[idx] += 1
