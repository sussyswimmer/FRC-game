"""Swerve drivetrain dynamics (high fidelity).

Each robot has four swerve modules. Every physics step does what a real robot's code and
hardware do:

1. The field-relative drive command becomes robot-relative chassis speeds, using the robot's
   *own* heading estimate (a drifting gyro makes it drive slightly crooked).
2. Inverse kinematics gives every module a target speed and angle. Speeds are desaturated so
   no wheel is asked to beat its motor, a module turns the short way and reverses its wheel
   when the target is more than 90 degrees away, and the speed is scaled by the cosine of the
   steering error (WPILib's ``desaturateWheelSpeeds``, ``optimize`` and ``cosineScale``).
3. Each module steers toward its angle at a limited rate. Its drive motor runs a velocity
   loop; the motor's torque is limited by the battery voltage, its back-EMF and the stator
   current limit.
4. The carpet pushes back through each wheel: the drive force along the wheel plus the
   sideways friction that stops the wheel sliding sideways, together limited by the friction
   circle (wheel_cof times the weight on the wheel). Beyond it the wheel slips, and a slipping
   wheel's encoder no longer matches the ground, which is what makes odometry drift.
5. The wheel forces move the chassis: F = m a, torque = I alpha.

Motor constants are WPILib's ``DCMotor`` values. Everything is vectorized over robots (rows)
and modules (columns: front-left, front-right, back-left, back-right).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import constants as C

STEER_TIME = 0.04  # s, time constant of a module's steering loop (before its rate limit)
LOOP_GAIN = 0.4  # fraction of a speed error the drive velocity loop removes per step, at most


@dataclass(frozen=True)
class DCMotor:
    """A brushed/brushless DC motor at 12 V (WPILib ``DCMotor``)."""

    stall_torque: float  # N m
    stall_current: float  # A
    free_current: float  # A
    free_speed: float  # rad/s
    voltage: float = 12.0

    @property
    def resistance(self) -> float:
        return self.voltage / self.stall_current

    @property
    def kv(self) -> float:  # rad/s per volt
        return self.free_speed / (self.voltage - self.resistance * self.free_current)

    @property
    def kt(self) -> float:  # N m per amp
        return self.stall_torque / self.stall_current


def _rpm(x: float) -> float:
    return x * 2 * math.pi / 60


MOTORS = {
    "kraken_x60": DCMotor(7.09, 366.0, 2.0, _rpm(6000)),
    "kraken_x60_foc": DCMotor(9.37, 483.0, 2.0, _rpm(5800)),
    "falcon_500": DCMotor(4.69, 257.0, 1.5, _rpm(6380)),
    "neo": DCMotor(3.28, 181.0, 1.3, _rpm(5676)),
    "neo_vortex": DCMotor(3.6, 211.0, 3.6, _rpm(6784)),
}


def module_positions(spec) -> np.ndarray:
    """(4, 2) wheel positions in the robot frame (x forward, y left): FL, FR, BL, BR."""
    hx = spec.length / 2 - C.BUMPER_DEPTH - C.MODULE_INSET
    hy = spec.width / 2 - C.BUMPER_DEPTH - C.MODULE_INSET
    return np.array([(hx, hy), (hx, -hy), (-hx, hy), (-hx, -hy)])


def moment_of_inertia(spec) -> float:
    """Yaw inertia: a uniform plate the size of the footprint, times 0.75 because the heavy
    parts (battery, motors, gearboxes) sit well inside the bumpers."""
    return 0.75 * spec.mass * (spec.length ** 2 + spec.width ** 2) / 12.0


def free_speed(spec) -> float:
    """m/s at the tread when the drive motors spin freely at 12 V."""
    d = spec.drive
    return MOTORS[d.motor].free_speed / d.gear_ratio * d.wheel_radius


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class SwerveDrive:
    """Module states and dynamics for all robots of a match."""

    def __init__(self, robots, dt: float) -> None:
        self.dt = dt
        n = len(robots)
        pos = np.array([module_positions(r.spec) for r in robots])  # (n, 4, 2)
        self.mx, self.my = pos[..., 0], pos[..., 1]
        self.mass = np.array([r.spec.mass for r in robots])
        self.inertia = np.array([moment_of_inertia(r.spec) for r in robots])
        motors = [MOTORS[r.spec.drive.motor] for r in robots]
        gear = np.array([r.spec.drive.gear_ratio for r in robots])
        wheel = np.array([r.spec.drive.wheel_radius for r in robots])
        self.res = np.array([m.resistance for m in motors])[:, None]  # ohm
        self.ke = (gear / (np.array([m.kv for m in motors]) * wheel))[:, None]  # V per m/s at the tread
        self.kf = (np.array([m.kt for m in motors]) * gear / wheel)[:, None]  # N at the tread per A
        self.volts = np.array([m.voltage for m in motors])[:, None]
        self.amp_limit = np.array([r.spec.drive.current_limit for r in robots])[:, None]
        self.grip = (np.array([r.spec.drive.wheel_cof for r in robots]) * self.mass * C.GRAVITY / 4)[:, None]
        self.steer_rate = np.array([r.spec.drive.steer_rate for r in robots])[:, None]
        self.top = np.array([free_speed(r.spec) for r in robots])[:, None]  # desaturation limit
        # Velocity loop: voltage = ke * target + kp * error. kp is set so the loop takes out at most
        # LOOP_GAIN of a speed error per step (stiffer gains would need a smaller physics step).
        share = self.mass[:, None] / 4
        self.kp = np.clip(LOOP_GAIN * share * self.res / (dt * self.kf) - self.ke, 0.0, 3.0)
        # effective mass of the chassis at each wheel for sideways pushes is computed per step
        self.angle = np.zeros((n, 4))  # module angles in the robot frame (all start pointing forward)
        self.wheel_speed = np.zeros((n, 4))  # what each wheel's encoder reads, m/s at the tread
        self.current = np.zeros((n, 4))  # A per drive motor
        self.slipping = np.zeros((n, 4), dtype=bool)

    def step(self, robots, target: np.ndarray, heading_est: np.ndarray, active: np.ndarray) -> None:
        """Advance every ``active`` robot by one step.

        ``target`` is (n, 3): field-relative vx, vy (m/s) and omega (rad/s) each robot's code asks
        for; ``heading_est`` is the heading its code believes it has.
        """
        dt = self.dt
        idx = np.flatnonzero(active)
        if idx.size == 0:
            return
        sel = slice(None) if idx.size == len(robots) else idx  # a view instead of a copy when all drive
        th = np.array([robots[i].heading for i in idx])
        vel = np.array([(robots[i].vx, robots[i].vy, robots[i].omega) for i in idx])
        tvx, tvy, tw = target[sel, 0], target[sel, 1], target[sel, 2]
        mx, my = self.mx[sel], self.my[sel]

        # 1-2. chassis speeds in the robot frame (as the robot believes it is facing) -> module targets
        ch, sh = np.cos(heading_est[sel]), np.sin(heading_est[sel])
        rvx, rvy = ch * tvx + sh * tvy, -sh * tvx + ch * tvy
        mvx = rvx[:, None] - tw[:, None] * my
        mvy = rvy[:, None] + tw[:, None] * mx
        speed = np.hypot(mvx, mvy)
        top = self.top[sel]
        worst = speed.max(axis=1, keepdims=True)
        speed = np.where(worst > top, speed * top / np.maximum(worst, 1e-9), speed)
        angle = self.angle[sel]
        err = _wrap(np.arctan2(mvy, mvx) - angle)
        flip = np.abs(err) > np.pi / 2
        speed = np.where(flip, -speed, speed)
        err = np.where(flip, _wrap(err + np.pi), err)
        err = np.where(np.abs(speed) < 1e-3, 0.0, err)  # stopped: hold the module where it is
        speed = speed * np.cos(err)

        # 3. steering, then the drive motors
        rate = self.steer_rate[sel] * dt
        angle = _wrap(angle + np.clip(err * min(1.0, dt / STEER_TIME), -rate, rate))
        self.angle[sel] = angle
        wa = th[:, None] + angle  # wheel direction on the field
        dx, dy = np.cos(wa), np.sin(wa)
        ct, st = np.cos(th)[:, None], np.sin(th)[:, None]
        px, py = ct * mx - st * my, st * mx + ct * my  # wheel positions relative to the center, field frame
        gvx = vel[:, 0:1] - vel[:, 2:3] * py
        gvy = vel[:, 1:2] + vel[:, 2:3] * px
        roll = gvx * dx + gvy * dy  # ground speed along the wheel
        side = -gvx * dy + gvy * dx  # ground speed across the wheel (sliding)
        ke, res, kf = self.ke[sel], self.res[sel], self.kf[sel]
        volts = np.clip(ke * speed + self.kp[sel] * (speed - roll), -self.volts[sel], self.volts[sel])
        amps = np.clip((volts - ke * roll) / res, -self.amp_limit[sel], self.amp_limit[sel])
        f_drive = kf * amps

        # 4. sideways friction: the push that would stop this wheel sliding, shared by the four
        # wheels (they all act at once), then the friction circle
        arm = px * dx + py * dy  # (wheel position) x (sideways direction)
        m_eff = 1.0 / (1.0 / self.mass[sel, None] + arm ** 2 / self.inertia[sel, None])
        f_side = -m_eff * side / (4 * dt)
        total = np.hypot(f_drive, f_side)
        grip = self.grip[sel]
        k = np.where(total > grip, grip / np.maximum(total, 1e-9), 1.0)
        f_drive, f_side = f_drive * k, f_side * k
        slipping = k < 0.999
        # A wheel that keeps its grip rolls with the ground. One that lost it has almost no inertia, so
        # its velocity loop spins it to where the motor makes just the force the carpet can take: near
        # the commanded speed, drooping with that force (and less if the battery voltage runs out).
        loop = ke + self.kp[sel]
        spin = speed - res * f_drive / (kf * loop)
        sat = np.abs(ke * speed + self.kp[sel] * (speed - spin)) > self.volts[sel]
        spin = np.where(sat, (np.sign(volts) * self.volts[sel] - res * f_drive / kf) / ke, spin)
        self.wheel_speed[sel] = np.where(slipping & (np.abs(amps) > 1e-3), spin, roll)
        self.current[sel] = f_drive / kf
        self.slipping[sel] = slipping

        # 5. chassis motion
        fx = f_drive * dx - f_side * dy
        fy = f_drive * dy + f_side * dx
        torque = (px * fy - py * fx).sum(axis=1)
        vel[:, 0] += fx.sum(axis=1) / self.mass[sel] * dt
        vel[:, 1] += fy.sum(axis=1) / self.mass[sel] * dt
        vel[:, 2] += torque / self.inertia[sel] * dt
        for j, i in enumerate(idx):
            r = robots[i]
            r.vx, r.vy, r.omega = float(vel[j, 0]), float(vel[j, 1]), float(vel[j, 2])
            r.x += r.vx * dt
            r.y += r.vy * dt
            r.heading = (r.heading + r.omega * dt + math.pi) % (2 * math.pi) - math.pi

    def stop(self, i: int) -> None:
        """A robot that can't drive (climbing, broken): wheels stop."""
        self.wheel_speed[i] = 0.0
        self.current[i] = 0.0
        self.slipping[i] = False
