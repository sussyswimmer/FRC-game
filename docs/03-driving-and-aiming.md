# Step 3: driving and aiming fidelity

Step 2's physics is built for **match strategy**. Robots are circles that reach the speed they ask for as fast as `max_accel` allows, and a shot scores with a hit probability. That is fast, calibrated, and fine for choosing *what* to do. It is too forgiving for learning *how* to drive and aim.

Step 3 adds a **high-fidelity physics mode** for that. It runs the same match, rules, referee, bots and scoring, with the parts a robot's own code has to cope with made realistic:

| | Strategy physics (step 2, the default) | High fidelity (step 3) |
|---|---|---|
| Physics step | 0.05 s | **0.02 s**, WPILib's 50 Hz robot loop |
| Drivetrain | Velocity changes at up to `max_accel` | **Swerve modules**: steering, motor torque and current limits, traction, wheel slip |
| Commands | Act immediately | **Latency**: the motors act on the command from 40 ms ago |
| Bumpers | Circles | **Rectangles** that turn when hit off-center |
| What a robot knows | The truth | A **pose estimate**: odometry, a drifting gyro, AprilTag vision fixes |
| Shooting | Hit probability by distance and speed | **3D FUEL flight** with drag, into the 41.73 in hexagonal HUB opening 72 in up |
| Aiming | Automatic | The robot's **aim software** (a shot table and shoot-on-the-move lead), or **your agent sets exit speed, hood and turret** |

The rules engine and the 34 original tests are unchanged, and the default is still the strategy physics. Turn step 3 on with one flag.

## 1. Turning it on

```python
from rebuilt_sim.sim import HiFiConfig, MatchConfig, make_match
m = make_match(["elite", "strong", "mid"], ["strong", "low", "mid"], seed=0,
               config=MatchConfig(hifi=HiFiConfig()))

from rebuilt_sim.env import EnvConfig, RebuiltEnv
env = RebuiltEnv(EnvConfig(action_mode="continuous_aim", hifi=HiFiConfig()))

import gymnasium as gym, rebuilt_sim.env
env = gym.make("Rebuilt-Aim-HiFi-v0")
```

| Gymnasium id | Actions | What the agent controls |
|---|---|---|
| `Rebuilt-Strategy-HiFi-v0` | `Discrete(8)` macros | Strategy, on the realistic physics (the controller drives, the aim software shoots) |
| `Rebuilt-Control-HiFi-v0` | `Box(6)` | Driving, intake, when to shoot, climb; the aim software sets the shot |
| `Rebuilt-Aim-HiFi-v0` | `Box(9)` | All of the above plus exit speed, hood angle and turret angle |

The scripts take `--hifi`:

```bash
python scripts/play.py --hifi             # drive it yourself: feel the latency and the traction limit
python scripts/watch.py --hifi
python scripts/calibrate.py --matches 96 --hifi
python scripts/evaluate.py --baseline-only --tier strong --hifi
```

A model trained on high fidelity is watched and evaluated on it automatically: `run_config_for` reads `hifi` from the run's `config.json`.

**Every part can be switched off on its own**, for example to see how much one of them changes a policy:

| `HiFiConfig` field | Default | Off, or other values |
|---|---|---|
| `dt` | 0.02 s | any physics step (0.01 was tested) |
| `swerve` | on | the strategy drive model (acceleration-limited), at `dt` |
| `footprints` | on | circles |
| `ballistics` | on | the hit probability |
| `latency` | 0.04 s | 0 |
| `sensor_noise` | 1.0 (the values in §5) | 0: robots know the truth; 2: twice the noise |
| `fuel_every` | 2 | FUEL rolling physics runs every Nth step (0.04 s), to save time; 1 = every step |

## 2. Swerve drivetrain (`drivetrain.py`)

Every step does what a robot's code and hardware do:

1. **Field-relative to robot-relative.** The driver or agent asks for field-relative speeds (vx, vy, turn rate). The robot converts them with *its own* heading estimate, so a drifting gyro makes it drive slightly crooked.
2. **Inverse kinematics.** Each of the four modules gets a target wheel speed and direction. As in WPILib:
   - speeds are **desaturated**, so no wheel is asked to beat its motor (turning while driving flat out costs top speed);
   - a module that would have to turn more than 90° instead turns the short way and runs its wheel backwards (**optimize**);
   - the wheel speed is scaled by the cosine of the steering error (**cosine compensation**).
3. **Steering and motors.**
   - Each module turns toward its target at a limited rate.
   - Each drive motor runs a velocity loop: voltage = feedforward + proportional correction, capped at 12 V.
   - The motor's current is set by that voltage minus its **back-EMF** (a spinning motor generates voltage against the battery, so torque falls as speed rises). It is capped at the **stator current limit**.
   - Torque = kT × current, geared down to a force at the tread.
4. **The carpet pushes back.**
   - Each wheel carries a quarter of the weight.
   - The drive force along the wheel, plus the sideways friction that stops the wheel sliding sideways, are limited together by the **friction circle**: μ × the weight on that wheel.
   - Beyond it the wheel **slips**. A slipping wheel's encoder no longer matches the ground, which is what makes odometry drift.
5. **The chassis moves:** F = m·a and torque = I·α, with I computed from the footprint.

Motor constants are WPILib's `DCMotor` values (stall torque, stall current, free current, free speed at 12 V):

| Motor | Stall torque | Stall current | Free speed |
|---|---|---|---|
| Kraken X60 | 7.09 N·m | 366 A | 6000 rpm |
| Kraken X60 (FOC) | 9.37 N·m | 483 A | 5800 rpm |
| Falcon 500 | 4.69 N·m | 257 A | 6380 rpm |
| NEO | 3.28 N·m | 181 A | 5676 rpm |
| NEO Vortex | 3.6 N·m | 211 A | 6784 rpm |

Each tier has hardware that reproduces its calibrated `max_speed` and `max_accel`. The current limit is chosen so that the low-speed acceleration equals `max_accel`. Mass is 60 kg, wheels have 4 in diameter, and μ = 1.1 (tread on carpet).

| Tier | Bumpers (m) | Drive | Gearing | Free speed | Current limit | Steering |
|---|---|---|---|---|---|---|
| elite | 0.84 × 0.84 | Kraken X60 FOC | 6.12 (MK4i L3) | 5.04 m/s | 55 A | 30 rad/s |
| strong | 0.86 × 0.86 | Kraken X60 | 6.75 (L2) | 4.73 m/s | 44 A | 25 rad/s |
| mid, climber | 0.90 × 0.82 | NEO | 6.75 (L2) | 4.47 m/s | 40 A | 20 rad/s |
| low | 0.92 × 0.82 | NEO | 8.14 (L1) | 3.71 m/s | 26 A | 15 rad/s |
| elite_climber | 0.84 × 0.84 | Kraken X60 FOC | 6.12 (L3) | 5.04 m/s | 55 A | 30 rad/s |

**What it does** (measured by `tests/test_drivetrain.py` and while building):

- **Acceleration and top speed:** peak acceleration is within 2% of each tier's `max_accel`, and 0 → 90% of top speed takes 0.52–0.58 s (the kinematic model: 0.48–0.54 s).
- **Stopping:** a strong robot stops from 4 m/s in 0.6 s, over 1.2 m.
- **Changing direction:** the modules must turn first, and the tires scrub the old momentum off at up to μ·g ≈ 10.8 m/s².
- **Traction:** with a 150 A current limit, acceleration still tops out at μ·g and the wheels slip.
- **Spin and skew:** it spins in place without drifting. Driving straight while spinning at 3 rad/s wanders about 9 cm over 2 m, the classic swerve "skew" (WPILib's `ChassisSpeeds.discretize` exists to correct it).
- **Pushing matches** are decided by current limits and traction, not mass:
  - an elite robot shoves a braking low robot 4 m in 3 s;
  - the low robot moves the elite only 0.7 m;
  - pushed sideways, a braking robot's wheels grip with the full μ·m·g.

**Numerics.**

- The wheel's own inertia is negligible, so wheels are treated as massless: a wheel that keeps its grip rolls with the ground. One that slips spins to where its motor makes only the force the carpet allows, which is near the commanded speed, because the velocity loop holds it there.
- The sideways-friction force is solved as an impulse shared by the four wheels, which is stable at any step size.
- The velocity loop gain is chosen per robot so that one step removes at most 40% of a speed error.

## 3. Latency

Every command reaches the motors 40 ms late (`latency`). Commands queue per robot, and nothing is executed while robots are disabled. Real robots see 20–60 ms from joystick or network to motor output.

- The bots' code compensates where a real team's would: before firing, it checks legality at the position it will reach after the latency.
- An agent has to learn to lead its own commands.

## 4. Bumpers and collisions (`collision.py`)

Robots are their **bumper rectangles**: a 27.5 in frame plus 3.25 in bumpers, turning with the robot.

- **Contacts** come from the separating axis test. The contact point is found by clipping the touching edges:
  - two robots meeting bumper-to-bumper push through the middle of the overlap;
  - a corner hit pushes at the corner and **spins** the robot that was hit.
- **Each contact gets an impulse** that stops the bodies moving into each other, with a little bounce and bumper friction. Leftover overlap is pushed apart by mass.
- **Field walls and elements** are fixed rectangles: the HUBs, TRENCH columns and TOWER uprights for every robot, and the TRENCH openings and the TOWER's rungs and supports for robots too tall to pass under them (the clearance classes in [02-simulator.md](02-simulator.md) §4).
- **Zone rules use the real footprint.** G407 (bumpers in the zone), G403 (fully across the CENTER LINE), the TOWER and OUTPOST checks all use a rotated rectangle's extent. A robot turned 45° reaches further.
- **FUEL** is bulldozed out through the nearest side of the rectangle, and the intake sits at the front bumper.

**Bot changes that only apply to rectangles:**

- **Steering along a touching wall, teammate or parked robot.** A circle slides off by itself. A swerve robot's wheels grip sideways, so a robot driving diagonally into a TRENCH column or a teammate simply stops, as it would on a real field. Opponents are still pushed: that is defense. **An agent in `continuous` mode has to learn this itself.**
- **Driving the intake right up to the DEPOT wall.** A flat bumper pins FUEL against the wall, where the circle's approach point could not reach it.

## 5. Sensors and the pose estimate (`sensors.py`, `apriltags.py`)

In high fidelity a robot's software (and the learner's observation) uses **what the robot believes**, which is not the truth. The estimator works like WPILib's `SwerveDrivePoseEstimator`:

- **Odometry, every step.** Wheel encoders and module angles, through least-squares forward kinematics, turned onto the field with the gyro heading.
- **Gyro:** integrates the true rotation plus a random-walk bias.
- **Vision:**
  - At 20 Hz the cameras see the **official 2026 AprilTags** (`data/2026-rebuilt-welded.json`, WPILib).
  - A tag counts as seen when it faces the robot and is within 5 m.
  - A fix's noise grows with the square of the distance to the nearest tag, and with motion blur. Several tags in view make it tighter.
- **Blending:** a Kalman filter on the position uncertainty, so odometry is trusted less the further the robot has driven since a good fix.

| Noise (`sensor_noise` = 1) | Value |
|---|---|
| Wheel encoder | 1% of the reading |
| Module angle | 0.005 rad |
| Gyro drift | 0.002 rad/√s |
| Odometry uncertainty growth | 3% of distance driven, plus 0.05 m/√s |
| Vision position | 0.02 m + 0.006 m × d² to the nearest tag, × (1 + 0.25·speed + 0.2·turn rate), ÷ √(tags seen, up to 4) |
| Vision heading | 0.01 rad + 0.003 rad × d², scaled the same way |
| Other robots, as a robot perceives them | 0.10 m, 0.25 m/s |
| FUEL positions (object detection) | 0.05 m |

**How it fails realistically:**

- Encoders don't notice a wheel slipping, or a robot being shoved sideways. Odometry then drifts, and vision pulls it back:
  - with the cameras blind, a shoved robot's estimate is off by the whole shove;
  - with them back, it is within 10 cm in 1.5 s (`tests/test_hifi.py`).
- **Measured error** over full bot matches is fine for driving and aiming, with occasional large misses during aggressive scrubbing:

  | Statistic | Position (8 matches) | Heading |
  |---|---|---|
  | median | 1.3 cm | 0.08° |
  | 95% of the time under | 9.5 cm | 0.24° |
  | 99% of the time under | 24 cm | 0.34° |
  | worst | 0.8 m | 0.8° |

- **Aiming uses the estimate too,** so a bad pose estimate makes shots miss, as it does on a real robot.

## 6. FUEL ballistics and shooting (`ballistics.py`)

**Flight:**

- FUEL leaves the shooter at the exit speed, hood elevation and aim direction the mechanism is actually at, plus shot-to-shot spread, on top of the robot's own velocity.
- In flight it falls under gravity and slows with quadratic air drag. FUEL is 0.227 kg with a drag coefficient of 0.5, so drag shortens a typical shot by 8–15% (a strong robot's 2.4 m shot would fly 2.73 m in a vacuum).

**Scoring and bounces:**

- It **scores** by dropping through the **hexagonal opening at the top of the HUB** (41.73 in across the flats inside, 72 in up; drawing GE-26300). Its flats face the guardrails and its corners point at the alliance walls, so a shot from straight in front has 48.2 in of opening (corner to corner) along its line of flight, and one from 30° to the side 41.7 in. A ball clipping the rim counts as in when its center is inside the rim by at least half a radius.
- It **bounces** off the HUB's top rim outside the opening, off the HUB's sides, off the field walls, and on the carpet, then rolls.
- It **can drop into the other alliance's HUB**, which scores for that alliance.

**The mechanism lags:**

- the flywheel spins up with a time constant, and each shot takes a few percent of its speed;
- the hood and turret move at limited rates.

**The aim software.** The scripted bots, the `macro` mode and the `continuous` mode use it, like a team's shooter code:

- **A shot table** (`shot_table`) is computed per shooter from this same flight model and time step:
  - for every distance (0.05 m apart) and every hood angle, the exit speed that crosses the opening's height right over the HUB center, coming down, and clearing the HUB's side on the way;
  - for each distance, the hood whose shot is **least sensitive to that shooter's spreads**. It picks high arcs.
- **Lead:** robots whose spec says `lead` (the elite tiers) aim at a virtual target that cancels their own motion, so they can **shoot on the move**. The others don't: a moving shot from them misses by the robot's motion.
  - The lead ignores drag, so it gets less exact with speed: with no spread, strafing past the HUB at 1.5 m/s scores 8 of 8, and at 3 m/s 6 of 8.
  - Without lead, the same passes score 0 and 1 of 8.
- **Interlock with hysteresis:** it starts a volley only when the flywheel is within 2%, the hood within 0.015 rad and the aim within the tier's tolerance, all judged from the robot's *estimated* pose. It then keeps feeding FUEL while the flywheel stays within 6%, the hood within 0.05 rad and the aim within twice the tolerance.
  - Each shot takes some flywheel speed out, and waiting for full recovery before every ball would stall a volley (real shooter code doesn't).
  - Later balls in a volley are a little less accurate, which is part of the calibrated spread.

**The shooters** (spreads calibrated so hit rates match each tier's calibrated `accuracy` at its sweet range):

| Tier | Exit speed | Hood | Turret | Spin-up | Speed spread | Angle spread | Aim tolerance | Leads | Can score from |
|---|---|---|---|---|---|---|---|---|---|
| elite | 4.5–12 m/s | 40–80° | yes | 0.25 s | 3.9% | 1.5° | 1.7° | yes | 1.05 m and out |
| strong | 4.5–10 m/s | 49–74° | no | 0.4 s | 9% | 3.0° | 3.4° | no | 1.3–7.2 m |
| mid, climber | 4.5–9 m/s | 57–74° | no | 0.6 s | 12.4% | 4.5° | 4.6° | no | 1.2–5.6 m |
| low | 4.5–8.5 m/s | 72° (fixed) | no | 0.8 s | 13% | 4.9° | 5.7° | no | 1.35–3.35 m |
| elite_climber | as elite | | | | | | | | |

- **How they were tuned:** by bisection, so that a stationary robot at its sweet range hits at its step-2 `accuracy`. Each tier's speed and angle spreads were scaled together, with default sensor noise, over 320 shots per try.
- **Why they are large:** a tier bundles every early-season error source: foam compression, rough tuning, and flywheel speed lost to the previous ball.
- **Each shot's flywheel loss:** elite 2%, strong 3%, mid 4%, low 5%.

**Validation:** tuned only at the sweet range, the physics produces a drop-off with distance much like the one step 2 hand-calibrated. Stationary hit rates, with default sensor noise (about 240 shots per cell, ±3%):

| Tier | Distance: hit rate, high fidelity (step 2 hit probability) |
|---|---|
| elite | 2.6 m: 92% (90%) · 3.2 m: 92% (90%) · 4.2 m: 88% (85%) · 6.5 m: 82% (74%) |
| strong | 1.8 m: 72% (65%) · 2.4 m: 64% (65%) · 3.4 m: 54% (55%) · 4.5 m: 42% (44%) |
| mid | 1.4 m: 45% (50%) · 2.0 m: 53% (50%) · 3.0 m: 36% (36%) · 3.5 m: 31% (29%) |
| low | 1.6 m: 35% (35%) · 2.6 m: 30% (17%) · 2.8 m: 26% (13%) |

- Only the sweet-range value was tuned.
- Beyond it, the elite, strong and mid tiers fall off at a similar rate to step 2's hand-tuned rule, within about 10 points.
- The table was re-measured after the geometry fix turned the HUB's hexagon 30° (corners toward the alliance walls, as in the drawings). The sweet-range rates stayed on target, so the spreads were kept. Long shots gained the most, because a shot from in front now has the opening's corner-to-corner length along its flight: elite at 4.2 m went from 76% to 88%, strong at 4.5 m from 34% to 42%.
- The low tier's fixed hood keeps it flatter.

**Manual aiming** (`RobotCommand.shot_speed/hood/turret`, the `continuous_aim` action mode): the shooter goes where it is told, and every shot fires as commanded. There is no interlock, so the agent has to learn to:

- wait for the flywheel;
- pick speed and hood for the distance;
- lead its own motion;
- keep the shot legal (G407).

## 7. The RL interface in high fidelity

**Observation: 170 floats.** The first 162 have the same layout as step 2 (`docs/02-simulator.md` §6), with three differences:

- the robot's own pose, velocity, zone flags and HUB distance come from its **pose estimate**;
- other robots are seen with 0.10 m and 0.25 m/s noise;
- the nearest-FUEL positions have 0.05 m noise. The FUEL count grid stays exact, as a driver's view of the field.

The new slice 162–169 (`OBS_LAYOUT["shooter"]`) is the shooter, which the agent can't otherwise see: flywheel speed, hood angle, turret angle (cos, sin, relative to the robot), and the exit-speed and hood ranges.

**Actions:**

- **`continuous_aim`, `Box(9)`:** the 6 `continuous` actions, then:
  - exit speed (−1…1 spans the robot's speed range);
  - hood (−1…1 spans its hood range);
  - turret angle (−1…1 is ±π; ignored for robots without a turret).
- **`continuous`:** the same 6 values as in step 2. The aim software sets the shot, and the robot fires when the agent asks *and* the shot is ready.
- **`macro`:** unchanged; the controller drives and the aim software shoots.

**Decisions this leaves to you** (training design is yours; see the curriculum in HANDOFF.md):

- **Decision interval.** The env default stays `decision_dt = 0.1` s: 5 physics steps per decision, 1660 decisions per match. `0.02` matches a real robot loop but makes episodes 5× longer.
- **Which parts to switch on.** They can be switched on together or one at a time, for example as a curriculum.
- **Which action mode to learn in.**

Rewards are unchanged (`RewardConfig`).

## 8. Calibration: high fidelity against the strategy physics

`python scripts/calibrate.py --matches 96 [--hifi]`, same seeds and robot mix (random Day 1 alliances):

| | Strategy physics | High fidelity | Real Day 1 |
|---|---|---|---|
| FUEL per robot: elite / strong / mid / low | 279 / 58 / 25 / 6.8 | 235 / 43 / 17 / 7.5 | ~225–250 / 40–70 / 15–30 / <15 |
| Share of shots that go in: elite / strong / mid / low | 85% / 62% / 47% / 30% | 86% / 63% / 48% / 44% | – |
| Elite + strong + strong alliance, FUEL | 403 | about 313 (24 matches) | 393, 394, 409 |
| Alliance FUEL: median / mean / max | 55 / 79 / 392 | 38 / 63 / 343 | 34 / 87 / 409 |
| Alliances at 100+ FUEL | 17% | 11% | 24% |
| AUTO share of FUEL | 20.6% | 24.4% | 19.8% |
| Foul awards per alliance | 0.28 | 0.26 | 0.26 |
| TOWER points per alliance | 0.47 | 0.26 | 0.48 |
| Median winning margin | 32 | 31 | 42 |

These are the numbers after the geometry fix (HANDOFF.md §11). TOWER points rest on about a dozen
climber robots with a 50% success chance, so they move by about ±0.2 between runs: the two modes
started the same number of climbs here. The median winning margin is a few points lower than before
the fix (32 and 28 on two sets of seeds, against 40 and 33 before): mid robots now collect from the
DEPOT reliably, which evens out the Day 1 mix's matches a little.

**What the table shows:**

- **Accuracy carries over:** the elite, strong and mid tiers hit the same share of shots in both physics modes. The low tier hits more (44% vs 30%): its shots are mostly the preload from close range, and its hit rate falls off less steeply with distance than step 2's linear rule assumes (see the table in §6).
- **Robots score about 15–30% less FUEL, from slower cycles:**
  - Alone on an empty field, a robot collects almost as fast as before: in 20 s the elite collects 8% less and the strong robot 4% less.
  - In full matches, robots carry more FUEL on average (62 across the field vs 54). Unloading takes longer: non-turret robots must line up within a few degrees instead of ±15°, the flywheel spins up, the high-arc shots fly about 1 s, and every command is 40 ms late.
  - A strong robot drives about 20% less distance per match.
  - This is what realistic physics costs a scripted driver, not a bug; the diagnosis runs are described in HANDOFF.md §9.
- **Against the real matches:**
  - Per robot, every tier stays inside its real Day 1 range. The elite now sits right on the estimate for team 9483.
  - Full elite alliances fall about 20% short of the real 393–409 FUEL.
  - If the bots should reproduce those alliance totals in high fidelity, the knobs are the bots' high-fidelity driving (motion profiles that plan for latency and module steering) or the tier hardware. That is a possible follow-up.
- **Choosing a mode:** strategy work that needs the calibrated opponents should use the strategy physics; control work uses high fidelity.

## 9. Speed

On the 4-core cloud container used to build this:

| | Time per match |
|---|---|
| Strategy physics | 3.1 s |
| High fidelity | 10.2 s (96 bot matches in 238 s on 4 workers) |

- Most of the increase is the 2.5× more steps (8,300 vs 3,320); the rest is spread over the drivetrain, collisions, bots, sensors and FUEL flight.
- Expect roughly half those times on the RTX 3070 / i7-13700F PC.
- In `continuous` mode with `decision_dt = 0.1`, one core then gives about 300 decisions per second; parallel environments scale with cores.

## 10. Assumptions to verify (APPROX)

| Assumption | Where | Why it matters |
|---|---|---|
| FUEL mass 0.227 kg and drag coefficient 0.5 | `constants.FUEL_MASS`, `FUEL_DRAG_COEF` | Shot tables and ranges |
| Spin (Magnus lift) is not modeled | `ballistics.flight_step` | Backspin makes real shots float farther |
| Bumper depth 3.25 in, module inset 0.065 m, 60 kg robots | `constants.py`, `robot.py` | Footprints, turning inertia |
| All robots are swerve | `robot.DriveSpec` | Some 2026 teams ran tank drive (the KitBot) |
| Cameras see every tag within 5 m that faces them; robots don't block tags | `sensors.py` | Vision availability |
| FUEL flies through robots and is held in by the field walls at any height | `ballistics.py` | Blocked shots; FUEL leaving the field (G405 is still not modeled) |

## 11. Tests

`python -m pytest -q` runs the original 34 tests, unchanged, plus:

- **`tests/test_drivetrain.py`:**
  - acceleration per tier, the traction cap, module turn-before-go, holding still;
  - latency; pushing matches; off-center hits spinning the robot;
  - bumpers staying on the field and out of each other and the field elements under random driving;
  - TRENCH height gating; footprint-based zone rules.
- **`tests/test_ballistics.py`:**
  - the hexagonal opening; drag against the vacuum formula;
  - every tier able to score from where it shoots;
  - spread-free aim-software shots all scoring, from 1.6 m to 5.5 m;
  - flywheel spin-up; short and long misses landing on the carpet; rim bounces;
  - shoot-on-the-move with and without lead; manual aiming.
- **`tests/test_hifi.py`:**
  - the AprilTag table against the official file;
  - a full bot match: FUEL conserved, the elite robot still carries, pose errors sane;
  - determinism; perfect sensing;
  - odometry drift when shoved and the vision fix;
  - observations built from the estimate;
  - Gymnasium's checker on all three action modes; `continuous_aim` action mapping and validation.
