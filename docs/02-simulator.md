# Step 2: REBUILT simulator and strategy-track environments

This step builds the shared core (rules and field) and the **match-strategy** track: a fast 2D simulator, scripted opponents calibrated to real matches, Gymnasium/PettingZoo environments, and training scripts **you run yourself**. The driving/aiming and vision tracks will build on the same core (see §8).

## 1. Architecture

```
constants.py  field geometry, timing, points, RP thresholds (manual TU22 + AprilTag layout)
rules.py      clock, HUB schedule, grace windows, scoring, ranking points, fouls (pure, unit-tested)
field.py      obstacles, zones, BUMP/TRENCH crossings, TOWER/DEPOT/OUTPOST queries
robot.py      RobotSpec (what a team built) + calibrated tiers; Robot state; RobotCommand
sim.py        Match: physics for 6 robots and 504 FUEL, shooting, HUB counting, climbing, referee
controller.py macro actions -> RobotCommand (navigation, FUEL targeting, shooting spots, defense)
bots.py       scripted drivers with Day 1 habits (sweep AUTO, hoard/dump, jams, distraction)
obs.py        162-float observation in the robot's own alliance frame
env.py        Gymnasium env (1 learner + 5 bots); MatchRunner shared with pz_env.py (PettingZoo)
viewer.py     Pygame top-down renderer with a broadcast-style scoreboard
runner.py     matches mixing trained policies and bots (watch / evaluate / play)
policies.py   ActorCritic network (self-play) + load_policy for .pt and SB3 .zip models
```

The physics runs at 20 Hz (`dt = 0.05 s`). A full match is 166 s: 20 s AUTO, a 3 s pause, 140 s TELEOP and a 3 s final grace window. That is 3,320 physics steps, which take **about 1–2 s of CPU time**. Parallel workers scale with CPU cores: 96 bot matches take about 23 s on 22 workers.

## 2. Field and coordinates

- **Coordinate system:** WPILib's blue-origin frame, in meters. x runs 0 → 16.541 from the blue alliance wall to the red one; y runs 0 → 8.069. This is the frame your robot code already uses.
- **Geometry from the official 2026 Field Dimension Drawings** (FE-2026 rev B, welded field) and the Game Manual. The official AprilTag layout (`2026-rebuilt-welded.json`, allwpilib v2026.2.1) agrees with them, and `tests/test_field.py` checks it does. Blue side:

| Element | Position |
|---|---|
| HUB | 1.194 m (47 in) square centered at (4.626, 4.035) |
| Alliance zone | 0 → 4.029 m (the far edge of the ROBOT STARTING LINE, against the HUB) |
| HUB line | 4.029 → 5.222 m |
| TRENCH openings | y 0 → 1.279 and 6.790 → 8.069, 0.565 m clearance |
| TOWER | Open frame in the alliance wall, centered at y 3.746 (tag 31). Uprights 1.016–1.105 m from the wall, rungs spanning y 3.149–4.343 |
| OUTPOST | CHUTE centered at y 0.666 (tag 29) |
| DEPOT | 42 × 27 in floor area centered at y 5.965, 1.2 m past the TOWER's floor plate |

- **Red side:** every red element is the blue one rotated 180° about the field center. The TRENCHes are the exception: both alliances' fixed TRENCH arms are on the scoring-table side (see `field.py`).
- **Estimated, not in the drawings (marked APPROX in `constants.py`):**
  - OUTPOST feed zone size
  - HUB processing time (0.4–0.9 s)
  - HUB exit speed (1.2–3.2 m/s)

## 3. Rules implemented

| Rule | Simulator behavior |
|---|---|
| Match timing | AUTO 20 s, pause 3 s (robots disabled, in-flight FUEL still counts for AUTO), TRANSITION 10 s, four 25 s SHIFTs, END GAME 30 s |
| HUB schedule | Both HUBs are on in AUTO, TRANSITION and END GAME. The AUTO FUEL leader is **off in SHIFTs 1 and 3**; the other alliance is off in SHIFTs 2 and 4. A tie is a random coin flip. |
| Grace windows | FUEL passing the counter up to 3 s after its HUB switches off, after AUTO ends, or after the match ends still scores. So shots fired just before a switch count, and so do shots in the first moments after it. |
| Scoring | FUEL 1 pt in an active HUB, 0 in an inactive one (tracked as `wasted_fuel`). TOWER: 15 for LEVEL 1 in AUTO (max 2 robots), 10/20/30 for LEVELS 1/2/3 in TELEOP. |
| Ranking points | ENERGIZED ≥100 FUEL, SUPERCHARGED ≥360, TRAVERSAL ≥50 TOWER points (DCMP/CMP thresholds selectable). Win 3, tie 1. |
| G403 | Robot fully past the CENTER LINE in AUTO: MAJOR, plus a MAJOR per opponent contact. Macro actions never plan past the line. |
| G407 | Launching FUEL while no part of the bumpers is in your own ALLIANCE ZONE: MAJOR (at most one per second). |
| G418 | Pinning an opponent that is trying to get away: MINOR after 3 s, then a MAJOR every 3 s. Resets when the robots are 72 in apart for 3 s, or after 5 s without pinning. |
| G420 | In END GAME, contact with an opponent at its TOWER: MAJOR, and the victim gets LEVEL 3 if it was off the ground. |
| Human player | The OUTPOST chute (24 FUEL) feeds a friendly robot parked at the chute with its intake on, at 4 FUEL/s. |
| Not modeled | G405 (FUEL out of the field), G408 (catching HUB output), G415–417 (damage/tipping), G419 (collusion), throwing FUEL over the wall, the CORRAL return path, robot damage. The scripted bots add a small *incidental* foul rate so foul totals match the real data. |

## 4. Physics

- **Robots:**
  - Collision circles of radius 0.46 m (bumpers included) with holonomic (swerve) motion and acceleration limits.
  - Mass-weighted pushing between robots.
  - BUMPs cap speed at 1.6 m/s.
  - TRENCH openings are walls for robots taller than 0.565 m.
- **The TOWER is an open frame, not a block:**
  - Its two **uprights** block every robot and FUEL. They are 0.82 m apart, narrower than a robot (0.92 m across its bumpers), so no robot drives in between them.
  - Its **rungs** (1.19 m long, the LOW RUNG's underside 0.665 m up) block robots taller than that, but FUEL rolls under them.
  - Its **supports** back to the wall start 0.721 m up. A robot shorter than that can drive along the wall behind the uprights, over the floor plate.
  - So each robot has a **clearance class**: 0 fits under everything, 1 is stopped by the TRENCH arms, 2 also by the rungs, 3 also by the supports. Every tier is class 0 (0.55 m tall) or class 2 (0.70 m).
- **Intake:** a rectangle in front of the robot (its width and reach come from the spec) collects FUEL at the spec's rate up to hopper capacity.
- **Shooting:**
  - Auto-aimed at your own HUB within `min_range`–`max_range`. Robots without a turret must face the HUB within ±15°.
  - Hit probability falls with distance beyond `sweet_range` and with robot speed.
  - Flight time is 0.45 s + 0.11 s per meter.
  - Misses land around the HUB, long or short.
- **FUEL:**
  - All 504 FUEL are individual 7.5 cm-radius particles with rolling friction.
  - They bounce off walls and field elements, and robots bulldoze them.
  - A grid-based contact pass keeps them from overlapping.
  - Scored FUEL leaves through 4 exits on the HUB's neutral-zone face, which closes the resource loop.
- **Jams:** mechanisms occasionally jam while in use, per the spec's `jam_rate`. A jammed robot can still drive but can't intake or shoot.
- **Climbing:** stop at one of **three climbing positions in front of your TOWER's uprights** and request a level. It takes `climb_time`, succeeds with `climb_success`, and the robot stays put until it drives off.
  - The positions are the middle of the rungs and 0.95 m to each side, where a robot reaches the 5.875 in (0.15 m) of rung that sticks out past each upright.
  - Three robots side by side touch each other, which Game Manual 6.5.2 allows while climbing. The manual credits up to 2 robots in AUTO and needs 50 TOWER points for TRAVERSAL, so at least two must fit.
- **Navigation** (macro actions and bots):
  - Robots cross each HUB line through a TRENCH (short robots only) or over a BUMP, avoiding lanes a dead or climbing robot blocks.
  - They detour around the TOWER's front when moving along the wall between DEPOT and OUTPOST. A robot already hugging the wall passes behind the uprights instead, if it is short enough for the supports.
  - A robot that pushes for a second without moving backs off sideways and re-plans. In 60 test matches, no robot stalled for more than 1.4 s.
  - The bots leave FUEL that has rolled into a TOWER alone: no robot fits between the uprights.
  - At the DEPOT, a robot drives its intake up to the wall once the FUEL left there is out of reach from its usual spot.

## 5. Robot tiers, bots and calibration

| Tier | Speed | Hopper | Intake/s | Shots/s | Accuracy (sweet range) | Max range | Turret | TRENCH | Climb |
|---|---|---|---|---|---|---|---|---|---|
| elite | 4.5 | 44 | 7.0 | 5.5 | 0.90 (3.2 m) | 6.5 m | yes | yes | – |
| strong | 4.0 | 18 | 3.2 | 2.2 | 0.65 (2.4 m) | 4.5 m | no | yes | – |
| mid | 3.6 | 10 | 2.0 | 1.4 | 0.50 (2.0 m) | 3.5 m | no | no | – |
| low | 3.0 | 6 | 1.2 | 0.8 | 0.35 (1.6 m) | 2.8 m | no | no | – |
| climber | as mid | | | | | | | | L1, 6 s, 50% |
| elite_climber | 4.4 | 40 | 8.0 | 6.0 | 0.90 (3.0 m) | 6.0 m | yes | yes | L3, 7 s |
| broken | – | | | | | | | | doesn't move |

Scripted drivers copy the Day 1 habits:

- **Elite and strong:** sweep the center FUEL in AUTO, then run hoard-and-dump.
- **Mid:** shoots its preload, then raids the DEPOT.
- **Low:** only shoots its preload.
- **Weaker teams:**
  - often ignore the HUB schedule (`awareness`), wasting FUEL into an off HUB
  - hesitate (`distraction`)
  - jam more often

**Calibration** (`python scripts/calibrate.py --matches 96`, random Day 1 robot mix):

| | Simulated | Real Day 1 |
|---|---|---|
| FUEL per robot: elite / strong / mid / low | 279 / 58 / 25 / 7 | ~225–250 / 40–70 / 15–30 / <15 |
| Elite + strong + strong alliance, FUEL (AUTO) | 403 (66), over 40 matches | 393, 394, 409 (73–78) |
| AUTO share of FUEL | 20.6% | 19.8% |
| Foul awards per alliance | 0.28 | 0.26 |
| TOWER points per alliance | 0.47 | 0.48 |
| Median winning margin | 32 | 42 |

The elite tier runs a little above the per-robot estimate for team 9483. That estimate comes from noisy practice data, so the tier is tuned to reproduce the real elite alliance totals instead.

**Alliance FUEL distribution:**

| | Simulated | Real Day 1 |
|---|---|---|
| Median | 55 | 34 |
| Mean | 79 | 87 |
| Alliances at 100+ FUEL | 17% | 24% |

The real distribution is more extreme because team 9483 played in a third of the Day 1 matches. Change `DAY1_TIER_WEIGHTS` in `robot.py` to model a later, stronger event.

## 6. The RL interface

- **Episode:** one full match.

| Mode | Decision interval | Decisions per match |
|---|---|---|
| Macro | 0.25 s | 664 |
| Continuous | 0.1 s | 1660 |

- **Observation:** 162 floats, roughly in [-1, 1]. Each robot sees the field as if it were blue (red is rotated 180°), so one policy can play either side.

| Slice | Contents |
|---|---|
| 0–16 match | time, period one-hot, own/opp HUB active, **seconds until own HUB toggles**, AUTO-winner flag, scores, FUEL counts |
| 17–35 self | position, heading, velocity, FUEL held (absolute and fraction), in own zone, can shoot now, climb state, jammed, HUB distance, and robot capabilities (capacity, TRENCH, turret, climb level, shot rate, speed) |
| 36–75 robots | 2 teammates then 3 opponents, nearest first: relative position, velocity, FUEL held, climbing, mobile, distance |
| 76–147 FUEL map | 12 × 6 grid of FUEL counts on the floor |
| 148–157 nearest FUEL | relative position of the 5 closest FUEL |
| 158–161 stock | own/opponent DEPOT FUEL, own/opponent OUTPOST chute FUEL |

- **Actions:**
  - `macro` is `Discrete(8)`: IDLE, COLLECT, COLLECT_DEPOT, COLLECT_OUTPOST, SHOOT, STAGE, DEFEND, CLIMB. The controller handles the driving.
    - SHOOT only fires from legal spots.
    - STAGE waits at a shooting spot without firing.
    - DEFEND backs off before a pin becomes a foul.
  - `continuous` is `Box(6)` in [-1, 1]: vx, vy (own-alliance frame, times max speed), turn rate, intake, shoot, and climb (each > 0 means on). Nothing stops the robot from fouling, so it has to learn the rules.
- **Reward** (`RewardConfig`, per step):

| Term | Default |
|---|---|
| Change in the alliance score margin | 0.1 per point |
| FUEL collected (shaping) | 0.005 |
| FUEL this robot put into an inactive HUB | −0.05 |
| End of match | ±1 for a win or loss, plus 0.5 per bonus ranking point |
| This robot's own scored FUEL (`self_fuel`) | 0 |
| This robot's own fouls (`own_foul`) | 0 |

  Fouls already cost the margin, because the opponent gets the points.
- **`EnvConfig` options:**
  - `learner_tiers`: which robot you drive, sampled each match
  - `learner_slots`: which seat
  - `bot_tier_weights`: the opponent/teammate mix
  - `event_level`: sets the RP thresholds
  - `decision_dt`
  - `random_starts`

## 7. Training recipes

Throughput measured on this PC with only 4 parallel envs:

| Mode | Throughput (4 envs) | 16-env estimate | Run |
|---|---|---|---|
| Macro PPO | 433 steps/s | ~1,500/s | 3M steps ≈ 30–40 min |
| Continuous PPO | 909 steps/s | ~3,000/s | 20M steps ≈ 2 h |
| Self-play | 2,600 decisions/s | ~9,000/s | 20M decisions ≈ 40 min |

The 16-env numbers are estimates; throughput should scale roughly with cores.

1. **Start with macro strategy on one robot type:**

   ```bash
   python scripts/train_ppo.py --mode macro --tiers strong --timesteps 3000000 --name strategy
   ```

   Then compare it with a typical team's driving on the same matches:

   ```bash
   python scripts/evaluate.py runs/strategy/final_model.zip --tier strong
   ```

   In the Day 1 mix the scripted "strong" driver wins 77% ± 4% of its matches (100-match check), with a +37 average margin and 59 FUEL, so the margin change is the number to watch. Things worth checking in `watch.py`:
   - Does it hoard while its HUB is off?
   - Does it stage just before the switch?
   - Does it stop wasting FUEL?
2. **Generalize:** train with `--tiers elite strong mid climber` so one policy learns to use whatever robot it has. The capabilities are in the observation.
3. **Self-play:** run `train_selfplay.py` for 3v3 coordination. Watch `vs_bots/win` and `vs_bots/margin`. If self-play drifts into odd strategies that lose to bots, raise `--vs-bots-frac`.
4. **Continuous control:** once macro strategy works, `--mode continuous` learns the driving too. Expect it to need far more steps; `--pickup-reward 0.01` helps it discover the intake early.
5. **Things worth trying:** turn `pickup` shaping off after the first million steps, use the DCMP/CMP thresholds (`--event-level`), add `self_fuel` credit for individual scoring, or change `bot_tier_weights` to face stronger alliances.

## 8. Tests

`python -m pytest -q` runs the 43 tests below, plus step 3's high-fidelity tests ([03-driving-and-aiming.md](03-driving-and-aiming.md) §11):

- **Broadcast cases** (the section 5.6 tests):
  - the AUTO winner is inactive first (P12)
  - 99 FUEL earns no ENERGIZED (P18)
  - the 3 s grace at 2.9 s vs 3.1 s
  - 90 s of active time for each alliance
- **Geometry:**
  - the HUB, TOWER and OUTPOST checked against the AprilTag positions
  - the open-frame TOWER: robots meet the uprights or pass under the rungs by height, pass behind the uprights along the wall, and FUEL rolls under the rungs
  - every climbing position counts as being at the TOWER
- **Physics invariants:**
  - FUEL is conserved (504)
  - identical seeds give identical matches
  - no points from an inactive HUB
  - G407 is enforced
  - robots stay in bounds
- **API conformance:** Gymnasium's `check_env` for both action modes, and PettingZoo's `parallel_api_test`.
- **Regression tests from the code review:**
  - a robot starting near a HUB line crosses it
  - a dead robot on a BUMP lane gets routed around
  - DEPOT ↔ OUTPOST drives go around the TOWER
  - three robots can climb and earn TRAVERSAL
  - robot types given by seat are honored
  - observations stay within [-2, 2]
  - stacked FUEL separates
- **Regression tests from the geometry fix:**
  - FUEL pushed back against the DEPOT's wall still gets collected
  - FUEL inside the TOWER doesn't trap a collecting robot

## 9. What comes next

- **Driving and aiming track:** ✅ built in step 3 as a high-fidelity physics mode. See [03-driving-and-aiming.md](03-driving-and-aiming.md):
  - a 0.02 s physics step
  - swerve modules with motor and traction limits
  - latency and a pose estimate from odometry and AprilTags
  - rectangular bumpers
  - 3D FUEL ballistics with exit speed, hood and turret as actions
- **Vision track:**
  - a 3D render of the same match state to produce labeled images (FUEL, robots, AprilTags)
  - a detector/localizer trained on your RTX 3070

  Free about 8 GB of disk first for CUDA PyTorch.
