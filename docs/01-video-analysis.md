# Step 1: Video analysis of FRC 2026 *REBUILT* for an ML training game

**Source video:** [2026 İstanbul Regional, Day 1](https://www.youtube.com/watch?v=zKZk-Y1703E), FIRSTRoboticsCompetition channel, livestream of 3 Mar 2026, 9 h 27 min. Event code `2026TUIS`. The FTA calls it "the first event of the whole season."
**Game:** *REBUILT presented by Haas* (2026 season *FIRST AGE*).
**Cross-check:** Official 2026 Game Manual, final version TU22 ([PDF](https://firstfrc.blob.core.windows.net/frc2026/Manual/2026GameManual.pdf)). Rule and section numbers below come from that manual. Anything tagged **(video)** was measured from the broadcast.

---

## TL;DR

- The video covers all of Day 1: drivers' meeting, field calibration, then **21 scored matches** (18 practice, 3 filler) that I tracked end to end.
- **The game:** 3v3. Robots collect **FUEL** (15 cm foam balls, 504 per match) and shoot them into their alliance's **HUB** for 1 point each. They also climb a **TOWER** for 10, 20 or 30 points. The twist is that in teleop the two HUBs take turns being active in **25-second SHIFTs**. FUEL scored into an inactive HUB is worth nothing.
- **What actually decides matches (video):** elite alliances score about 0 while their HUB is off. Meanwhile they hoard FUEL, then unload **100–165 FUEL in a 25 s active SHIFT**, peaking at 6–8 FUEL/s. The top alliances scored 271–409. The median alliance scored only about 34 FUEL.
- **Why it suits ML:**
  - The objective changes over time (the HUB schedule).
  - FUEL circulates in a closed loop: scored FUEL comes back out into the middle of the field.
  - Payoffs are delayed: hoard now, score 25 s later.
  - It is naturally 3v3 multi-agent.
  - It can be simulated cheaply in 2D.
- **Recommendation:** a 2D top-down simulator exposed through the Gymnasium and PettingZoo interfaces. At its core sits an exact, unit-tested rules engine covering the clock, SHIFTs, 3-second grace windows, scoring, ranking points and key fouls. Calibrate it against the data in `data/`.

---

## 1. What was analyzed and how

| Pass | What | Coverage |
|---|---|---|
| Transcript | Full YouTube transcript (English and Turkish mixed; the auto-captions mislabel it as Danish) | Whole video. Almost all speech is the drivers' meeting (0:03–0:54). After that there are only "3-2-1 başla" ("go") countdowns and a few announcements. There is no play-by-play commentary. |
| Storyboard sweep | YouTube storyboard thumbnails, 1 frame per 10 s | All 3,404 frames (whole video) |
| Full-res frames | 1080p frames pulled at chosen timestamps | 171 frames: every results screen, 5 complete match timelines, AUTO and END GAME close-ups, empty-field shots |
| Rules | Official manual TU22, extracted by a research sub-agent | Timing, scoring, RP, fouls, geometry, FUEL, robot limits |

**Data produced:**
- `data/istanbul2026_day1_match_results.csv`: all 21 scored matches with the full breakdown (AUTO/TELEOP FUEL, TOWER, foul points, ranking points).
- `data/istanbul2026_day1_score_timelines.csv`: live scoreboard samples every 3–10 s for 5 high-scoring matches (P3, P8, P12, P15, P20).

---

## 2. Timeline of the video

| Stream time | What happens |
|---|---|
| 0:00–0:05 | Overhead field camera; arena filling |
| 0:03–0:54 | **Drivers' meeting** on the field, in English and Turkish (details below) |
| 0:35–0:55 | Alternate low-angle field camera with no overlay |
| 0:55–1:06 | REBUILT / İstanbul Regional holding slide |
| 1:06–2:28 | **Field open for measurement and calibration**: teams on the field with robots and laptops (vision/AprilTag setup) |
| 2:28–2:38 | "Up Next: Pratik 1 / 22" |
| 2:38–2:40 | Broadcast glitch: a Windows desktop goes on air, so **Practice 1 is not shown** |
| 2:40–8:06 | **Practice 1–20** (P5, P6 and P21 never appear) and **3 filler matches** ("Deneme Maçı"). Each match cycle takes about 12–17 min. **P22 was stopped by a field fault** with 0:28 left (7:01, "once the field error is resolved…") and the field was reset under purple lights. |
| 8:06–9:27 | Field lighting tests (red, blue, purple, gold) and idle field. No more matches. |

**Match cycle as broadcast:**
1. "Up Next" card with team names.
2. Field shot with the clock at 0:20.
3. AUTO, then TELEOP.
4. Post-match hold, then field reset. Purple lights and "CLEAR" on the timer mean field staff are clearing FUEL.
5. Results card, then the next "Up Next".

**Drivers' meeting points that matter for a sim:**
- Robots for the next match wait behind their alliance's end of the field; queueing runs through the Nexus app.
- The A-STOP (right side of the shelf) cancels AUTO only; the E-STOP (left side) disables the robot for the whole match.
- Practice ran 12:00–17:30, then filler-line matches until about 18:30.
- Referee signals: MINOR foul is a flag wave; MAJOR is a flag wave plus crossed arms; a pin count is a chopping motion. Yellow and red cards were explained.
- People crossing the field should use the open TRENCHes; the BUMPs are slippery.

**Turkish overlay terms:**

| Turkish | Meaning |
|---|---|
| KIRMIZI | red |
| MAVİ | blue |
| OTONOM | AUTO |
| UZAKTAN KONTROL | TELEOP |
| YAKIT | FUEL |
| KULE | TOWER |
| PENALTI | foul points received |
| SIRALAMA PUANI | ranking points |
| KAZANAN | winner |
| Pratik | practice |
| Deneme Maçı | trial/filler match |

---

## 3. The game: rules a simulator must reproduce

### 3.1 Field

The field is 651.2 × 317.7 in (**16.54 × 8.07 m**), carpeted and enclosed by guardrails. The layout is roughly point-symmetric. The diagram below is schematic; exact positions will come from the official field drawings.

```
                        far side (scoring table in the broadcast)
 ┌──────────────┬────────┬───────────────────────────────┬────────┬──────────────┐
 │ RED          │ TRENCH │                               │ TRENCH │         BLUE │
 │ DEPOT        │ BUMP   │                               │ BUMP   │      OUTPOST │
 │ TOWER (wall) │ [HUB]  │   ▓▓▓▓ ~360 FUEL block ▓▓▓▓   │ [HUB]  │ TOWER (wall) │
 │              │ BUMP   │          CENTER LINE          │ BUMP   │        DEPOT │
 │ OUTPOST      │ TRENCH │                               │ TRENCH │              │
 └──────────────┴────────┴───────────────────────────────┴────────┴──────────────┘
  red ALLIANCE WALL        NEUTRAL ZONE                   blue ALLIANCE WALL
  (driver stations)                                          (driver stations)
```

| Element | Key numbers (manual) | Notes (video) |
|---|---|---|
| ALLIANCE ZONE | 158.6 in (4.03 m) deep, full width | Robots must be (at least partly) in their own zone to shoot (G407) |
| NEUTRAL ZONE | 283 in (7.19 m) deep, split by the CENTER LINE | Starts with the big FUEL block |
| HUB (1 per alliance) | 47 × 47 in footprint; hexagonal opening 41.7 in across; opening lip **72 in (1.83 m)** high; centered **158.6 in from own alliance wall** | Light ring shows the alliance color when active. A tall clear backboard sits above it. AprilTags on every face. **Scored FUEL leaves through 4 exits at the base into the NEUTRAL ZONE, at random.** The video shows FUEL piling up on the neutral side of the HUB during heavy scoring. |
| BUMP (2 per HUB) | 73.0 × 44.4 in, **6.5 in tall**, 15° ramps | Drivable ramps on either side of the HUB |
| TRENCH (2 per alliance) | Clear opening **22.25 in tall × 50.34 in wide** | Between the BUMP and the guardrail. Only robots 22.25 in or shorter fit under it. |
| TOWER | 49.25 W × 45 D × 78.25 in tall, set into the alliance wall between driver stations 2 and 3. Rungs at **27 / 45 / 63 in** (LOW, MID, HIGH). | Ladder-like frame, clearly visible in the video |
| DEPOT | 42 × 27 in floor area along the alliance wall, **24 FUEL** | Often emptied early (video) |
| OUTPOST | Human-player station in a corner. A sloped CHUTE holds about 24 FUEL behind a door; a CORRAL sits at the base. | Human-player feeding was not clearly visible on the broadcast |

The three depth figures (alliance zone 158.6 in, HUB center 158.6 in from the wall, neutral zone 283 in) don't add up exactly to 651.2 in. **Before building collision geometry, take exact coordinates from the official field drawings** or the WPILib 2026 AprilTag field layout.

**AprilTags:** 32 tags (36h11 family), 8.125 in. 16 on the HUBs (center height 44.25 in), 4 on the TOWER walls, 4 on the OUTPOSTs, 8 on the TRENCHes. This only matters if you later add a vision or localization layer.

### 3.2 FUEL
- 5.91 in (**15.0 cm**) high-density foam ball, 0.20–0.23 kg.
- **504 per match:**
  - 24 in each DEPOT (48 total)
  - 24 in each OUTPOST CHUTE (48 total)
  - Up to **8 preloaded per robot**
  - The rest, about 360, sits in a ~206 × 72 in block in the NEUTRAL ZONE, split at the center line.
- **There is no possession limit during the match** (§5.10). Robots with huge hoppers are legal.

### 3.3 Match timing and HUB shifts

| Period | Clock | HUB status | Broadcast indicator (video) |
|---|---|---|---|
| AUTO | 0:20 → 0:00 (20 s) | both active | yellow arrows on both sides |
| *(AUTO → TELEOP hold)* | clock holds at 2:20 for about 3 s (video) | FUEL still in flight is credited to AUTO | |
| TRANSITION | 2:20 → 2:10 (10 s) | both active | `1/6 :ss` |
| SHIFT 1 | 2:10 → 1:45 (25 s) | **the AUTO loser's HUB only** | `2/6 :ss`, one arrow |
| SHIFT 2 | 1:45 → 1:20 | the AUTO winner's HUB only | `3/6` |
| SHIFT 3 | 1:20 → 0:55 | the AUTO loser's | `4/6` |
| SHIFT 4 | 0:55 → 0:30 | the AUTO winner's | `5/6` |
| END GAME | 0:30 → 0:00 (30 s) | both active | `6/6` |

- **The alliance that scores more FUEL in AUTO has its HUB turned off first, in SHIFT 1.** A tie in AUTO is broken at random by the FMS (§6.4.1). This was confirmed in every match I checked (video).
- **3-second grace window:** FUEL still counts for 3 s after a HUB deactivates, and for 3 s after the clock hits 0:00 at the end of AUTO and at the end of the match (§6.5). This is visible on the broadcast (video):
  - P12: blue's count rose from 208 to 223 in the first 3 s of its inactive SHIFT 3. That was FUEL shot at the end of SHIFT 2 that was still being counted.
  - Final scores beat the live score at 0:00 by up to 22 FUEL.
- **Signals to drivers:**
  - An FMS game-data message at the start of TELEOP tells each driver station which alliance won AUTO.
  - The HUB light bars show the alliance color when active and give a 3-second warning pattern before switching off.
- **The consequence both alliances plan around:** each alliance gets exactly **90 s of active HUB time in TELEOP**, but distributed differently.
  - AUTO winner: TRANSITION, SHIFT 2, then **SHIFT 4 plus END GAME as one continuous 55 s window** at the end.
  - AUTO loser: **TRANSITION plus SHIFT 1 as one continuous 35 s window** at the start, then SHIFT 3 and END GAME.

### 3.4 Scoring and ranking points (Table 6-4 / 6-5)

| Action | AUTO | TELEOP |
|---|---|---|
| FUEL into an **active** HUB | 1 | 1 |
| FUEL into an **inactive** HUB | 0 | 0 |
| TOWER LEVEL 1 (off the carpet), per robot | **15** (max 2 robots) | 10 |
| TOWER LEVEL 2 (bumpers above the LOW rung) | none | 20 |
| TOWER LEVEL 3 (bumpers above the MID rung) | none | 30 |

There is **no leave or mobility bonus** this year.

| Ranking point | Regional / district | District Championship | World Championship |
|---|---|---|---|
| ENERGIZED: active-HUB FUEL at or above | **100** | 240 | 360 |
| SUPERCHARGED: active-HUB FUEL at or above | **360** | 360 | 500 |
| TRAVERSAL: TOWER points at or above | **50** | 50 | 50 |
| Win / tie | 3 / 1 | same | same |

The maximum is 6 RP per qualification match. **Video check:** P18 red scored 99 FUEL and did *not* get ENERGIZED. The overlay counter reads `N / 100`, then switches to `N / 360` after the first threshold.

### 3.5 Fouls and the rules that shape strategy

- A **MINOR foul** gives 5 points to the opponent; a **MAJOR** gives 15. The video showed awards of 5, 15 and **45**. The 45 is almost certainly three stacked MAJORs, for example an escalating pin (G418).
- These rules should be enforced by the sim, or at least penalized in the reward:

| Rule | Effect in the sim |
|---|---|
| **G407** | You may only launch FUEL into your own HUB while your bumpers are at least partly in your own ALLIANCE ZONE. **This defines where shooting is legal.** |
| **G403** | In AUTO, a robot may not fully cross the CENTER LINE (MAJOR, plus a MAJOR per contact). |
| **G408** | No catching or controlling FUEL as it comes out of the HUB (no camping at the HUB exits). |
| **G405** | No intentionally sending FUEL out of the field. |
| **G418** | Pinning for more than 3 s: a MINOR, then a MAJOR for every further 3 s. The count resets after the robots separate by 72 in for 3 s. |
| **G419** | Two or more robots may not jointly lock down key elements (both TRENCHes, both BUMPs, and so on). |
| **G420** | In END GAME, no contact with an opponent that is touching its TOWER. Penalty is a MAJOR, and the victim is awarded LEVEL 3 if it was off the ground. |
| **G415–G417** | Damaging, tipping or entangling opponents. |
| **G425 / G427** | Human players may only feed FUEL through the CHUTE, through the OUTPOST opening, or by throwing it over the alliance wall. |

The manual research found no rule against knocking FUEL into the opponent's side. That is a "not found", not a confirmed "allowed".

### 3.6 Robot constraints
- Weight: 115 lb, or 135 lb with bumpers.
- Frame perimeter at most 110 in.
- **Height at most 30 in at all times.** A robot has to be 22.25 in or shorter to use the TRENCH, so every robot faces a design choice: fit under the TRENCH or go over the BUMP.
- Extension at most 12 in beyond the frame, on one side at a time.
- Bumper zone: 2.5–5.75 in above the floor.

---

## 4. What happened on the field (Day 1 data)

### 4.1 Results of all 21 scored matches (`data/istanbul2026_day1_match_results.csv`)

**FUEL per alliance (42 alliance results):**

| Statistic | Value |
|---|---|
| Median | **33.5** |
| Mean | 87 |
| Maximum | **409** |

The distribution is strongly skewed:

| Group | Alliance FUEL totals |
|---|---|
| Most alliances | 0–40 |
| Middle group | 62–125 |
| Top cluster | 271–409 |

- **Every one of the 7 alliances with 250+ FUEL included team 9483.**
- **Ranking points:**
  - ENERGIZED (100 FUEL) was reached by 10 of 42 alliances.
  - SUPERCHARGED (360 FUEL) was reached by 3.
  - TRAVERSAL (50 TOWER points) was **never** reached.
- **AUTO** produced 20% of all FUEL. The six best AUTOs scored 52–78 FUEL, all by alliances that included 9483. Its seventh alliance, P3, managed only 21 in AUTO.
- **TOWER:** only 2 climbs all day, both LEVEL 1 and both by team 10428 (P18 and filler match 2). LEVEL 2 and 3 were never attempted.
- **Fouls:** 11 of 42 alliances received foul points (6 × 5, 4 × 15, 1 × 45). Live scores sometimes changed after the match: foul points were revised and TOWER points were added once the referees finished scoring.
- **AUTO leader and winner:** in 15 of the 20 matches where AUTO wasn't tied, the alliance that led AUTO won. This is confounded, since strong alliances win both.
- **Winning margin:** median 42. Four matches were decided by 10 points or fewer (margins of 1, 2, 8 and 10).

### 4.2 Anatomy of a 400-point match (`data/istanbul2026_day1_score_timelines.csv`)

All five elite alliances won AUTO, so their HUB was **off in SHIFT 1 and SHIFT 3** and on in SHIFT 2 and in the combined SHIFT 4 + END GAME window. The table shows FUEL scored per window.

**How counts are assigned to windows:** a HUB can only score while active or during its 3 s grace. So any rise in the counter that first shows up early in an inactive SHIFT is credited to the active SHIFT just before it, as grace or counting lag. Samples are about 10 s apart, so the exact split at a boundary is approximate. The counter rise first seen after SHIFT 2 ended, and credited to SHIFT 2, was:

| Match | Rise credited back to SHIFT 2 |
|---|---|
| P3 | +16 |
| P8 | +10 |
| P12 | +15 (seen within 3 s) |
| P15 | up to +29 (sample straddles the boundary) |
| P20 | +28 |

| Match (alliance) | AUTO | TRANSITION (on, 10 s) | SHIFT 1 (off) | SHIFT 2 (on, 25+3 s) | SHIFT 3 (off) | SHIFT 4 + END GAME (on, 55 s + grace) | Final |
|---|---|---|---|---|---|---|---|
| P3 (blue) | 21 | 0 | 0 | +100 | 0 | +150 | 271 |
| P8 (red) | 73 | 0 | 0 | +101 | 0 | +219 | 393 |
| P12 (blue) | 75 | +5 | 0 | +143 | 0 | +186 | 409 |
| P15 (red) | 69 | 0 | +4 (grace) | +115 | 0 | +154 | 342 |
| P20 (red) | 78 | 0 | 0 | +165 | 0 | +136 | 379 |

**Rates for simulator calibration:**

| Measurement | Rate |
|---|---|
| Short bursts, alliance-wide (P12 80 → 172 in about 14 s; P20 101 → 177 in 10 s) | **6–8 FUEL/s** |
| Across a SHIFT 2 window | 3.6–5.9 FUEL/s |
| Across the 55 s closing window | 2.3–3.8 FUEL/s |

The drop in the last window shows that **scoring is limited by collecting FUEL, not by shooting.** The first active window unloads a stockpile; later ones depend on gathering FUEL again. P20 flatlined at 360 for about 20 s of END GAME.

Minute by minute, P12 went like this (blue: 6431, 9483, 9077):

| Clock | Phase | Blue score |
|---|---|---|
| AUTO | Sweep the center pile, return to own zone, shoot | 0 → 58 with 2 s left; 75 once the grace window closed |
| 2:20–2:10 | TRANSITION (both on) | 75 → 80 |
| 2:10–1:45 | SHIFT 1 (blue off): hoard; blue robots gather FUEL in the neutral zone | stays 80 |
| 1:45–1:20 | SHIFT 2 (blue on): unload | 80 → 208, then 223 during grace |
| 1:20–0:55 | SHIFT 3 (off): hoard again | stays 223 |
| 0:55–0:00 | SHIFT 4 + END GAME (on) | 223 → 381 at 0:13, 398 at 0:00, **409 final** |

### 4.3 Behaviors observed
1. **Elite AUTO (P20, team 9483):**
   - Starts beside its HUB.
   - Drives into the NEUTRAL ZONE and scoops dozens of FUEL from the center block in about 4 s. It stays on its own side of the CENTER LINE (G403).
   - Backs into its own alliance-zone corner.
   - Lobs long arcs into the HUB. The alliance went 5 → 78 FUEL in the last about 9 s of AUTO, counting the grace window.
2. **Hoard and dump.** While their HUB is off, strong robots fill their hoppers and gather or push FUEL. They score nothing. When the HUB turns on they fire continuously. The first 10 seconds of an active window are the most productive part of the match.
3. **Shooting positions.** In the frames I checked, robots shot from inside their own alliance zone, as G407 requires: either right next to the HUB or from deep corners 4–5 m away, using high arcs.
4. **FUEL flow.** The center block gets bulldozed into streaks within seconds. Scored FUEL comes out on the neutral-zone side of the HUB. Over a match, FUEL piles up against the alliance walls and corners; likely causes are misses, rebounds, or human-player input, but I didn't confirm which. DEPOTs were often emptied early.
5. **Climbing is almost unused.** Only 10428 climbed (LEVEL 1, in the last about 10 s). A TRAVERSAL RP needs 50 points, for example two LEVEL 3 climbs or a LEVEL 3 plus a LEVEL 2. **Nobody attempted it, so a learning agent has a large untapped source of points here.**
6. **Low-tier robots** (0–20 FUEL per match) mostly drove around, pushed FUEL, or sat still. P1 red and P4 blue scored 0.

### 4.4 Robot skill tiers (rough OPR from the Day 1 FUEL data)

These estimates are least-squares with ridge regularization over 42 alliance results and 34 teams. Practice data is noisy (substitutions, 1–11 matches per team), so treat them as **tiers, not ratings**.

| Tier | Estimated FUEL per match | Teams |
|---|---|---|
| Elite | about 225 (AUTO about 40) | 9483 |
| Strong | 40–70 | 6431, 9077, 4481, 3646 (11281, 1 match) |
| Mid | 15–30 | about 12 teams |
| Low | 15 or less | the rest, about half the field |

**This spread should drive the sim's opponent pool.** A realistic early-season match is usually one strong robot plus two weak ones.

---

## 5. Turning REBUILT into a game for ML

### 5.1 First decide what the ML is for
| If the ML project is… | Fidelity needed | Recommended sim |
|---|---|---|
| **Strategy or decision-making** (which task, when to hoard, when to climb, 3v3 coordination) | Low. Top-down 2D with abstracted shooting. | **Custom 2D Python simulator (recommended starting point)** |
| **Autonomous routines** (the best 20 s AUTO path, intake/shoot sequencing) | Medium. Accurate kinematics and timing. | Same 2D simulator plus path optimization. Output converts to PathPlanner or Choreo paths. |
| **Driving or control policy** (a learned swerve controller, sim-to-real) | High. Dynamics, latency, noise. | 2D simulator with a realistic swerve model, or a physics engine (MuJoCo/Isaac) |
| **Perception** (FUEL detection, AprilTag localization) | 3D rendering | Unity ML-Agents, Isaac Sim, or Godot using the official CAD |

### 5.2 Recommended architecture (v1)
- **Language and tooling:** Python with NumPy (vectorizable, later portable to JAX).
- **Interfaces:** a Gymnasium env for single-agent work and a PettingZoo ParallelEnv for 3v3. Pygame renders the game so humans can play it and so you can debug.
- **Physics:** 50 Hz, the same 20 ms loop as WPILib. Agents act at 5–10 Hz with frame-skip.
- **Rules engine (build and unit-test this first):** a pure-function state machine covering:
  - the match clock, including the 3 s hold after AUTO
  - HUB active status with the AUTO-winner rule and random tie-break
  - 3 s grace windows after every deactivation and after AUTO/match end
  - scoring and ranking points
  - fouls G407, G403, G418 and G420
- **Entities:**
  - **Robots:** rounded rectangles about 0.9 × 0.9 m, holonomic (swerve) kinematics with acceleration limits. Parameters: `max_speed` (~4–5 m/s), `accel`, `height` (can it use the TRENCH?), `intake_width`, `intake_rate`, `capacity`, `shoot_rate`, `accuracy(d)`, `min/max_range`, `climb_level`, `climb_time`.
  - **FUEL:** about 500 disk particles (r = 7.5 cm) with friction. Robots can bulldoze them. In-flight FUEL follows ballistic time-of-flight.
  - **HUB:** a sensor with processing delay. FUEL scores if it enters while the HUB is active or within the grace window. It then **exits through 4 neutral-zone exits with random velocity**, which closes the resource loop.
  - **BUMP:** passable with a speed cap and time cost.
  - **TRENCH:** height-gated, 22.25 in or less.
  - **TOWER:** a climb action with a duration and a success probability per level.
  - **DEPOT and OUTPOST:** FUEL sources; the OUTPOST is fed by a human-player model at a limited rate.

### 5.3 Observations and actions
- **Observations for each agent:**
  - The match clock and a one-hot period.
  - Own and opposing HUB active flags, and **seconds until own HUB status changes**.
  - Scores, and FUEL counts toward 100/360.
  - Own pose, velocity, FUEL carried, and whether it is in its own zone.
  - Teammate and opponent poses, velocities and estimated loads.
  - A FUEL density grid (for example 64 × 32 over the field), plus DEPOT and OUTPOST counts.
- **Actions (low level):** field-relative `(vx, vy, ω)` plus `intake`, `shoot` (auto-aim), and `climb(level)`.
- **Actions (high level, for hierarchical RL):** options such as `collect(region)`, `goto_shoot_spot(k)`, `unload`, `hoard`, `defend(robot)`, `climb(level)`, `raid_depot`.

### 5.4 Rewards
- **Match points:** FUEL into an active HUB earns +1, credited to the shooter; climbs earn their TOWER points.
- **Fouls:** −5 or −15 when committed. The opponent receives the points.
- **End of match:** a win bonus, plus ranking points (ENERGIZED, SUPERCHARGED, TRAVERSAL) weighted as in the real standings.
- **Light shaping:** optional bonuses for holding FUEL when your HUB is about to turn on, and a penalty for shooting at an inactive HUB, which wastes FUEL for 0 points. Anneal shaping away over training so it doesn't create reward loops.
- **3v3:** use team reward plus individual credit (for example MAPPO with a centralized critic).

### 5.5 Calibration targets from this video
| Quantity | Target |
|---|---|
| Elite robot | about 200–250 FUEL/match solo; best alliance AUTOs 52–78 (21 in an off match); 6–8 FUEL/s bursts |
| Active-window output (elite alliance) | 100–165 FUEL per 28 s (SHIFT 2); 136–219 per 55 s closing window |
| Inactive-window output | about 0 new FUEL. The counter keeps rising for a few seconds after the switch from FUEL shot just before it (P12: +15 within 3 s). |
| Alliance FUEL distribution | median about 34, mean about 87, max about 410; about 25% of alliances reach 100 FUEL |
| Climbing (early season) | rare, LEVEL 1 only; TRAVERSAL never reached |
| Fouls | about 0.26 foul awards per alliance per match, mostly 5 points |
| End-of-match grace | final score up to +22 FUEL over the live 0:00 score |

### 5.6 Tests the sim must pass (taken from the video)
1. AUTO FUEL red 3 vs blue 75 gives **red active in SHIFT 1**, then alternation (P12).
2. FUEL entering 2.9 s after deactivation scores; at 3.1 s it doesn't.
3. 99 active-HUB FUEL does not earn ENERGIZED; 100 does (P18).
4. An alliance of elite scripted bots reproduces the *shape* of P12 and P20: flat while off, steep while on, flattening late as FUEL runs short.
5. Total FUEL is conserved (504) across the field, robots, HUB processing and exits.

### 5.7 Training plan
1. **Scripted baseline bots** in three tiers (elite, strong, low) using the observed heuristics: sweep in AUTO, hoard while off, unload while on, climb at about 0:15. These give opponents and a benchmark.
2. **Single-agent curriculum** for one robot against scripted opponents:
   - collect FUEL
   - score with the HUB always on
   - the full SHIFT schedule
   - opponents, defense and fouls
3. **3v3 self-play or league play** (MAPPO/IPPO) with a population drawn from the skill tiers in §4.4.
4. **AUTO optimizer:** search over waypoint paths and shooting spots for the 20 s AUTO, then export to PathPlanner or Choreo.
5. **Sim-to-real:** randomize robot parameters, then validate against more event video. The overlay decoder in Appendix A makes timeline extraction automatable.

---

## 6. What the video can't tell us
- **Robot internals:** exact hopper capacities, shooter speeds and turret presence. Everything here is inferred from bursts and visuals.
- **Human-player effectiveness at the OUTPOST:** not visible on the broadcast.
- **Defense:** no clear sustained defense was observed on Day 1. Qualification and playoff days would show more.
- **Whether an inactive HUB physically passes FUEL through:** the manual doesn't say. Assume it does, unscored.
- **Mid-season strategy:** this was the first event of the season. Later events (DCMP/CMP thresholds of 240/500) will look very different, with more climbing and more defense.

## 7. Scope decided after step 1, and the proposed step 2

**Decided (2026-09-28):**
- The ML must cover **match strategy**, **driving and aiming control**, and **vision**.
- **You train the models yourself.** So the deliverables are environments, baseline training scripts, configs and docs that run on your own PC, not pre-trained models.
- Target hardware: an RTX 3070 (8 GB), an i7-13700F (16 cores / 24 threads) and 16 GB of RAM. Python 3.14 is installed with NumPy; PyTorch, Gymnasium and PettingZoo are not yet installed.

**Proposed build order (each one a separate step):**
1. **Core package:** the exact rules engine and field geometry, with the unit tests from §5.6. All three tracks share it.
2. **Strategy track:** a fast 2D simulator with tiered scripted bots and a Pygame viewer you can also play. Gymnasium and PettingZoo envs, plus a PPO/MAPPO training script you run and watch in TensorBoard.
3. **Driving and aiming track:** a high-fidelity mode in the same simulator: swerve module dynamics, motor limits, latency and sensor noise, and 3D FUEL ballistics into the 72 in HUB opening. It gets its own control envs.
4. **Vision track:** a 3D render of the same world state, used to generate labeled synthetic images for FUEL detection and AprilTag localization, plus a detector training script sized for 8 GB of VRAM. Pick a lightweight Python renderer or Unity rather than Isaac Sim, which is too heavy for 16 GB of RAM.

---

### Appendix A: Broadcast overlay decoder (for pulling more data from other event videos)
- **Top center:** match name (for example `Pratik 12 / 22`), red teams, red live score, **clock**, blue live score, blue teams.
- **Below the clock (TELEOP):** `k/6 :ss` is period *k* of 6, with *ss* seconds left in it.
- **Yellow arrow next to an alliance's counter:** that alliance's HUB is active. Arrows on both sides mean both are active.
- **Far left and far right:** FUEL counter per alliance, `N / 100`, which switches to `N / 360` after 100.
- **Live scores include foul points.** Final scores come from the results card, which shows AUTO FUEL, AUTO TOWER, TELEOP FUEL, TELEOP TOWER and foul points received, plus ranking point icons (ENERGIZED, SUPERCHARGED, TRAVERSAL, three win trophies).

### Appendix B: Files
| File | Contents |
|---|---|
| `data/istanbul2026_day1_match_results.csv` | 21 matches: teams, per-phase FUEL and TOWER, foul points received, totals, winner, ranking point flags |
| `data/istanbul2026_day1_score_timelines.csv` | Live scoreboard samples for P3, P8, P12, P15 and P20, with the source resolution of each row |
