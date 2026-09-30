# HANDOFF: FRC 2026 REBUILT simulator and ML training project

*Written 2026-09-29 at the end of the first Claude Code chat, updated 2026-09-30 after the second (step 3) and the third (step 4). For the next Claude Code chat, and for the user.*

---

## ⚠️ READ THIS FIRST: the user is in charge of training

The user's own words:

> **"When we get to the part of training the model I want to be in charge of everything and I want to learn how to setup a model and actually train it, from step 0 to 100, just like I would in the work force."**

This is the most important instruction in the project.

- **Once the project reaches model training, Claude stops being the builder and becomes the teacher.**
- The user wants a real learning experience: set up a model and train it the way an ML engineer does at work.
- Doing it *for* them defeats the purpose, even when it would be faster.

| | The USER does | CLAUDE does |
|---|---|---|
| Decisions | Picks the algorithm, the network design, hyperparameters, reward weights, and what experiment to run next | Lays out options and trade-offs, and says what professionals usually do and why |
| Code | Writes the model, training loop, configs and evaluation code | Explains concepts, points to the relevant files and docs, reviews their code, and gives hints before answers |
| Running | Types and runs every training and evaluation command | Explains what the command does and what to expect; never starts training runs |
| Results | Reads TensorBoard, diagnoses problems, writes the experiment journal | Explains what each curve means, asks "what do you think happened?", and confirms or corrects the reasoning |
| Pace | Decides when to move on | Checks understanding with short questions before moving to the next phase |

- **Never:**
  - run training for the user, even a "quick" one
  - write the training pipeline unprompted
  - silently choose hyperparameters
  - push through phases to "save time"
- **If the user explicitly asks Claude to write some training code:**
  1. Ask whether they'd like to try it with guidance first.
  2. If they still want Claude to write it, keep it small.
  3. Explain every line.
  4. Let them run it.
- **`scripts/train_ppo.py` and `scripts/train_selfplay.py` already exist.** Claude wrote them in step 2, *before* this rule was stated, as reference implementations. They are **not the plan**. The user decides whether to:
  - build their own pipeline from scratch (best for learning), or
  - dissect these scripts, or
  - use them only as an answer key to compare against.
- **The boundary:** simulator engineering can still be done by Claude when the user asks, one step at a time with their approval. That covers physics, game rules, bots, rendering, and generating vision data. Anything about **models and training** belongs to the user. When in doubt, ask.

---

## 1. Where the project stands

| Step | Status | Output |
|---|---|---|
| **1. Video analysis:** learn the game from the 2026 İstanbul Regional Day 1 livestream | ✅ Done | `docs/01-video-analysis.md`, `data/istanbul2026_day1_match_results.csv` (21 matches), `data/istanbul2026_day1_score_timelines.csv` (5 match timelines) |
| **2. Core + strategy simulator:** rules engine, field, physics, calibrated bots, Gymnasium/PettingZoo environments, viewer and playable game, reference training scripts | ✅ Done | `src/rebuilt_sim/`, `scripts/`, `tests/` (34 passing), `docs/02-simulator.md`, `README.md` |
| **Model training** (the user leads; curriculum in §4) | ⏳ Not started | No models have been trained. `runs/` is empty. |
| **3. Driving/aiming fidelity:** a high-fidelity physics mode (swerve modules, latency, pose estimation, rectangular bumpers, 3D FUEL ballistics) | ✅ Done (chat 2) | `docs/03-driving-and-aiming.md`, `HiFiConfig` in `sim.py`, `drivetrain.py`, `collision.py`, `sensors.py`, `ballistics.py`, 3 new Gymnasium ids, `--hifi` on the scripts, 62 tests |
| **4. Vision track:** 3D rendering of matches and labeled synthetic images | ✅ Done (chat 3) | `docs/04-vision.md`, `src/rebuilt_sim/vision/`, `scripts/make_dataset.py`, `export_dataset.py`, `check_dataset.py`, `render_view.py`, 85 tests |
| **Vision model training** (the user leads; outline in §4b) | ⏳ Not started | No datasets generated yet; `datasets/` is gitignored |

**Scope the user chose after step 1:**

- Covered: **match strategy**, **driving/aiming control**, and **vision**.
- Not chosen: "autonomous routine optimization".

**Order is the user's call.** The strategy and high-fidelity environments and the vision data generator are all ready, so either training curriculum (RL in §4, vision in §4b) could start now.

---

## 2. Open questions: ask the user these at the start of the next chat

1. **What next?** Steps 1–4 are built. Start the RL curriculum (§4), the vision curriculum (§4b), or fix the simulator geometry first (question 6)? Have them try `python scripts/play.py --hifi` and `python scripts/render_view.py`, and look at `docs/03-driving-and-aiming.md` and `docs/04-vision.md`.
2. **Background:** how comfortable are they with Python, NumPy, PyTorch, general ML, and reinforcement learning? Adapt depth and pace to the answer.
3. **Learning path for the training loop:**
   - **(A)** Write PPO from scratch, single-file CleanRL style. This is the most learning; they can then compare against Stable-Baselines3.
   - **(B)** Start with Stable-Baselines3 and study how it works inside.
   - **(C)** Dissect the existing reference scripts.
4. **Experiment tracking:** TensorBoard (already installed, local), or Weights & Biases (the industry-standard web dashboard; needs a free account)?
5. **Vision data choices** (only when they start vision training): which classes, visible or whole-object boxes, filters, flips (they corrupt AprilTags), dataset size, and whether to hand-label a small real test set. `docs/04-vision.md` §9 lists the trade-offs; the decisions are theirs.
6. **Simulator vs the official field drawings.** Chat 3 got the official 2026 Field Dimension Drawings. They show the simulator's TOWER and OUTPOST 0.22 m off, the DEPOT 1.07 m off, climb positions beyond the rung ends, the HUB hexagon rotated 30°, and the TOWER as an open frame, not a box (`docs/04-vision.md` §4). The 3D render already follows the drawings; the simulator was left unchanged. Options: (A) fix the simulator in its own small step, then re-run the tests and `calibrate.py` (and `--hifi`) and update docs 02/03 (recommended); (B) keep it; (C) render the simulator's geometry.

---

## 3. Getting set up (new chat or new machine)

```powershell
git clone https://github.com/sussyswimmer/FRC-game.git
cd FRC-game
uv sync --all-extras                      # creates .venv (Python 3.11, CPU PyTorch, SB3, Gymnasium, PettingZoo, pygame-ce)
.venv\Scripts\python.exe -m pytest -q     # expect: 85 passed
.venv\Scripts\python.exe scripts\play.py  # drive a robot yourself (WASD, Q/E, SPACE intake, F shoot, C climb)
```

**Machine:**

| Part | Detail |
|---|---|
| OS | Windows 11, PowerShell |
| GPU | RTX 3070 (8 GB) |
| CPU | i7-13700F (16 cores / 24 threads) |
| RAM | 16 GB |
| Disk | About 4.7 GB free on C: |

**Why CPU PyTorch:**

- The CUDA build needs about 5 GB, which the free disk space couldn't hold.
- Small MLP policies train just as fast on the CPU, since the simulator is the bottleneck.
- The **vision track will need the GPU** for training (generating the data is CPU-only). First free about 8 GB. Then change the `pytorch-cpu` index URL in `pyproject.toml` to `https://download.pytorch.org/whl/cu126`, run `uv sync --all-extras` again, and check `torch.cuda.is_available()`. The user makes this switch (`docs/04-vision.md` §11).
- **Vision datasets need disk too:** about 45 KB per image (10k images ≈ 0.45 GB). `make_dataset.py --out D:\datasets` can put them on another drive.

**Useful commands:**

| Command | What it does |
|---|---|
| `scripts\watch.py` | Watch bots, or a trained model, play |
| `scripts\play.py --hifi`, `scripts\watch.py --hifi` | The same with the high-fidelity physics (step 3) |
| `scripts\calibrate.py --matches 96` | Compare bot matches with the real Day 1 results |
| `scripts\evaluate.py MODEL --tier strong --matches 100` | Compare a model with the scripted driver in the same seat on the same matches |
| `tensorboard --logdir runs` | Training curves |
| `scripts\render_view.py --t 30` | What a robot's camera sees at 30 s, with labels drawn on |
| `scripts\make_dataset.py --name v1 --matches 250 --workers 12 [--dry-run]` | Generate a labeled vision dataset (estimate first) |
| `scripts\check_dataset.py datasets\v1 --sheet sheet.png --decode-tags 100` | Check a dataset and look at it |
| `scripts\export_dataset.py datasets\v1 --name NAME --format yolo --classes fuel --box visible --camera-robot keep` | Training files; the user picks format, classes, box style and filters |

---

## 4. The training curriculum: step 0 → 100, like a job

This is a **proposed** plan for Claude to mentor the user through. Adapt it to the answers in §2.

- Each phase has a goal, what the user does, what Claude does, and when it counts as done.
- **The user produces every deliverable.** Suggested location: `docs/training/`, a user-owned folder.
- Keep an **experiment journal** (`docs/training/journal.md`) from Phase 1 onward. Every run gets an entry: date, git commit, config, hypothesis, result, and what they learned. That is how ML teams work.

| Phase | Goal | The user does | Claude does | Done when |
|---|---|---|---|---|
| **0 (0%): Kickoff and project brief** | Frame the problem the way ML projects start at work | Answers the background questions. Writes `docs/training/00-project-brief.md`: what the model is for, the success metric, the baseline to beat, constraints (CPU, time), and risks | Explains problem framing, metrics and baselines; reviews the brief | The success metric is concrete and numeric. Example: "beat the scripted strong driver's +37 average margin by at least 20 points, 95% confidence, 200 held-out matches" |
| **1 (10%): Workstation and reproducibility** | Set up like a professional | Creates a git branch per experiment line. Verifies `uv sync` and the tests. Starts TensorBoard. Starts the journal. Decides the folder layout for runs and configs | Explains *why*: reproducibility, lockfiles, seeds, and config-as-code | The user can explain how to reproduce any run from its config plus commit hash |
| **2 (20%): Understand the environment** (the RL version of exploring your data) | Know exactly what the agent sees, does, and is rewarded for | Reads `docs/02-simulator.md` §6, `src/rebuilt_sim/obs.py` and `env.py`. Plays `play.py`. Writes a small script that runs a random policy for N matches. Runs `evaluate.py --baseline-only`. Fills in a baseline table | Explains the Gymnasium API (`reset`/`step`, `terminated` vs `truncated`), observation normalization, and discrete vs continuous action spaces. Quizzes | The journal has a baseline table: random vs scripted, with confidence intervals |
| **3 (30%): Just-enough RL theory** | Understand what training actually does | Explains concepts back in their own words. Computes a discounted return and a 3-step GAE by hand | Teaches using this game as the running example: MDP, policy, return, discount γ, value, advantage, policy gradient, PPO's clipped objective, GAE λ, entropy bonus, on- vs off-policy, why PPO here | The user can explain every term in PPO's loss and every logged metric |
| **4 (40%): Design the model** | Build the policy/value network | Writes the PyTorch module: 162 inputs (170 with the high-fidelity physics), hidden layers, output head (categorical for the 8 macro actions, Gaussian for continuous). Tests shapes. Counts parameters | Explains the options: layer sizes, activations, shared vs separate actor/critic, initialization. Reviews | A forward pass works on real observations, and the user can justify each design choice |
| **5 (50%): Build the training pipeline** | A working PPO loop the user understands end to end | Writes, piece by piece: vectorized envs, rollout buffer, GAE, PPO update, logging, checkpoints, config file, seeding. Unit-tests GAE with hand-computed numbers | Guides one component at a time and reviews each piece. Explains SB3's equivalents if they chose path B | A ~50k-step smoke run works end to end: it logs to TensorBoard, and a checkpoint saves and reloads |
| **6 (60%): First real run and debugging** | Learn to read curves and debug RL | Runs the first real training (e.g. 1–3M steps, macro actions, "strong" robot). Watches reward, episode length, entropy, approx_kl, clip fraction, explained variance, value loss, and the match stats | Explains what healthy and unhealthy curves look like, and the RL debugging checklist: reward scale, advantage normalization, learning rate, entropy collapse, reward hacking | The user can explain every curve in their run and what it says |
| **7 (70%): Evaluate like a professional** | Prove what the model does, with statistics | Evaluates on fresh seeds (paired comparison with `evaluate.py`), reviews play in `watch.py`, finds failure cases, writes an evaluation report | Explains held-out evaluation, paired comparisons, confidence intervals, and qualitative review | A numbers-backed statement: "model X beats the baseline by Y ± Z" (or honestly doesn't) |
| **8 (80%): Iterate with experiments** | The scientific loop: hypothesis → one change → run → compare | Plans and runs 3–5 experiments: reward shaping, curriculum, a small hyperparameter sweep, an ablation. Logs each in the journal | Explains experimental design: one variable at a time, seeds, significance, and avoiding overfitting to the evaluation | A documented improvement, or a documented negative result |
| **9 (90%): Scale up** | Generalize and go multi-agent | Trains across robot tiers (the observation includes robot capabilities). Tries 3v3 self-play (opponent pools, non-stationarity). Optionally continuous control, on the high-fidelity envs from step 3 (driving, and aiming with `continuous_aim`) | Explains self-play pitfalls, league training, and why continuous control needs far more steps; explains the sim-to-real reasons behind latency, sensor noise and traction | A policy that handles several robot types or beats the bots in 3v3, with evidence |
| **10 (100%): Ship it** | Hand it over the way a team would | Versions the model (git tag plus a models folder), writes a **model card** (intended use, training setup, metrics, limitations, failure modes), exports it (TorchScript/ONNX), measures inference latency, writes a retrospective | Explains model cards, versioning, export and deployment, and how this could help an FRC team (strategy advisor, scouting simulations, auto choices) | Someone else could reproduce and use the model from the repo alone |

**Facts the curriculum will lean on:**

- **Baseline to beat:** the scripted "strong" driver in the Day 1 robot mix wins **77% ± 4%**, averages **+37** margin, and scores **59 FUEL** per match (100 matches, `evaluate.py --baseline-only --tier strong`).
- **Episode length:** one full match = 166 s of game time.

| Mode | Decision interval | Decisions per match |
|---|---|---|
| Macro | 0.25 s | 664 |
| Continuous | 0.1 s | 1660 |

- **Speed:** a match simulates in about 1–2 s on one CPU core, so parallel environments scale with cores.
- **The reward** is configurable in `RewardConfig` (`src/rebuilt_sim/env.py`). The defaults are the team score margin ×0.1, win/loss ±1, bonus RPs ×0.5, and small shaping terms. Reward design is a user decision in Phases 5–8.
- **Game dynamics a good agent should discover:**
  - hoard FUEL while its HUB is off, and dump it when the HUB turns on
  - the 3 s grace window after each shift change
  - climbing: nobody climbed high in the real Day 1 data, so the TRAVERSAL ranking point was never earned there

### 4b. A vision curriculum (outline; same rules: the user decides and does, Claude mentors)

Step 4 generates the data (`docs/04-vision.md`). Training a detector follows the same professional arc as §4, with different content. Adapt it to the user.

| Phase | The user does | Claude does |
|---|---|---|
| Brief | Picks the task (FUEL detector? robots by alliance? tag corners?), the target hardware (Limelight/Hailo, Orange Pi + PhotonVision int8 YOLO, a laptop GPU), the metric (mAP50, mAP50-95, recall on small FUEL, frames per second) and a baseline to beat (e.g. a classic yellow-color-blob pipeline) | Explains detection metrics (IoU, precision/recall, mAP), and the hardware limits of each target |
| Data | Generates a dataset, looks at contact sheets, reads the dataset card, makes the export decisions (`docs/04-vision.md` §9), and decides whether to hand-label a small **real** test set | Explains leakage, splits by match, class balance, amodal vs visible boxes, why flips break AprilTags, and the sim-to-real gap |
| GPU | Frees disk and switches to CUDA PyTorch (§3) | Explains what CUDA is and how to verify it |
| Model and training | Chooses a detector family and library and writes or configures the training | Explains the options (one-stage vs two-stage, anchors, input size, augmentation), reviews code and configs, reads curves with them |
| Evaluation | Evaluates on held-out synthetic data and, if built, the real test set; studies failures by distance, visibility and camera effect using the labels' metadata | Explains error analysis and why synthetic scores overestimate real ones |
| Iterate and ship | Changes one thing at a time (data size, randomization ranges, model), keeps the journal, exports the model for its target, writes a model card | Explains quantization (int8), export formats, and deployment on PhotonVision/Limelight |

---

## 5. Remaining simulator engineering (Claude may build it, one step at a time, with the user's go-ahead)

**Step 3: driving and aiming fidelity** is ✅ done (chat 2). See `docs/03-driving-and-aiming.md`. Follow-ups it left, if wanted:

- Tank-drive robots (the 2026 KitBot): every robot is swerve now.
- Magnus lift from backspin in the FUEL flight model.
- Shots blocked by robots; vision latency; robots hiding AprilTags from each other.
- `ChassisSpeeds.discretize`-style skew correction for the bots' driving.

**Step 4: vision** is ✅ done (chat 3). See `docs/04-vision.md`. Follow-ups it left, if wanted (§13 there):

- **Align the simulator with the official drawings** (question 6 in §2): TOWER, OUTPOST and DEPOT positions, climb positions, the HUB hexagon's rotation, the TOWER as an open frame.
- Tinted, reflective polycarbonate ("glass"); more shadows; FUEL logos; people and carts outside the field; the AndyMark tag layout.
- A 2-3x faster renderer by batching small shapes, if generation time ever matters.

**Smaller improvements, if wanted:**

- Human players throwing FUEL over the wall, and the CORRAL return path.
- Rules not yet modeled: G405 (FUEL out of the field), G408 (catching HUB output), G415–G417 (damage/tipping), G419 (collusion).
- Later-season robot mixes (DCMP/CMP events), via `bot_tier_weights` and `event_level`.

---

## 6. Assumptions to verify (from the official 2026 Field Dimension Drawings)

| Assumption | Where it lives | Why it matters |
|---|---|---|
| **Resolved by the drawings (chat 3), not yet applied to the simulator** (question 6 in §2): the DEPOT is at blue y = 5.965 m, not 7.03; the TOWER and OUTPOST centers are 0.22 m lower in y; the rungs span only ±0.60 m, so the climb offsets of ±0.95 m are off the rungs; the HUB opening is 41.7 in across the flats *inside*, but its corners (not flats) face the alliance walls | `constants.py`, `field.py`, `ballistics.py` | FUEL routes, climbing, high-fidelity scoring |
| OUTPOST feed-zone size and human-player feed rate (4 FUEL/s) | `constants.py`, `sim.py` | OUTPOST strategy |
| HUB processing time (0.4–0.9 s) and exit speed (1.2–3.2 m/s) | `constants.py` | Grace-window scoring, FUEL recirculation |
| Official manual version used: TU22 (final) | `docs/01-video-analysis.md` | All point values and timings |
| FUEL mass 0.227 kg (the top of the official 0.203–0.227 kg range), drag coefficient 0.5, no backspin lift | `constants.py` | High-fidelity shot tables and ranges |
| Vision: colors, the TRENCH arm's section, the HUB cap and funnel bottom, the net's shape (all APPROX in `vision/field_model.py`) | `docs/04-vision.md` §3 | How realistic the synthetic images look |

---

## 7. Map of the repo

| Path | What it holds |
|---|---|
| `CLAUDE.md` | Short rules for Claude Code (auto-loaded) |
| `HANDOFF.md` | This file |
| `README.md` | Setup and every command |
| `docs/01-video-analysis.md` | Step 1: how the game works and what the real matches showed |
| `docs/02-simulator.md` | Step 2: simulator design, what is modeled, calibration results, observation/action/reward spec, training recipes |
| `data/` | Match results and score timelines extracted from the broadcast |
| `src/rebuilt_sim/constants.py`, `rules.py`, `field.py` | Game rules and field geometry (official manual + AprilTag layout) |
| `src/rebuilt_sim/robot.py`, `sim.py` | Robot capabilities, skill tiers, and the physics/referee |
| `src/rebuilt_sim/controller.py`, `bots.py` | Macro actions → driving (navigation), and the scripted drivers |
| `src/rebuilt_sim/obs.py`, `env.py`, `pz_env.py` | What the agent observes; the Gymnasium and PettingZoo environments |
| `src/rebuilt_sim/viewer.py`, `runner.py`, `policies.py` | Rendering, mixed model/bot matches, model loading |
| `src/rebuilt_sim/drivetrain.py`, `collision.py`, `sensors.py`, `apriltags.py`, `ballistics.py` | Step 3's high-fidelity physics: swerve modules, rectangular bumpers, the pose estimator and AprilTag layout, 3D FUEL flight and the aim software |
| `docs/03-driving-and-aiming.md` | Step 3: the high-fidelity physics, its calibration and its RL interface |
| `src/rebuilt_sim/vision/` | Step 4: `render.py` (the ray-casting renderer), `camera.py` (lenses, mounts, WPILib/OpenCV poses), `field_model.py` (the field from the drawings), `robot_model.py`, `tags.py` (36h11 patterns), `scene.py` (a match in 3D), `labels.py`, `effects.py` and `randomize.py` (domain randomization), `dataset.py`, `export.py`, `check.py`, `card.py` |
| `docs/04-vision.md` | Step 4: how the rendering and labels work, the camera presets, the dataset format and conventions, and the user's export decisions |
| `scripts/play.py`, `watch.py`, `calibrate.py`, `evaluate.py` | Tools |
| `scripts/render_view.py`, `make_dataset.py`, `check_dataset.py`, `export_dataset.py` | Vision tools |
| `scripts/train_ppo.py`, `train_selfplay.py` | **Reference only.** See the rule at the top |
| `tests/` | 85 tests: rules, field, physics, navigation, environment APIs, step 3's drivetrain, ballistics and sensors, and step 4's rendering, labels and datasets |

---

## 8. What happened in chat 1 (context)

- **Step 1: video analysis.**
  - Sources:
    - the full 9.5-hour livestream: transcript, 3,404 storyboard thumbnails, 171 full-resolution frames, every results screen
    - the official game manual (TU22)
  - Main findings:
    - elite alliances hoard FUEL while their HUB is off, then dump 100–165 FUEL per 25 s active shift
    - one team (9483) was in every 250+ FUEL alliance
    - climbing was almost unused
    - scores ranged 0–409
- **Step 2: the simulator.**
  - The field was built from the official 2026 AprilTag layout.
  - Bots were calibrated to the real matches:

    | Measure | Simulated | Real |
    |---|---|---|
    | Elite + strong + strong alliance FUEL | 398 | 393–409 |
    | AUTO share of FUEL | 20.8% | 19.8% |
    | TOWER points per alliance | 0.46 | 0.48 |

  - An independent code review found 11 bugs, including robots stalling at crossings and only one robot able to climb. All are fixed, each with a regression test.
  - Smoke-test training runs were made only to check that the scripts work, and were deleted. **No trained models exist.**

## 9. What happened in chat 2 (step 3)

- The user said "continue building the game", so chat 2 built **step 3**. That is simulator engineering, which Claude may do. No training was run.
- **A high-fidelity physics mode** (`MatchConfig(hifi=HiFiConfig())`, `EnvConfig(hifi=...)`, three new Gymnasium ids). The standard strategy physics is untouched and still the default. It was checked bit-for-bit against recorded matches, observations and env rewards from before the change.
  - **Swerve:** four modules per robot with WPILib motor constants, the velocity loop, current limits, the friction circle and wheel slip. Per-tier hardware reproduces each tier's calibrated speed and acceleration.
  - **Latency:** 40 ms. **Physics step:** 0.02 s.
  - **Bumpers:** rectangles, with spin from off-center hits.
  - **Pose estimate:** odometry, gyro drift, and vision from the official AprilTag layout (now in `data/`).
  - **3D FUEL ballistics:** into the hexagonal HUB opening. The shooter mechanism lags, and the aim software uses shot tables and shoot-on-the-move lead.
  - **Manual aiming:** the `continuous_aim` action mode.
- **Calibration** (`docs/03` §6 and §8):
  - Shooter spreads were tuned so each tier's stationary hit rate at its sweet range equals its step-2 accuracy. For elite, strong and mid, the drop-off with distance then came close to step 2's hand-tuned falloff without further tuning.
  - In full bot matches accuracy carries over, but robots score about 20% less FUEL, and full elite alliances reach about 316 instead of 393–409. Per robot, every tier is still inside its real Day 1 range.
  - Diagnosis: a lone robot collects almost as fast, and FUEL is spread the same way. The loss is slower unloading and cycling under realistic physics:
    - lining up within a few degrees;
    - flywheel spin-up;
    - about 1 s high-arc flights;
    - 40 ms latency;
    - an aim interlock that at first waited for full flywheel recovery before every ball. It was changed to start-a-volley / keep-firing hysteresis, as real shooter code does.
  - Left as documented; tuning the bots' high-fidelity driving to hit the alliance totals is a possible follow-up if the user wants it.
- **Bugs found and fixed while building, all in the new code:**
  - the latency queue grew while robots were disabled, so after AUTO every robot acted on 3-second-old commands;
  - encoder readings for slipping wheels were too extreme;
  - bots wedged against TRENCH columns and teammates, fixed by steering along what they touch;
  - bots stuck at the DEPOT, fixed by driving the intake up to the wall.
- **Official sources:** the WPILib AprilTag layout could be downloaded. The game manual site was blocked by the network policy; the HUB opening (41.7 in hexagon) and FUEL weight (about 0.5 lb) came from search results. Both are marked APPROX.

## 10. What happened in chat 3 (step 4)

- The user said "Build step 4". That is simulator engineering (rendering and vision data), which Claude may do. No models were trained and no training code was written.
- **Research first**, from official sources: the 2026 Field Dimension Drawings and Field Manual (downloadable with curl), the AprilTag 36h11 code table (checked against the official tag images), and what cameras and mounts FRC teams actually use.
- **What was built** (`docs/04-vision.md`):
  - a NumPy ray-casting renderer (exact per-pixel geometry, lens distortion, supersampling, deferred lighting);
  - the field from the drawings, with all 32 tags;
  - randomized robots with bumper numbers, which tilt on BUMPs;
  - FUEL on the ground, in flight and in the chutes; HUB lights that follow the shift schedule;
  - camera presets and mounts from real teams; domain randomization;
  - exact labels; a dataset generator (parallel, resumable, deterministic, split by match); COCO/YOLO exporters where the user makes the training choices; a checker with contact sheets; a dataset card and manifest.
- **Checked against independent tools:** the AprilTag C library (the one WPILib and PhotonVision use) reads the rendered tags with the labeled IDs, and its corners agree with the labels to a fraction of a pixel. solvePnP on the labeled corners reproduces the labeled tag poses exactly.
- **Found on the way:**
  - the simulator disagrees with the official drawings in a few places (question 6 in §2); left for the user to decide;
  - random lens distortion can describe impossible lenses, which are now rejected;
  - the AprilTag library reports corners +0.5 px from OpenCV's convention;
  - OpenCV's ArUco returns AprilTag corners rotated 180°.

  All of these are documented.
- **The simulator code was not touched.** The strategy physics was checked bit-for-bit against recordings from before the change.
- A 3-agent design panel plus a judge reviewed the draft; its correctness and dataset-workflow fixes were applied. The batched-renderer speed-up and the glass effect were left as follow-ups.
