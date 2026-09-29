# HANDOFF: FRC 2026 REBUILT simulator and ML training project

*Written 2026-09-29 at the end of the first Claude Code chat. For the next Claude Code chat, and for the user.*

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
| **3. Driving/aiming fidelity** (simulator engineering) | ⏳ Not started | See §5 |
| **4. Vision track** (simulator engineering plus user-led training) | ⏳ Not started | See §5 |

**Scope the user chose after step 1:**

- Covered: **match strategy**, **driving/aiming control**, and **vision**.
- Not chosen: "autonomous routine optimization".

**Order is the user's call.** The strategy environment is ready now, so the training curriculum could start immediately. The alternative is to build step 3 or 4 first.

---

## 2. Open questions: ask the user these at the start of the next chat

1. **What next?** Start the training curriculum (§4) on the existing strategy environment, or build step 3 (driving/aiming) or step 4 (vision) first?
2. **Background:** how comfortable are they with Python, NumPy, PyTorch, general ML, and reinforcement learning? Adapt depth and pace to the answer.
3. **Learning path for the training loop:**
   - **(A)** Write PPO from scratch, single-file CleanRL style. This is the most learning; they can then compare against Stable-Baselines3.
   - **(B)** Start with Stable-Baselines3 and study how it works inside.
   - **(C)** Dissect the existing reference scripts.
4. **Experiment tracking:** TensorBoard (already installed, local), or Weights & Biases (the industry-standard web dashboard; needs a free account)?
5. **Field-drawing check:** can they get the official 2026 Field Dimension Drawings (§6)? Two simulator assumptions should be checked against them.

---

## 3. Getting set up (new chat or new machine)

```powershell
git clone https://github.com/sussyswimmer/FRC-game.git
cd FRC-game
uv sync --all-extras                      # creates .venv (Python 3.11, CPU PyTorch, SB3, Gymnasium, PettingZoo, pygame-ce)
.venv\Scripts\python.exe -m pytest -q     # expect: 34 passed
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
- The **vision track will need the GPU.** First free about 8 GB. Then change the `pytorch-cpu` index URL in `pyproject.toml` to `https://download.pytorch.org/whl/cu126` and run `uv sync --all-extras` again.

**Useful commands:**

| Command | What it does |
|---|---|
| `scripts\watch.py` | Watch bots, or a trained model, play |
| `scripts\calibrate.py --matches 96` | Compare bot matches with the real Day 1 results |
| `scripts\evaluate.py MODEL --tier strong --matches 100` | Compare a model with the scripted driver in the same seat on the same matches |
| `tensorboard --logdir runs` | Training curves |

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
| **4 (40%): Design the model** | Build the policy/value network | Writes the PyTorch module: 162 inputs, hidden layers, output head (categorical for the 8 macro actions, Gaussian for continuous). Tests shapes. Counts parameters | Explains the options: layer sizes, activations, shared vs separate actor/critic, initialization. Reviews | A forward pass works on real observations, and the user can justify each design choice |
| **5 (50%): Build the training pipeline** | A working PPO loop the user understands end to end | Writes, piece by piece: vectorized envs, rollout buffer, GAE, PPO update, logging, checkpoints, config file, seeding. Unit-tests GAE with hand-computed numbers | Guides one component at a time and reviews each piece. Explains SB3's equivalents if they chose path B | A ~50k-step smoke run works end to end: it logs to TensorBoard, and a checkpoint saves and reloads |
| **6 (60%): First real run and debugging** | Learn to read curves and debug RL | Runs the first real training (e.g. 1–3M steps, macro actions, "strong" robot). Watches reward, episode length, entropy, approx_kl, clip fraction, explained variance, value loss, and the match stats | Explains what healthy and unhealthy curves look like, and the RL debugging checklist: reward scale, advantage normalization, learning rate, entropy collapse, reward hacking | The user can explain every curve in their run and what it says |
| **7 (70%): Evaluate like a professional** | Prove what the model does, with statistics | Evaluates on fresh seeds (paired comparison with `evaluate.py`), reviews play in `watch.py`, finds failure cases, writes an evaluation report | Explains held-out evaluation, paired comparisons, confidence intervals, and qualitative review | A numbers-backed statement: "model X beats the baseline by Y ± Z" (or honestly doesn't) |
| **8 (80%): Iterate with experiments** | The scientific loop: hypothesis → one change → run → compare | Plans and runs 3–5 experiments: reward shaping, curriculum, a small hyperparameter sweep, an ablation. Logs each in the journal | Explains experimental design: one variable at a time, seeds, significance, and avoiding overfitting to the evaluation | A documented improvement, or a documented negative result |
| **9 (90%): Scale up** | Generalize and go multi-agent | Trains across robot tiers (the observation includes robot capabilities). Tries 3v3 self-play (opponent pools, non-stationarity). Optionally continuous control | Explains self-play pitfalls, league training, and why continuous control needs far more steps | A policy that handles several robot types or beats the bots in 3v3, with evidence |
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

---

## 5. Remaining simulator engineering (Claude may build it, one step at a time, with the user's go-ahead)

**Step 3: driving and aiming fidelity** (for training low-level control):

- Physics step 0.02 s (WPILib's loop rate) instead of 0.05 s.
- Swerve-module dynamics: module steering and drive, motor current and torque limits, wheel slip.
- Latency and sensor noise.
- Rectangular robot footprints instead of circles.
- Real 3D FUEL ballistics into the 72 in HUB opening, with launch speed and angle as actions, instead of a hit probability.
- Keep the same rules engine and tests, and add a "high-fidelity" config flag.

**Step 4: vision** (for training detection and localization):

- Render the same match state in 3D: FUEL, robots, field elements, and the 32 AprilTags.
- Generate labeled synthetic images: bounding boxes, poses, domain randomization.
- The user trains a detector or localizer on the RTX 3070, which needs the CUDA PyTorch switch (§3).
- Use a lightweight renderer or Unity. Avoid Isaac Sim: 16 GB of RAM is too little.

**Smaller improvements, if wanted:**

- Human players throwing FUEL over the wall, and the CORRAL return path.
- Rules not yet modeled: G405 (FUEL out of the field), G408 (catching HUB output), G415–G417 (damage/tipping), G419 (collusion).
- Later-season robot mixes (DCMP/CMP events), via `bot_tier_weights` and `event_level`.

---

## 6. Assumptions to verify (from the official 2026 Field Dimension Drawings)

| Assumption | Where it lives | Why it matters |
|---|---|---|
| **3 robots can climb side by side** across the TOWER face (offsets 0, ±0.95 m). The manual only implies at least 2. | `field.py` `CLIMB_SLOT_OFFSETS` | TRAVERSAL ranking point, END GAME strategy |
| **DEPOT position** along the alliance wall (blue center y = 7.03 m, estimated from the broadcast) | `constants.py` `DEPOT_CENTER_Y_BLUE` | Early-match FUEL routes |
| OUTPOST feed-zone size and human-player feed rate (4 FUEL/s) | `constants.py`, `sim.py` | OUTPOST strategy |
| HUB processing time (0.4–0.9 s) and exit speed (1.2–3.2 m/s) | `constants.py` | Grace-window scoring, FUEL recirculation |
| Official manual version used: TU22 (final) | `docs/01-video-analysis.md` | All point values and timings |

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
| `scripts/play.py`, `watch.py`, `calibrate.py`, `evaluate.py` | Tools |
| `scripts/train_ppo.py`, `train_selfplay.py` | **Reference only.** See the rule at the top |
| `tests/` | 34 tests: rules, field, physics, navigation, environment APIs |

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
