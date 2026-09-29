# FRC 2026 REBUILT simulator

A match simulator and reinforcement-learning environments for the 2026 FRC game *REBUILT presented by Haas*.

- **Field:** built from the official AprilTag layout.
- **Rules:** follow the game manual (TU22).
- **Calibration:** robot skill levels are tuned against the real 2026 İstanbul Regional Day 1 matches.
- **Training:** you train the models yourself. This repo gives you the environments, the training scripts and the tools to watch and measure the results.

- **Start here if you're picking the project back up: [HANDOFF.md](HANDOFF.md)** (status, next steps, and the hands-on training curriculum)
- Step 1 (video analysis): [docs/01-video-analysis.md](docs/01-video-analysis.md)
- Simulator design, rules coverage and calibration: [docs/02-simulator.md](docs/02-simulator.md)

## Setup

Uses [uv](https://docs.astral.sh/uv/) and Python 3.11:

```bash
uv sync --all-extras
```

That creates `.venv/` with NumPy, Gymnasium, PettingZoo, pygame-ce, Stable-Baselines3, TensorBoard and CPU PyTorch. The CPU build is intentional: the policies are small MLPs, which train as fast on the CPU, and the CUDA build needs about 5 GB of disk. For GPU training later, change the `pytorch-cpu` index URL in `pyproject.toml` to `https://download.pytorch.org/whl/cu126` and run `uv sync --all-extras` again.

Run the commands below from this folder with `.venv\Scripts\python.exe`, or activate the venv first with `.venv\Scripts\activate`.

## Try it

```bash
python -m pytest -q
```

34 tests: the rules, the field geometry, the physics invariants, navigation, and the Gymnasium/PettingZoo API checkers.

```bash
python scripts/play.py
```

Drive a robot yourself against the bots. Keys: WASD move, Q/E rotate, SPACE intake, F shoot, C climb.

```bash
python scripts/watch.py --blue elite strong mid --red strong low low
```

Watch scripted bots play.

```bash
python scripts/calibrate.py --matches 96
```

Compare bot matches with the real Day 1 results.

## Train

> The scripts below are **reference implementations**. The plan is for you to build and run your own training pipeline step by step, from project brief to shipped model, the way ML engineers do at work. See the curriculum in [HANDOFF.md](HANDOFF.md). Use these to study or to compare against.

**1. Match strategy.** The agent picks one of 8 macro actions for one robot, and bots drive the rest. It takes roughly half an hour for 3M steps on this PC.

```bash
python scripts/train_ppo.py --mode macro --tiers strong --timesteps 3000000 --name strategy
```

**2. Low-level driving:** continuous drive, intake, shoot and climb control. This is a much harder problem.

```bash
python scripts/train_ppo.py --mode continuous --tiers strong --timesteps 20000000 --name control
```

**3. 3v3 self-play:** one shared policy drives all six robots. A quarter of the matches are played against the bots so you can track progress.

```bash
python scripts/train_selfplay.py --mode macro --timesteps 20000000 --name selfplay
```

Watch the curves while any of these run:

```bash
tensorboard --logdir runs
```

The most useful curves:

- `match/win`, `match/margin`: the win rate and score margin.
- `vs_bots/win`: the self-play agent's win rate against the scripted bots.
- `match/robot_fuel_scored`: FUEL your robot scored.
- `match/robot_fouls`: fouls your robot committed.

## Check what you trained

Did the training beat a typical team's driving, on identical matches?

```bash
python scripts/evaluate.py runs/strategy/final_model.zip --tier strong --matches 100
```

Watch your model drive:

```bash
python scripts/watch.py --model runs/strategy/final_model.zip --seat blue_1 --tier strong
```

Watch a self-play model drive all six robots:

```bash
python scripts/watch.py --model runs/selfplay/latest.pt --all
```

## Using the environments in your own code

```python
import gymnasium as gym
import rebuilt_sim.env  # registers the ids

env = gym.make("Rebuilt-Strategy-v0")  # Discrete(8) macro actions, 162-float observation
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

from rebuilt_sim.env import EnvConfig, RebuiltEnv, RewardConfig
env = RebuiltEnv(EnvConfig(action_mode="continuous", learner_tiers=("elite",),
                           reward=RewardConfig(pickup=0.0, self_fuel=0.05)))

from rebuilt_sim.pz_env import RebuiltParallelEnv  # PettingZoo, several learning robots
```

## Layout

| Path | What it holds |
|---|---|
| `src/rebuilt_sim/constants.py` | Field geometry (from the AprilTag layout), match timing, point values, RP thresholds |
| `src/rebuilt_sim/rules.py` | Clock, HUB shift schedule, 3 s grace windows, scoring, ranking points, fouls |
| `src/rebuilt_sim/field.py` | Obstacles, zones, BUMPs, TRENCHes, TOWERs, DEPOTs, OUTPOSTs |
| `src/rebuilt_sim/robot.py` | Robot capabilities and the calibrated skill tiers (elite, strong, mid, low, climber, broken) |
| `src/rebuilt_sim/sim.py` | The physics and referee: 504 FUEL, shooting, HUB counting, climbing, fouls |
| `src/rebuilt_sim/controller.py` | Macro actions to drive commands (navigation, targeting, shooting spots, defense) |
| `src/rebuilt_sim/bots.py` | Scripted drivers that play like the Day 1 teams |
| `src/rebuilt_sim/obs.py`, `env.py`, `pz_env.py` | Observations and the Gymnasium/PettingZoo environments |
| `src/rebuilt_sim/viewer.py`, `runner.py`, `policies.py` | Rendering, mixed policy/bot matches, loading trained models |
| `scripts/` | play, watch, calibrate, train_ppo, train_selfplay, evaluate |
| `data/` | Match results and score timelines extracted from the Day 1 broadcast |
| `runs/` | Your training runs (created when you train) |
