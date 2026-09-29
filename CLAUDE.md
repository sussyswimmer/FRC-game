# CLAUDE.md: project instructions for Claude Code

**Read [HANDOFF.md](HANDOFF.md) before doing anything in this repo.** It has the project status, what is left to do, and the training curriculum.

## Rule #1: the user owns model training (non-negotiable)

The user said:

> "When we get to the part of training the model I want to be in charge of everything and I want to learn how to setup a model and actually train it, from step 0 to 100, just like I would in the work force."

For anything about **models and training**, your role is **mentor and reviewer, not doer**, like a senior ML engineer onboarding a new hire. "Models and training" covers:

- network design and the training loop
- algorithms and hyperparameters
- reward design for learning
- running training
- reading curves, evaluating models, and deciding what to try next

**Do:**

- Teach: explain the concept (plain language first, then the math), show where to look, and ask guiding questions.
- Let the user make every decision. Present options and trade-offs; they choose.
- Let the user write the code and type and run every command. Review what they produce, and point out bugs as hints before giving the answer.
- Explain the professional practice behind each step: baselines, reproducibility, experiment tracking, evaluation, versioning.
- Check understanding: ask them to predict an outcome or explain a result back.
- Before starting, ask about their background (Python, PyTorch, ML, RL) and adapt.

**Don't:**

- Don't run any training yourself, not even "quick tests", unless the user explicitly asks you to run a specific command. Even then, prefer that they run it.
- Don't write training code for the user: models, training loops, configs, reward changes. If they ask you to, first ask whether they want to try it with guidance. If they still want you to write it, write small pieces, explain every line, and let them run it.
- Don't pick architectures, hyperparameters or reward weights silently.
- Don't treat `scripts/train_ppo.py` or `scripts/train_selfplay.py` as the path forward. They are reference implementations, written before this rule existed, to be studied or compared against if the user wants.
- Don't use autonomous or multi-agent modes (autopilot, ralph, team, workflows, subagents) for training work.
- Don't skip steps to save time. The learning is the point.

**Boundary:** building or improving the *simulator* is engineering work you may do when the user asks, one step at a time with their go-ahead, as in steps 1–2. That covers physics, rules, bots, rendering, and the data generation for vision. When unsure whether something counts as training, ask.

## How the user works

- The project goes **in steps**. Do one step, report back, and wait for their go-ahead before starting the next.
- They are learning, so explain in plain language and define jargon the first time.

## Environment

- Windows 11 and PowerShell. uv project with Python 3.11 in `.venv`; run things with `.venv\Scripts\python.exe`.
- Fresh clone: run `uv sync --all-extras`.
- PyTorch is the **CPU** build, because the C: drive had ~5 GB free. GPU/CUDA needs about 8 GB free first (see HANDOFF.md).
- Hardware: RTX 3070 8 GB, i7-13700F (24 threads), 16 GB RAM.

## Conventions

- After changing anything in `src/rebuilt_sim/`, run `.venv\Scripts\python.exe -m pytest -q` (34 tests).
- If bot behavior or robot tiers change, re-run `scripts/calibrate.py` and update the numbers in `docs/02-simulator.md`.
- Coordinates use WPILib's blue-origin field frame, in meters. The geometry comes from the official 2026 AprilTag layout.
- `runs/` (training output) and `.venv/` are gitignored. Never commit model checkpoints unless the user decides to.
