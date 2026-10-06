# strategy-v1

The first trained model: PPO (Stable-Baselines3) driving the **strong** robot with **macro actions**
(a decision every 0.25 s) on the strategy physics, against the Day 1 mix of scripted bots.

- **Trained:** 2026-10-03 with `scripts/my_train.py` (`RUN_NAME = "strategy-v1"`, `TOTAL_STEPS = 3_000_000`,
  everything else as committed): 16 parallel matches, SB3's default PPO settings. 2 h 29 min on a
  4-core cloud machine. `config.json` has the exact settings and code version; `train.log` the printed
  tables; `tensorboard/` the curves (`tensorboard --logdir models/strategy-v1/tensorboard`).
- **Training curve:** reward per match rose from random-play level (about -1) to about +3 to +4.5;
  entropy fell from -2.08 (random) to -0.99.

## Evaluation (`eval.txt`)

`scripts/evaluate.py models/strategy-v1/final_model.zip --tier strong --matches 200`: the same 200
held-out matches (seeds 10000-10199) played once by the model and once by the scripted strong driver.

| | Scripted driver | strategy-v1 |
|---|---|---|
| Average margin | +32.2 ± 8.0 | +31.5 ± 7.9 |
| Win rate | 78% | 77% |
| Ranking points per match | 2.75 | 2.73 |
| The robot's own FUEL | 59.2 | 56.7 |
| FUEL wasted into an inactive HUB | 2.7 | 1.2 |
| Fouls | 0.04 | 0.00 |

**Paired margin change: -0.7 ± 1.4** (standard error; 95% interval about -3.5 to +2.1).
The model plays as well as the scripted driver, not better: the project brief's target of +20 is not met.

Running the evaluation on Linux needs `OMP_NUM_THREADS=1` (otherwise PyTorch threads in the worker
processes compete and it stalls). Windows and macOS start worker processes differently and should be
less affected; if it is slow there too, set the same variable.
