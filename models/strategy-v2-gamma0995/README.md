# strategy-v2-gamma0995

The second experiment: strategy-v1 with **one change, `gamma` 0.99 → 0.995**. The agent values rewards
about 200 decisions (50 s) ahead instead of about 100 (25 s), long enough to plan across a whole 25 s
HUB shift. Everything else is identical: PPO (Stable-Baselines3) driving the **strong** robot with
**macro actions** (a decision every 0.25 s) on the strategy physics against the Day 1 bot mix, 3M steps,
16 parallel matches, seed 0.

- **Trained:** 2026-10-18 with `scripts/my_train.py` at commit `950c39d` (which sets `GAMMA = 0.995`),
  2 h 5 min on a 4-core cloud machine. `config.json`, `train.log` and `tensorboard/` as for strategy-v1.
- **Training curve:** ahead of strategy-v1 at every one of the nine checkpoints (rounds 10-90), and
  settling on its choices sooner (entropy -0.93 at the end vs -0.99).

## Evaluation (`eval.txt`, `compare_v1.txt`)

All on the same 200 held-out matches (seeds 10000-10199) as the baseline and strategy-v1.

| | Scripted driver | strategy-v1 (gamma 0.99) | strategy-v2 (gamma 0.995) |
|---|---|---|---|
| Average margin | +32.2 | +31.5 | **+38.8** |
| Win rate | 78% | 77% | 80% |
| Ranking points per match | 2.75 | 2.73 | **2.96** |
| The robot's own FUEL | 59.2 | 56.7 | **63.9** |
| FUEL wasted into an inactive HUB | 2.7 | 1.2 | 1.5 |
| Fouls | 0.04 | 0.00 | 0.00 |

- **strategy-v2 vs the scripted driver:** paired margin change **+6.6 ± 1.3** (standard error; 95% interval
  about +4.0 to +9.2). The first model to beat the scripted driver; the brief's +20 target is not met yet.
- **strategy-v2 vs strategy-v1:** paired margin change **+7.3 ± 1.5** (95% interval about +4.4 to +10.2).
  Looking further ahead helped: v2 scores about 7 more FUEL itself and earns 0.2 more ranking points per match.

**Caveat:** each setting was trained once (seed 0). The ± values cover match-to-match luck, not
run-to-run luck: a second training seed for each setting would show how much of the +7.3 is `gamma`
and how much is this particular run.
