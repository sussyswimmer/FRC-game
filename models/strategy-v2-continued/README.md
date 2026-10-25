# strategy-v2-continued

strategy-v2-gamma0995 **trained for 3M more steps** (about 6M in total). Nothing else changed: the
continuation loads the saved model with all its settings (`gamma` 0.995, the network and the optimizer
state) and keeps training it, with seed 1 so it plays new matches instead of replaying the first 3M
steps' matches.

- **Trained:** 2026-10-25 with `scripts/my_train.py` at commit `07df6af` (`CONTINUE_FROM =
  "models/strategy-v2-gamma0995/final_model.zip"`), 2 h 9 min on a 4-core cloud machine. `config.json`
  records the starting model and step count; `train.log` and `tensorboard/` hold the curves (steps
  3.01M to 6.03M).
- **Training curve:** reward per match between about +4.7 and +7.8 (v2 ended near +4.5); entropy from
  -0.84 to -0.77 (v2 ended at -0.93).

## Evaluation (`eval.txt`, `compare_v2.txt`)

All on the same 200 held-out matches (seeds 10000-10199).

| | Scripted driver | strategy-v1 | strategy-v2 | strategy-v2-continued |
|---|---|---|---|---|
| Training steps | - | 3M | 3M | 6M |
| Average margin | +32.2 | +31.5 | +38.8 | **+44.0** |
| Win rate | 78% | 77% | 80% | 80% |
| Ranking points per match | 2.75 | 2.73 | 2.96 | **3.01** |
| The robot's own FUEL | 59.2 | 56.7 | 63.9 | **68.8** |
| FUEL wasted into an inactive HUB | 2.7 | 1.2 | 1.5 | 1.2 |
| Fouls | 0.04 | 0.00 | 0.00 | 0.00 |

- **vs the scripted driver:** paired margin change **+11.8 ± 1.3** (standard error; 95% interval about
  +9.3 to +14.3). The brief's +20 target is not met yet.
- **vs strategy-v2 (the same model before the extra 3M steps):** **+5.2 ± 1.4** (95% interval about
  +2.5 to +7.9). Training longer still helps; most of the gain is FUEL the robot scores itself (+4.9 per match).
