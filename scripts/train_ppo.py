"""Train a PPO agent (Stable-Baselines3) to drive one robot; the other five are scripted bots.

REFERENCE IMPLEMENTATION: the project's owner is building their own training pipeline as a
hands-on learning exercise (see HANDOFF.md). Study or compare against this file; it is not the plan.

    python scripts/train_ppo.py --mode macro --timesteps 3000000 --name strategy
    python scripts/train_ppo.py --mode continuous --timesteps 20000000 --name control
    tensorboard --logdir runs                     # watch the curves while it trains

Outputs go to runs/<name>/: checkpoints, final_model.zip, config.json and TensorBoard logs.
Resume with --resume runs/<name>/final_model.zip (or a checkpoint).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # before numpy loads: one BLAS thread per env process

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor  # noqa: E402

from rebuilt_sim.env import EnvConfig, RebuiltEnv, RewardConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    # macro actions change slowly, so decide every 0.25 s (664 decisions per match)
    "macro": dict(decision_dt=0.25, gamma=0.995, ent_coef=0.01, n_steps=512, batch_size=2048, n_epochs=6),
    # low-level control needs faster decisions (1660 per match)
    "continuous": dict(decision_dt=0.1, gamma=0.997, ent_coef=0.0, n_steps=1024, batch_size=4096, n_epochs=8),
}


class MatchStatsCallback(BaseCallback):
    """Logs match results (score, win, ranking points) to TensorBoard at the end of each match."""

    def __init__(self) -> None:
        super().__init__()
        self.buf: dict[str, list[float]] = {}

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "own_score" in info:
                for k in ("own_score", "opp_score", "own_fuel", "win", "tie", "ranking_points",
                          "robot_fuel_scored", "robot_fouls", "tower_points"):
                    self.buf.setdefault(k, []).append(float(info[k]))
                self.buf.setdefault("margin", []).append(float(info["own_score"] - info["opp_score"]))
        return True

    def _on_rollout_end(self) -> None:
        for k, v in self.buf.items():
            if v:
                self.logger.record(f"match/{k}", float(np.mean(v)))
        self.buf = {}


def make_env(cfg: EnvConfig, seed: int):
    def _init():
        env = RebuiltEnv(cfg)
        env.reset(seed=seed)
        return env
    return _init


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["macro", "continuous"], default="macro")
    ap.add_argument("--name", default=None, help="run name (default: <mode>-<timestamp>)")
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    ap.add_argument("--envs", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 4)))
    ap.add_argument("--tiers", nargs="+", default=["strong"],
                    help="robot types the agent drives, sampled per match (elite strong mid low climber ...)")
    ap.add_argument("--event-level", default="regional", choices=["regional", "district", "dcmp", "cmp"])
    ap.add_argument("--decision-dt", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=None)
    ap.add_argument("--pickup-reward", type=float, default=RewardConfig.pickup)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="path to a .zip model to keep training")
    ap.add_argument("--checkpoint-every", type=int, default=500_000)
    args = ap.parse_args()

    d = DEFAULTS[args.mode]
    name = args.name or f"{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = ROOT / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig(
        action_mode=args.mode,
        decision_dt=args.decision_dt or d["decision_dt"],
        learner_tiers=tuple(args.tiers),
        event_level=args.event_level,
        reward=replace(RewardConfig(), pickup=args.pickup_reward),
    )
    gamma = args.gamma or d["gamma"]
    ent = d["ent_coef"] if args.ent_coef is None else args.ent_coef
    (out / "config.json").write_text(json.dumps(
        {"env": asdict(cfg), "gamma": gamma, "lr": args.lr, "ent_coef": ent, "timesteps": args.timesteps,
         "envs": args.envs, "mode": args.mode}, indent=2, default=str))

    fns = [make_env(cfg, args.seed * 1000 + i) for i in range(args.envs)]
    venv = VecMonitor(SubprocVecEnv(fns) if args.envs > 1 else DummyVecEnv(fns))
    if args.resume:
        model = PPO.load(args.resume, env=venv, device="cpu", tensorboard_log=str(out))
    else:
        model = PPO(
            "MlpPolicy", venv, learning_rate=args.lr, n_steps=d["n_steps"], batch_size=d["batch_size"],
            n_epochs=d["n_epochs"], gamma=gamma, gae_lambda=0.95, clip_range=0.2, ent_coef=ent,
            vf_coef=0.5, max_grad_norm=0.5, policy_kwargs={"net_arch": [256, 256]},
            tensorboard_log=str(out), seed=args.seed, device="cpu", verbose=1,
        )
    callbacks = [
        MatchStatsCallback(),
        CheckpointCallback(save_freq=max(1, args.checkpoint_every // args.envs), save_path=str(out / "checkpoints"),
                           name_prefix="ppo"),
    ]
    print(f"training {args.mode} agent for {args.timesteps:,} steps on {args.envs} envs -> {out}")
    model.learn(total_timesteps=args.timesteps, callback=callbacks, tb_log_name="ppo",
                reset_num_timesteps=not args.resume)
    model.save(out / "final_model")
    venv.close()
    print(f"saved {out / 'final_model.zip'}")


if __name__ == "__main__":
    main()
