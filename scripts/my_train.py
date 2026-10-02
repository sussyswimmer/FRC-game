"""My first training run: PPO from Stable-Baselines3 on the macro-action strategy game.

    .venv\Scripts\python.exe scripts\my_train.py
"""

import json
import subprocess
from pathlib import Path

import rebuilt_sim.env  # noqa: F401  registers the "Rebuilt-..." environments with Gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# --- settings: every choice for this run, in one place ------------------------------
RUN_NAME = "strategy-v1-test"
TOTAL_STEPS = 100_000  # a short test run; a real run is about 3_000_000
N_ENVS = 16  # matches played at the same time, one per CPU core
DECISION_DT = 0.25  # seconds of game time between the agent's decisions
SEED = 0


def make_envs():
    return make_vec_env(
        "Rebuilt-Strategy-v0",
        n_envs=N_ENVS,
        seed=SEED,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"decision_dt": DECISION_DT, "learner_tiers": ("strong",)},
    )


def main():
    run_dir = Path("runs") / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    env = make_envs()
    model = PPO("MlpPolicy", env, seed=SEED, verbose=1, tensorboard_log="runs/tensorboard")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    config = {
        "env": {"id": "Rebuilt-Strategy-v0", "decision_dt": DECISION_DT, "learner_tiers": ["strong"]},
        "ppo": {k: getattr(model, k) for k in ("learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef")},
        "total_steps": TOTAL_STEPS, "n_envs": N_ENVS, "seed": SEED, "git_commit": commit,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    model.learn(total_timesteps=TOTAL_STEPS, tb_log_name=RUN_NAME)
    model.save(run_dir / "final_model")
    env.close()


if __name__ == "__main__":
    main()
