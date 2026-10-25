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
RUN_NAME = "strategy-v2-continued"
TOTAL_STEPS = 3_000_000  # steps in this run (a short test run is 100_000)
N_ENVS = 16  # matches played at the same time, one per CPU core
DECISION_DT = 0.25  # seconds of game time between the agent's decisions
SEED = 1  # strategy-v2 used 0: a new seed so the continuation plays new matches
GAMMA = 0.995  # how far ahead the agent looks: about 1 / (1 - GAMMA) decisions (strategy-v1 used 0.99)
# Keep training a saved model (its own settings, gamma included, come with it), or None to start from scratch
CONTINUE_FROM = "models/strategy-v2-gamma0995/final_model.zip"


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
    if CONTINUE_FROM:
        model = PPO.load(CONTINUE_FROM, env=env, verbose=1, tensorboard_log="runs/tensorboard")
        model.set_random_seed(SEED)
    else:
        model = PPO("MlpPolicy", env, gamma=GAMMA, seed=SEED, verbose=1, tensorboard_log="runs/tensorboard")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    config = {
        "env": {"id": "Rebuilt-Strategy-v0", "decision_dt": DECISION_DT, "learner_tiers": ["strong"]},
        "ppo": {k: getattr(model, k) for k in ("learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef")},
        "total_steps": TOTAL_STEPS, "n_envs": N_ENVS, "seed": SEED, "git_commit": commit,
        "continued_from": CONTINUE_FROM, "steps_before": model.num_timesteps,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # a continued run keeps counting its steps from where the saved model stopped
    model.learn(total_timesteps=TOTAL_STEPS, tb_log_name=RUN_NAME, reset_num_timesteps=not CONTINUE_FROM)
    model.save(run_dir / "final_model")
    env.close()


if __name__ == "__main__":
    main()
