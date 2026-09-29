"""Gymnasium and PettingZoo API conformance, plus a full random-policy episode."""

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import parallel_api_test

import rebuilt_sim.env  # noqa: F401  (registers the gym ids)
from rebuilt_sim.env import EnvConfig, RebuiltEnv
from rebuilt_sim.obs import OBS_SIZE
from rebuilt_sim.pz_env import RebuiltParallelEnv


def test_gym_env_passes_checker():
    for mode in ("macro", "continuous"):
        env = RebuiltEnv(action_mode=mode)
        check_env(env, skip_render_check=True)
        env.close()


def test_registered_ids():
    env = gym.make("Rebuilt-Strategy-v0")
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_SIZE,) and env.action_space.n == 8
    env.close()


def test_full_episode_macro():
    env = RebuiltEnv(EnvConfig(learner_tiers=("mid",)))
    obs, info = env.reset(seed=1)
    total, steps, done = 0.0, 0, False
    while not done:
        obs, r, done, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(obs).all() and np.abs(obs).max() <= 2.0
        total += r
        steps += 1
    assert steps == 1660
    for key in ("own_score", "opp_score", "win", "ranking_points"):
        assert key in info


def test_red_learner_sees_mirrored_field():
    env = RebuiltEnv()
    obs_b, _ = env.reset(seed=3, options={"learner_slot": 0})
    obs_r, _ = env.reset(seed=3, options={"learner_slot": 3})
    # both start at the mirrored spot of slot 0, so they see the same own position
    assert np.allclose(obs_b[17:19], obs_r[17:19], atol=0.1)


def test_pettingzoo_api():
    env = RebuiltParallelEnv()
    parallel_api_test(env, num_cycles=50)
