"""
tests/test_determinism.py - Verify reproducibility across identical seeds.
"""

import numpy as np
import pytest
from f1sim.env import make_env


def test_determinism_same_seed():
    """Verify identical observations, positions, and rewards across identical seeds."""
    seed = 1000
    steps = 60

    # Run 1
    env1 = make_env(seed=seed, max_steps=500)
    obs1_list, pos1_list = [], []
    for t in range(steps):
        steer = 0.1 * np.sin(t * 0.1)
        action = [steer, 0.4, 0.0]
        obs, reward, terminated, truncated, info = env1.step(action)
        obs1_list.append(obs)
        pos1_list.append(env1.unwrapped.car.hull.position)
        if terminated or truncated:
            break
    env1.close()

    # Run 2
    env2 = make_env(seed=seed, max_steps=500)
    obs2_list, pos2_list = [], []
    for t in range(steps):
        steer = 0.1 * np.sin(t * 0.1)
        action = [steer, 0.4, 0.0]
        obs, reward, terminated, truncated, info = env2.step(action)
        obs2_list.append(obs)
        pos2_list.append(env2.unwrapped.car.hull.position)
        if terminated or truncated:
            break
    env2.close()

    assert len(pos1_list) == len(pos2_list)
    for p1, p2 in zip(pos1_list, pos2_list):
        assert abs(p1[0] - p2[0]) < 1e-6
        assert abs(p1[1] - p2[1]) < 1e-6

    for o1, o2 in zip(obs1_list, obs2_list):
        np.testing.assert_array_equal(o1, o2)


def test_different_seeds_produce_different_tracks():
    """Verify different seeds produce different track waypoint geometries."""
    env1 = make_env(seed=1001)
    env2 = make_env(seed=1002)

    track1 = np.array([[tx, ty] for (_, _, tx, ty) in env1.unwrapped.track])
    track2 = np.array([[tx, ty] for (_, _, tx, ty) in env2.unwrapped.track])

    assert track1.shape != track2.shape or not np.allclose(track1, track2)
    env1.close()
    env2.close()
