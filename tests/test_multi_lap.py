"""
tests/test_multi_lap.py - Verifies that race_mode=True accurately counts multiple laps
without double-counting or infinite-lap loops.
"""

import numpy as np
import pytest
from f1sim.env import make_env
from expert import expert_steer, speed_controller


def test_two_laps_race_mode():
    # Use seed 1009 (short track, ~860 steps per lap)
    env = make_env(seed=1009, max_steps=2500, render_mode="rgb_array", race_mode=True)
    obs, info = env.reset(seed=1009)

    laps_seen = 0
    steps_at_laps = []

    for step in range(2200):
        if step < 50:
            action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            steer = expert_steer(env)
            gas, brake = speed_controller(env, steer)
            action = np.array([steer, gas, brake], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        if info.get("lap_finished", False):
            laps_seen += 1
            steps_at_laps.append(step + 1)
            print(f"Lap {laps_seen} finished at step {step + 1}")

        if laps_seen >= 2:
            break

    env.close()

    assert laps_seen == 2, f"Expected 2 laps completed, got {laps_seen}"
    # Verify laps are separated by at least 700 steps
    assert steps_at_laps[1] - steps_at_laps[0] > 700, f"Laps too close together: {steps_at_laps}"
    print(f"Two-lap test PASSED! Laps at steps: {steps_at_laps}")
