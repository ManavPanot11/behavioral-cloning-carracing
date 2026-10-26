"""
tests/test_track_crossval.py - Cross-validate f1sim.track.Track against Gymnasium native tracking
on dev tracks 1000-1004.
"""

import numpy as np
import pytest
from f1sim.env import make_env
from f1sim.track import Track
from expert import expert_steer, speed_controller, get_car_state


@pytest.mark.parametrize("seed", [1000, 1001, 1002, 1003, 1004])
def test_track_progress_matches_gym(seed):
    env = make_env(seed=seed, max_steps=400, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)
    uw = env.unwrapped

    track = Track(uw.track)
    curr_tile_idx = 0

    for step in range(350):
        car_x, car_y, heading, speed = get_car_state(env)
        steer = expert_steer(env)
        gas, brake = speed_controller(env, steer)
        action = np.array([steer, gas, brake], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        # Update windowed tracker
        curr_tile_idx = track.find_nearest_tile_windowed((car_x, car_y), curr_tile_idx)

        # Check: windowed index should stay within ±5 tiles of gym's tile_visited_count
        visited = uw.tile_visited_count
        diff = abs(curr_tile_idx - visited)
        # Account for start boundary or slight lead/lag
        assert diff < 8 or abs(diff - track.num_tiles) < 8, (
            f"Step {step}: Tracker idx {curr_tile_idx} diverged from Gym visited {visited}"
        )

    env.close()
