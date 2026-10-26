"""
tests/test_lap_logic.py - Verify lap completion and out-of-bounds termination distinction.
"""

import numpy as np
import pytest
from f1sim.env import make_env
from expert import expert_steer, speed_controller, LOOKAHEAD, GAIN, TARGET_SPEED_STRAIGHT, TARGET_SPEED_TURN, MIN_TARGET_SPEED


@pytest.mark.parametrize("seed", [1000, 1001, 1002, 1003, 1004])
def test_expert_completes_dev_tracks(seed):
    """Verify expert driver completes dev tracks 1000-1004 with info['lap_finished'] == True."""
    env = make_env(seed=seed, max_steps=2000)
    obs, info = env.reset(seed=seed)

    lap_finished = False
    for step in range(2000):
        if step < 50:
            steer = 0.0
        else:
            steer = expert_steer(env, lookahead=LOOKAHEAD, gain=GAIN)

        gas, brake = speed_controller(
            env, steer, TARGET_SPEED_STRAIGHT, TARGET_SPEED_TURN, MIN_TARGET_SPEED
        )
        action = [steer, gas, brake]
        obs, reward, terminated, truncated, info = env.step(action)

        if info.get("lap_finished", False):
            lap_finished = True

        if terminated or truncated:
            break

    assert lap_finished, f"Track seed {seed} was not completed as a lap by the expert"
    assert not info.get("out_of_bounds", True), f"Track seed {seed} erroneously marked out-of-bounds"
    assert info.get("progress_pct", 0) >= 95.0
    env.close()


def test_leaving_playfield_is_not_counted_as_lap():
    """
    Verify that when a car leaves the playfield (abs(x) > PLAYFIELD),
    terminated=True is triggered by Gym, but info['lap_finished'] is strictly False.
    """
    env = make_env(seed=1000, max_steps=500)
    obs, info = env.reset(seed=1000)

    # Teleport car far outside the playfield (PLAYFIELD = 333.33)
    from gymnasium.envs.box2d import car_racing
    uw = env.unwrapped
    oob_coord = car_racing.PLAYFIELD + 50.0
    uw.car.hull.position = (oob_coord, oob_coord)
    uw.car.hull.linearVelocity = (0.0, 0.0)

    action = [0.0, 0.0, 0.0]
    obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is True, "Out of bounds car must trigger terminated=True"
    assert info["out_of_bounds"] is True, "info['out_of_bounds'] must be True"
    assert info["lap_finished"] is False, "Leaving playfield must NEVER count as lap_finished"
    env.close()
