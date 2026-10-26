"""
tests/test_step_cap.py - Verify step cap overrides Gymnasium's silent 1000-step default.
"""

import pytest
from f1sim.env import make_env


def test_step_cap_configured():
    """Verify that make_env explicitly sets max_episode_steps to the requested limit."""
    env = make_env(seed=1000, max_steps=1500)
    assert env.configured_max_steps == 1500
    
    # Check underlying TimeLimit wrapper
    current = env
    found_timelimit = False
    while hasattr(current, "env"):
        if hasattr(current, "_max_episode_steps"):
            assert current._max_episode_steps == 1500
            found_timelimit = True
            break
        current = current.env
    assert found_timelimit, "TimeLimit wrapper with max_episode_steps was not found"
    env.close()


def test_can_run_past_1000_steps():
    """Verify that an episode runs past step 1000 without silent truncation."""
    # Use max_steps=1200
    env = make_env(seed=1000, max_steps=1200)
    obs, info = env.reset(seed=1000)

    step_count = 0
    # Hold brake to stay on track at low speed and advance steps
    for t in range(1050):
        action = [0.0, 0.05, 0.1]  # gentle crawl
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        if terminated or truncated:
            break

    # Must NOT have truncated at 1000
    assert step_count > 1000, f"Episode ended prematurely at step {step_count}"
    assert not truncated, f"Episode was truncated prematurely at step {step_count}"
    assert info["step_count"] > 1000
    env.close()
