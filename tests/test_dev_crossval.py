"""
tests/test_dev_crossval.py - Fast cross-validation of plain_s0 on dev tracks 1000-1002
verifying bit-identical step counts and centre-line distance against results/dev_final.csv.
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pytest
import keras

from f1sim.env import make_env
from f1sim.metrics import compute_centre_line_distance
from expert import speed_controller, get_car_state


@pytest.fixture(scope="module")
def model():
    model_path = os.path.join("models", "plain_s0.keras")
    assert os.path.exists(model_path), f"Model not found: {model_path}"
    return keras.models.load_model(model_path)


@pytest.mark.parametrize("seed,expected_steps,expected_cdist", [
    (1000, 1091, 1.96),
    (1001, 1215, 1.81),
    (1002, 914, 2.28),
])
def test_plain_s0_reproducibility(model, seed, expected_steps, expected_cdist):
    env = make_env(seed=seed, max_steps=2000, render_mode="rgb_array")
    uw = env.unwrapped
    waypoints = np.array([[p[2], p[3]] for p in uw.track], dtype=np.float32)

    obs, _ = env.reset(seed=seed)
    centre_dists = []
    lap_ok = False

    for step in range(2000):
        car_x, car_y, heading, speed = get_car_state(env)
        if step >= 50:
            cdist = compute_centre_line_distance(np.array([car_x, car_y]), waypoints)
            centre_dists.append(cdist)

        if step < 50:
            pred_steer = 0.0
        else:
            crop = obs[:84, :, :].astype(np.float32)
            crop_batch = np.expand_dims(crop, axis=0)
            pred_steer = float(model(crop_batch, training=False).detach().cpu().numpy()[0, 0])
            pred_steer = float(np.clip(pred_steer, -1.0, 1.0))

        gas, brake = speed_controller(env, pred_steer)
        action = np.array([pred_steer, gas, brake], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        if info.get("lap_finished", False):
            lap_ok = True
            break
        if terminated or truncated:
            break

    env.close()
    mean_cdist = float(np.mean(centre_dists)) if centre_dists else 0.0

    assert lap_ok is True, f"Seed {seed} failed to complete lap"
    assert step + 1 == expected_steps, f"Seed {seed} steps mismatch: {step + 1} vs {expected_steps}"
    assert abs(mean_cdist - expected_cdist) < 0.05, f"Seed {seed} cdist mismatch: {mean_cdist} vs {expected_cdist}"
