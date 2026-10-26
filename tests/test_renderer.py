"""
tests/test_renderer.py - Verifies Pygame Dashboard Renderer across resolutions:
- 1280x720 (720p)
- 1920x1080 (1080p Broadcast standard)
- 2560x1440 (1440p High DPI)
Also generates and saves a 1080p dashboard preview for inspection.
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import numpy as np
import pytest

from f1sim.renderer import DashboardRenderer
from f1sim.env import make_env
from f1sim.timing import TimingSystem
from f1sim.stewards import RaceControlEvent


@pytest.mark.parametrize("width,height", [
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
])
def test_renderer_resolutions(width, height):
    renderer = DashboardRenderer(width=width, height=height)
    assert renderer.width == width
    assert renderer.height == height

    env = make_env(seed=3000, max_steps=200, render_mode="rgb_array")
    obs, info = env.reset(seed=3000)

    # Mock cars state
    cars_state = {
        "car_1": {
            "pos": (0.0, 0.0),
            "heading": 0.0,
            "speed": 55.0,
            "steer": 0.15,
            "gas": 0.8,
            "brake": 0.0,
            "team_code": "APX",
            "code": "VAN",
            "offroad": False,
            "centre_dist": 1.2,
            "lateral_g": 0.45,
        },
        "car_2": {
            "pos": (2.0, -8.0),
            "heading": 0.05,
            "speed": 52.0,
            "steer": -0.10,
            "gas": 0.6,
            "brake": 0.0,
            "team_code": "VEL",
            "code": "SIL",
            "offroad": False,
            "centre_dist": 1.5,
            "lateral_g": 0.35,
        },
        "car_3": {
            "pos": (-2.0, -16.0),
            "heading": -0.02,
            "speed": 50.0,
            "steer": 0.0,
            "gas": 0.5,
            "brake": 0.0,
            "team_code": "NDX",
            "code": "LIN",
            "offroad": False,
            "centre_dist": 1.8,
            "lateral_g": 0.20,
        },
        "car_4": {
            "pos": (2.0, -24.0),
            "heading": 0.0,
            "speed": 48.0,
            "steer": 0.05,
            "gas": 0.5,
            "brake": 0.0,
            "team_code": "ZEN",
            "code": "TAN",
            "offroad": False,
            "centre_dist": 2.0,
            "lateral_g": 0.15,
        },
    }

    timing = TimingSystem()
    for cid, code, name, tcode in [
        ("car_1", "VAN", "Alex Vance", "APX"),
        ("car_2", "SIL", "Marco Silva", "VEL"),
        ("car_3", "LIN", "Lukas Lind", "NDX"),
        ("car_4", "TAN", "Kai Tanaka", "ZEN"),
    ]:
        timing.register_driver(cid, code, name, tcode)

    timing.update_driver_progress("car_1", current_step=100, progress_pct=15.0)
    timing.update_driver_progress("car_2", current_step=100, progress_pct=13.5)
    timing.update_driver_progress("car_3", current_step=100, progress_pct=11.0)
    timing.update_driver_progress("car_4", current_step=100, progress_pct=9.0)

    leaderboard = timing.compute_leaderboard(current_step=100)

    events = [
        RaceControlEvent(100, 2.0, "car_2", "SIL", "WARNING", "CAR 16 (SIL) - TRACK LIMITS WARNING 1/3")
    ]

    surf = renderer.render_frame(
        env=env,
        cars_state=cars_state,
        leaderboard=leaderboard,
        selected_car_id="car_1",
        current_step=100,
        total_laps=10,
        race_events=events,
        active_cam_obs=obs,
    )

    assert surf.get_width() == width
    assert surf.get_height() == height

    # Save a 1080p sample for inspection
    if width == 1920:
        os.makedirs("docs", exist_ok=True)
        preview_path = os.path.join("docs", "dashboard_preview_1080p.png")
        pygame.image.save(surf, preview_path)
        assert os.path.exists(preview_path)
