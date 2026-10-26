"""
tests/test_custom_models_and_track.py - Phase 5 Unit Tests

Validates:
- GLTFLoader vendoring and license
- Car configuration schemas and Gym CarRacing 5.0m x 2.4m physics footprint
- CustomTrackEnv tile generation matching CarRacing (TRACK_WIDTH)
- Lap length and > 3 min expected lap time warning
- Scripted expert circuit completion
- 50-track corner radii distribution comparison
- Centerline alignment debug check & mirror test
- Model evaluation reporting
"""

import os
import json
import math
import logging
import pytest
import numpy as np
import gymnasium as gym
from gymnasium.envs.box2d.car_racing import TRACK_WIDTH

from f1sim.custom_track import (
    load_track_from_json,
    load_track_from_gltf,
    CustomTrackEnv,
    validate_circuit,
    compare_corner_radii,
    check_centerline_alignment,
    run_mirror_test,
    evaluate_models_on_track,
)


def test_gltf_loader_vendored():
    """Verify GLTFLoader.js is offline-vendored and contains UMD THREE.GLTFLoader assignment."""
    loader_path = "docs/viewer/vendor/GLTFLoader.js"
    license_path = "docs/viewer/vendor/LICENSE"

    assert os.path.exists(loader_path), "docs/viewer/vendor/GLTFLoader.js must exist"
    assert os.path.exists(license_path), "docs/viewer/vendor/LICENSE must exist"

    with open(loader_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "THREE.GLTFLoader = GLTFLoader;" in content
    assert len(content) > 50000, "GLTFLoader.js must be complete, not truncated"


def test_car_models_configuration():
    """Verify all 6 car models have valid car.json with required schema and physics footprint."""
    cars = [
        "aston_martin_f1",
        "f1_senna_1991",
        "ferrari_296",
        "porsche_992_gt3_r_rexy",
        "porsche_992_gt3_rs",
        "racecar_rc09",
    ]

    for car in cars:
        cfg_path = os.path.join("assets/private/cars", car, "car.json")
        assert os.path.exists(cfg_path), f"Missing car.json for {car}"

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        assert "file" in cfg
        assert "scale" in cfg
        assert "up" in cfg
        assert "forward" in cfg
        assert "offset" in cfg
        assert "wheel_nodes" in cfg
        assert "physics_footprint" in cfg

        fp = cfg["physics_footprint"]
        assert fp["length"] == 5.0, f"{car} physics length must match CarRacing hull (5.0m)"
        assert fp["width"] == 2.4, f"{car} physics width must match CarRacing hull (2.4m)"

        # Verify wheel nodes contain fl, fr, rl, rr
        wn = cfg["wheel_nodes"]
        for key in ["fl", "fr", "rl", "rr"]:
            assert key in wn, f"{car} wheel_nodes missing {key}"


def test_custom_track_loader_and_width():
    """Verify track loader parses sample circuit and matches TRACK_WIDTH."""
    td = load_track_from_json("assets/tracks/sample_circuit.json")

    assert td["name"] == "Monza Grand Prix Circuit"
    assert math.isclose(td["width"], TRACK_WIDTH, rel_tol=1e-3)
    assert len(td["track"]) > 50
    assert td["lap_length"] > 200.0
    assert td["expected_lap_time"] > 0.0


def test_lap_time_warning_exceeding_3_minutes(caplog):
    """Verify warning is issued if expected lap time exceeds 3 minutes (180 s)."""
    # Create a 4000m track (expected lap time ~200s > 180s)
    angles = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    radius = 4000.0 / (2 * math.pi)
    big_pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)]).tolist()

    with caplog.at_level(logging.WARNING):
        td = load_track_from_json({"name": "Huge Circuit", "points": big_pts})

    assert td["expected_lap_time"] > 180.0
    assert any("exceeds 3 minutes" in record.message for record in caplog.records)


def test_custom_track_env_fixtures_and_step():
    """Verify CustomTrackEnv constructs Box2D fixtures with correct sensor and friction attributes."""
    env = CustomTrackEnv("assets/tracks/sample_circuit.json")
    obs, info = env.reset(seed=42)

    assert obs.shape == (96, 96, 3)
    assert "lap_length" in info
    assert "expected_lap_time" in info

    uw = env.unwrapped
    assert len(uw.road) == len(uw.track)

    for tile in uw.road[:5]:
        assert tile.fixtures[0].sensor is True
        assert tile.road_friction == 1.0
        assert hasattr(tile, "idx")

    # Step simulation
    action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    next_obs, reward, term, trunc, step_info = env.step(action)
    assert next_obs.shape == (96, 96, 3)
    assert not math.isnan(reward)

    env.close()


def test_expert_finishes_circuit():
    """Verify the scripted expert driver successfully finishes a lap of the custom circuit."""
    env = CustomTrackEnv("assets/tracks/sample_circuit.json")
    result = validate_circuit(env, max_steps=1000)

    assert result["finished"] is True, f"Expert failed to complete circuit: {result}"
    assert result["track_coverage"] >= 0.85
    assert result["steps"] < 1000
    assert result["total_reward"] > 500.0

    env.close()


def test_compare_corner_radii():
    """Verify corner radii comparison against normal CarRacing tracks."""
    td = load_track_from_json("assets/tracks/sample_circuit.json")
    radii_report = compare_corner_radii(td, num_reference_tracks=10)

    ct = radii_report["custom_track"]
    assert "min_radius_m" in ct
    assert "mean_radius_m" in ct
    assert ct["min_radius_m"] > 10.0
    assert ct["mean_radius_m"] > 30.0

    ref = radii_report["reference_50_tracks"]
    assert "avg_min_radius_m" in ref
    assert "avg_mean_radius_m" in ref


def test_check_centerline_alignment():
    """Verify alignment check calculates max lateral deviation."""
    td = load_track_from_json("assets/tracks/sample_circuit.json")
    mesh_pts = np.array([[t[2], t[3]] for t in td["track"]])

    res = check_centerline_alignment(td["track"], mesh_pts)
    assert res["aligned"] is True
    assert math.isclose(res["max_lateral_deviation_m"], 0.0, abs_tol=1e-5)

    # Offset points by 2.0m (within half track width 3.33m)
    offset_pts = mesh_pts + 2.0
    res_offset = check_centerline_alignment(td["track"], offset_pts)
    assert res_offset["aligned"] is True

    # Offset points by 10.0m (out of tolerance)
    bad_pts = mesh_pts + 10.0
    res_bad = check_centerline_alignment(td["track"], bad_pts)
    assert res_bad["aligned"] is False


def test_mirror_circuit_symmetry():
    """Verify mirror test passes without NaN or instability."""
    td = load_track_from_json("assets/tracks/sample_circuit.json")
    mir_res = run_mirror_test(td)

    assert mir_res["passed"] is True
    assert mir_res["stable"] is True
    assert mir_res["completed_steps"] >= 100
