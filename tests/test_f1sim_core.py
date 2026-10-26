"""
tests/test_f1sim_core.py - Comprehensive tests for simulation core:
- Track geometry, sector segmentation, windowed tracking
- Drivers and grid positioning
- Timing system, sector colors, and leaderboard intervals
- Stewards rules engine (track limits, penalties, DNF)
"""

import math
import numpy as np
import pytest

from f1sim.track import Track
from f1sim.drivers import ExpertDriver, TEAMS, compute_staggered_grid_positions
from f1sim.timing import TimingSystem, COLOR_PURPLE, COLOR_GREEN, COLOR_YELLOW
from f1sim.stewards import Stewards, RulesConfig


def test_track_geometry_and_sectors():
    # Synthetic circular track with 30 nodes
    angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    rad = 100.0
    raw_track = []
    for a in angles:
        raw_track.append((0.0, 0.0, rad * np.cos(a), rad * np.sin(a)))

    track = Track(raw_track)
    assert track.num_tiles == 30
    assert track.get_sector(5) == 1
    assert track.get_sector(15) == 2
    assert track.get_sector(25) == 3

    # Test windowed tracker
    # Start at node 0, car moves to near node 2
    near_node_2 = np.array([rad * np.cos(angles[2]), rad * np.sin(angles[2])])
    nearest = track.find_nearest_tile_windowed(near_node_2, current_idx=0)
    assert nearest == 2

    # Verify teleport prevention: target near node 20 cannot jump if window is [-6, +10]
    near_node_20 = np.array([rad * np.cos(angles[20]), rad * np.sin(angles[20])])
    clamped = track.find_nearest_tile_windowed(near_node_20, current_idx=0, forward_window=10)
    assert clamped != 20  # Cannot jump 20 tiles forward

    # Test centre line distance on track
    dist, signed = track.compute_centre_line_distance(near_node_2, tile_idx=2)
    assert dist < 0.1


def test_staggered_grid_positions():
    angles = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    nodes = np.array([[100.0 * np.cos(a), 100.0 * np.sin(a)] for a in angles], dtype=np.float32)

    grid = compute_staggered_grid_positions(nodes, grid_slots=4, spacing_meters=6.0, lateral_offset=1.5)
    assert len(grid) == 4
    for slot_idx, slot in enumerate(grid):
        assert slot["slot"] == slot_idx + 1
        assert "pos" in slot and len(slot["pos"]) == 2
        assert "heading" in slot


def test_timing_system_sectors_and_intervals():
    timing = TimingSystem(fps=50, num_checkpoints=100)
    d1 = timing.register_driver("car_1", "VAN", "Alex Vance", "APX")
    d2 = timing.register_driver("car_2", "SIL", "Marco Silva", "VEL")

    # Lap 1: car 1 advances faster than car 2
    for step in range(100):
        # Car 1 completes 40% of track (S1 finished)
        timing.update_driver_progress("car_1", current_step=step, progress_pct=step * 0.4)
        # Car 2 completes 20% of track
        timing.update_driver_progress("car_2", current_step=step, progress_pct=step * 0.2)

    # Car 1 completed Sector 1
    assert d1.current_sectors[0] is not None
    assert d1.current_sectors[0].status == "purple"  # Session best

    # Compute leaderboard
    lb = timing.compute_leaderboard(current_step=100)
    assert len(lb) == 2
    assert lb[0]["code"] == "VAN"
    assert lb[0]["pos"] == 1
    assert lb[1]["code"] == "SIL"
    assert lb[1]["pos"] == 2


def test_stewards_rules_engine():
    config = RulesConfig(max_track_limit_warnings=3, penalty_seconds_first=5.0, dnf_offroad_streak=150)
    stewards = Stewards(config=config, fps=50)
    stewards.register_car("car_1", "VAN")

    # Excursion 1: warning 1
    evs1 = stewards.check_step(step=10, car_id="car_1", is_offroad=True, speed=20.0)
    assert len(evs1) == 1
    assert evs1[0].level == "WARNING"

    # Return on track
    stewards.check_step(step=20, car_id="car_1", is_offroad=False, speed=20.0)

    # Excursion 2: warning 2
    evs2 = stewards.check_step(step=30, car_id="car_1", is_offroad=True, speed=20.0)
    assert len(evs2) == 1
    assert evs2[0].level == "WARNING"

    # Return on track
    stewards.check_step(step=40, car_id="car_1", is_offroad=False, speed=20.0)

    # Excursion 3: Black & White Flag (final warning)
    evs3 = stewards.check_step(step=50, car_id="car_1", is_offroad=True, speed=20.0)
    assert len(evs3) == 1
    assert evs3[0].level == "FLAG"

    # Return on track
    stewards.check_step(step=60, car_id="car_1", is_offroad=False, speed=20.0)

    # Excursion 4: 5s time penalty
    evs4 = stewards.check_step(step=70, car_id="car_1", is_offroad=True, speed=20.0)
    assert len(evs4) == 1
    assert evs4[0].level == "PENALTY"
    assert stewards.states["car_1"].penalty_seconds == 5.0

    # Extended excursion: 150 consecutive offroad frames -> DNF
    for s in range(71, 71 + 155):
        stewards.check_step(step=s, car_id="car_1", is_offroad=True, speed=10.0)

    assert stewards.states["car_1"].is_dnf is True
    assert stewards.states["car_1"].dnf_reason is not None
