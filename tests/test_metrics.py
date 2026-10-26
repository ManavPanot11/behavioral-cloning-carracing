"""
tests/test_metrics.py - Unit tests for metrics on synthetic geometry.
"""

import numpy as np
import pytest
from f1sim.metrics import (
    compute_progress_pct,
    compute_centre_line_distance,
    compute_steering_jitter,
    count_offroad_events,
    compute_autonomy_pct,
)


def test_progress_pct_synthetic():
    """Verify track progress percentage calculation."""
    assert compute_progress_pct(0, 300) == 0.0
    assert compute_progress_pct(150, 300) == 50.0
    assert compute_progress_pct(300, 300) == 100.0
    # Guard against over-counts
    assert compute_progress_pct(350, 300) == 100.0


def test_centre_line_distance_synthetic():
    """Verify centre-line Euclidean distance on synthetic circle track."""
    # Synthetic circular track of radius 50 centered at origin
    angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    radius = 50.0
    track_coords = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

    # Point directly on the track
    pos_on_track = (50.0, 0.0)
    assert abs(compute_centre_line_distance(pos_on_track, track_coords)) < 1e-4

    # Point 3.5 units outside the track radius
    pos_outside = (53.5, 0.0)
    assert abs(compute_centre_line_distance(pos_outside, track_coords) - 3.5) < 1e-3

    # Point 2.0 units inside the track radius
    pos_inside = (0.0, 48.0)
    assert abs(compute_centre_line_distance(pos_inside, track_coords) - 2.0) < 1e-3


def test_steering_jitter_synthetic():
    """Verify steering jitter calculation on synthetic waveforms."""
    # Constant steering -> 0 jitter
    const_steer = [0.4] * 50
    assert compute_steering_jitter(const_steer) == 0.0

    # Linear ramp with step 0.1 -> jitter 0.1
    ramp = [i * 0.1 for i in range(20)]
    assert abs(compute_steering_jitter(ramp) - 0.1) < 1e-5

    # Alternating square wave [-1.0, 1.0, -1.0, 1.0] -> step delta 2.0
    square_wave = [-1.0, 1.0] * 25
    assert abs(compute_steering_jitter(square_wave) - 2.0) < 1e-5


def test_offroad_events_synthetic():
    """Verify discrete off-road transition counts."""
    # Always on track -> 0 events
    always_on = [4, 4, 3, 2, 4, 4]
    assert count_offroad_events(always_on) == 0

    # 1 event: on -> off -> stays off
    one_event = [4, 4, 0, 0, 0]
    assert count_offroad_events(one_event) == 1

    # 2 events: on -> off -> on -> off -> on
    two_events = [4, 2, 0, 0, 3, 4, 0, 0, 2]
    assert count_offroad_events(two_events) == 2

    # Grazing event (single frame off) -> 1 event
    graze = [4, 4, 0, 4, 4]
    assert count_offroad_events(graze) == 1


def test_autonomy_pct_synthetic():
    """Verify Nvidia autonomy metric computation."""
    # 0 events in 60s -> 100%
    assert compute_autonomy_pct(0, 60.0) == 100.0

    # 1 event (6s penalty) in 60s -> (1 - 6/60) * 100 = 90%
    assert abs(compute_autonomy_pct(1, 60.0) - 90.0) < 1e-5

    # Extreme failures clamped to 0%
    assert compute_autonomy_pct(50, 10.0) == 0.0
