"""
f1sim/metrics.py - Canonical evaluation metrics implementation for CarRacing-v3

Implements pre-registered metrics defined in results/protocol.md:
- track_progress_pct
- centre_line_distance
- steering_jitter
- offroad_events (discrete state transitions from on-track to off-track)
- autonomy percentage
"""

import numpy as np


def compute_progress_pct(tiles_visited: int, total_tiles: int) -> float:
    """Return track progress percentage [0.0, 100.0]."""
    if total_tiles <= 0:
        return 0.0
    return float(min(100.0, max(0.0, (tiles_visited / total_tiles) * 100.0)))


def compute_centre_line_distance(pos: tuple[float, float], track_coords: np.ndarray) -> float:
    """
    Compute Euclidean distance from vehicle pos (x, y) to nearest waypoint in track_coords.
    track_coords: shape (N, 2)
    """
    if len(track_coords) == 0:
        return 0.0
    diff = track_coords - np.array(pos)
    dists_sq = np.sum(diff ** 2, axis=1)
    return float(np.sqrt(np.min(dists_sq)))


def compute_steering_jitter(steering_history: list[float] | np.ndarray) -> float:
    """
    Compute high-frequency steering oscillation metric:
    jitter = (1 / (T - 1)) * sum(|s_t - s_{t-1}|)
    """
    arr = np.asarray(steering_history, dtype=np.float32)
    if len(arr) < 2:
        return 0.0
    diffs = np.abs(np.diff(arr))
    return float(np.mean(diffs))


def count_offroad_events(wheel_contacts_history: list[int] | np.ndarray) -> int:
    """
    Count discrete off-road events.
    An event is logged each time the car transitions from on-track (wheels > 0)
    to off-track (wheels == 0).
    """
    arr = np.asarray(wheel_contacts_history)
    if len(arr) == 0:
        return 0

    on_track = arr > 0
    events = 0
    was_on = bool(on_track[0])

    for is_on in on_track[1:]:
        if was_on and not is_on:
            events += 1
        was_on = bool(is_on)

    return events


def compute_autonomy_pct(offroad_events: int, elapsed_seconds: float) -> float:
    """
    Standard Nvidia autonomy metric:
    autonomy = max(0.0, min(100.0, (1 - (events * 6 / max(seconds, 1))) * 100))
    """
    sec = max(1.0, float(elapsed_seconds))
    penalty = (offroad_events * 6.0) / sec
    return float(max(0.0, min(100.0, (1.0 - penalty) * 100.0)))
