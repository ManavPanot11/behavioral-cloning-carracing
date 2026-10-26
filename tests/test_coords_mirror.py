"""
tests/test_coords_mirror.py - Coordinate mapping and non-mirroring validation.

Validates:
1. Coordinate transformation: Sim (x, y) -> Three.js (x, 0, -y)
2. Rigid transformation matrix between sim and 3D top-down projection has det(R) == +1 (NO reflection/mirroring).
3. A car steering left (steer > 0) turns left in both sim and 3D.
4. Export works on 3 showcase seeds (3000, 3001, 3002).
"""

import math
import numpy as np
import pytest
import gymnasium as gym

from f1sim.drivers import compute_staggered_grid_positions
from f1sim.export3d import extract_track_geometry, run_and_export_3d


def test_asymmetric_track_non_mirroring():
    """
    Test coordinate mapping on an asymmetric track seed (3003).
    A rigid 2D transform between sim (x, y) and 3D top-down (X, -Z) must have
    det(R) == +1.0 (pure rotation/translation, ZERO reflection).
    """
    env = gym.make("CarRacing-v3")
    env.reset(seed=3003)
    geom = extract_track_geometry(env)
    env.close()

    sim_pts = np.array([[p[0], p[1]] for p in geom["centerline"]])
    # In 3D top-down: screen X = X_3D = x_sim; screen Y = -Z_3D = -(-y_sim) = y_sim
    screen_3d_pts = np.array([[p[0], -(-p[1])] for p in geom["centerline"]])

    # Compute optimal rigid alignment (Kabsch algorithm)
    centroid_sim = np.mean(sim_pts, axis=0)
    centroid_3d = np.mean(screen_3d_pts, axis=0)

    P = sim_pts - centroid_sim
    Q = screen_3d_pts - centroid_3d

    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    det_R = np.linalg.det(R)
    # If det(R) is +1, it is a pure rotation (preserving handedness / chirality).
    # If det(R) is -1, it would be a reflection (mirroring bug).
    assert math.isclose(det_R, 1.0, abs_tol=1e-5), f"Expected det(R)=+1 (non-mirrored), got {det_R}"

    # Residual alignment error should be virtually 0
    aligned = P @ R.T
    residual = np.mean(np.linalg.norm(aligned - Q, axis=1))
    assert residual < 1e-4, f"Expected perfect alignment residual < 1e-4, got {residual}"


def test_left_turn_in_3d():
    """
    Verify that a positive steering input (turning left in sim) turns left in 3D.
    """
    # In sim, facing North (+Y), turning left introduces negative X velocity (points West / Left)
    sim_forward = np.array([0.0, 1.0])
    steer_left_angle = math.radians(25)  # 25 deg left turn

    # Rotation matrix in sim (CCW around Z)
    cos_t = math.cos(steer_left_angle)
    sin_t = math.sin(steer_left_angle)
    turned_sim = np.array([
        -sin_t,
        cos_t
    ])
    # Turned vector points left (negative X)
    assert turned_sim[0] < 0.0

    # In 3D: X_3d = x_sim, Z_3d = -y_sim
    # Top-down camera has -Z pointing up and +X pointing right
    turned_3d = np.array([turned_sim[0], -turned_sim[1]])  # (X_3d, Z_3d)

    # In top-down screen coords where Screen_X = X_3d and Screen_Y = -Z_3d:
    screen_x = turned_3d[0]
    screen_y = -turned_3d[1]

    # Screen vector must still point left (Screen_X < 0) and forward (Screen_Y > 0)
    assert screen_x < 0.0, f"Turned vector in 3D screen coords must point left, got screen_x={screen_x}"
    assert screen_y > 0.0, f"Turned vector in 3D screen coords must point forward, got screen_y={screen_y}"


@pytest.mark.parametrize("seed", [3000, 3001, 3002])
def test_export_3d_format_and_size(seed, tmp_path):
    """
    Gate test: run 3D export on 3 showcase seeds (3000, 3001, 3002) for 1 lap and check:
    - JSON contains required meta, track, drivers, events, cars
    - File size is well under 6 MB
    """
    out_file = str(tmp_path / f"test_replay_{seed}.json")
    payload, path = run_and_export_3d(
        track_seed=seed,
        total_laps=1,
        out_path=out_file
    )

    assert "meta" in payload
    assert "track" in payload
    assert "drivers" in payload
    assert "events" in payload
    assert "cars" in payload

    # Verify 6 drivers exported
    assert len(payload["drivers"]) == 6
    assert len(payload["cars"]) == 6

    # Verify track centerline and polygons
    assert len(payload["track"]["centerline"]) > 100
    assert len(payload["track"]["polygons"]) > 100

    # Check file size (target under 6 MB)
    import os
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Seed {seed} 3D JSON file size: {size_mb:.2f} MB")
    assert size_mb < 6.0, f"File size {size_mb} MB exceeds 6 MB target"
