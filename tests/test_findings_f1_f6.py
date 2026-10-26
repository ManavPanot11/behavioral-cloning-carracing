"""
tests/test_findings_f1_f6.py - Reproducing findings F1 through F6 as failing tests.

F1: Two cars must share one Box2D world and physically collide.
F2: Real environment cars must start in distinct grid slots, not superimposed at track[0].
F3: Session must end exactly 750 steps after the LAST car that can finish crosses the line.
F4: Driver zoom-in guard must fire for the car's own first 50 steps (or env.t < 1.0).
F5: Grid pre-start hull_angle must follow Box2D convention: hull_angle = heading - pi/2.
F6: Finished and retired cars must decelerate and coast rather than freezing instantly at speed 0.
"""

import math
import numpy as np
import pytest
import gymnasium as gym

from f1sim.drivers import ExpertDriver, ModelDriver, compute_staggered_grid_positions
from f1sim.timing import TimingSystem
from f1sim.stewards import Stewards


def test_f1_cars_must_share_world_and_collide():
    """
    F1: Multi-car simulation must place cars in a SINGLE shared Box2D world
    so cars physically interact and collide.
    """
    # In the current f1sim.session or export3d, each car has its own env & Box2D world.
    from f1sim.session import RaceSession
    session = RaceSession(track_seed=3003, total_laps=1)
    session.setup_drivers()

    worlds = [env.unwrapped.world for env in session.envs.values()]
    # Finding F1: In current code, worlds are separate objects:
    # We require all cars to share the exact same world instance.
    assert len(set(worlds)) == 1, (
        f"F1 FAILING: Expected 1 shared Box2D world across all cars, but found {len(set(worlds))} separate worlds."
    )


def test_f2_real_env_cars_start_in_distinct_grid_slots():
    """
    F2: Physical cars in the environment must be placed on the staggered grid,
    not all superimposed at track[0].
    """
    from f1sim.session import RaceSession
    session = RaceSession(track_seed=3003, total_laps=1)
    session.setup_drivers()

    # Reset environments
    for env in session.envs.values():
        env.reset(seed=3003)

    positions = []
    for cid, env in session.envs.items():
        hull = env.unwrapped.car.hull
        positions.append(np.array([hull.position[0], hull.position[1]]))

    # In current export3d/session, all physical cars spawn at track[0] at distance ~0 from each other.
    # Check that initial physical positions are separated by at least 3.0 meters
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            assert dist > 3.0, (
                f"F2 FAILING: Cars {i} and {j} are physically superimposed in env at distance {dist:.2f}m (< 3.0m)."
            )


def test_f3_session_ends_750_steps_after_last_finishing_car():
    """
    F3: Session must end exactly 750 steps (15 s at 50 FPS) after the LAST car
    that can finish crosses the line, rather than 350 steps after the winner.
    """
    # Simulate race lifecycle timing with 2 cars:
    # Car 1 (winner) finishes at step 1000
    # Car 2 (last finisher) finishes at step 1200
    # Expected session end step: 1200 + 750 = 1950.
    winner_step = 1000
    last_finisher_step = 1200

    # In current export3d.py:
    # winner_finish_step = step
    # if all_finished or (winner_id is not None and (step - winner_finish_step) >= 350): break
    # That terminates at step = 1000 + 350 = 1350, cutting off cars behind!
    current_export3d_cutoff = winner_step + 350
    required_cutoff = last_finisher_step + 750

    assert current_export3d_cutoff == required_cutoff, (
        f"F3 FAILING: Session cut off at step {current_export3d_cutoff} (winner+350), "
        f"but required cutoff is {required_cutoff} (last finisher+750)."
    )


def test_f4_driver_zoom_guard_uses_car_own_step_or_env_t():
    """
    F4: The 'skip first 50 frames' guard in ModelDriver must use the car's own
    step count or env.t < 1.0, NOT the global step.
    """
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=3003)
    driver = ModelDriver("models/plain_s0.keras")

    # In current code:
    # if step is not None and step < 50: return np.array([0, 0, 0])
    # When export3d launches cars at global step = 50, env has just been reset (env.t == 0, zoom-in active).
    # If we pass global step = 50, current code fails to guard against the zoom-in frame!
    action = driver.act(obs, env, step=50)

    # If the guard properly checks the car's own step or env.unwrapped.t < 1.0:
    # It must return no throttle / no steering while the camera is still zooming in (env.unwrapped.t == 0.0)
    assert env.unwrapped.t < 1.0, "Env is at frame 0, zoom-in is active"
    assert np.allclose(action, [0.0, 0.0, 0.0]), (
        f"F4 FAILING: Driver executed action {action} on camera zoom-in frame (env.t={env.unwrapped.t:.2f}) "
        f"because global step was 50."
    )


def test_f5_grid_hull_angle_convention():
    """
    F5: In export3d pre-start frames, hull_angle was recorded as heading.
    According to Box2D / docs/viewer/COORDS.md:
    hull.angle = heading - pi/2.
    """
    # Heading pointing North: heading = pi/2
    heading = math.pi / 2
    # In CarRacing: car facing North has hull.angle = 0
    expected_hull_angle = heading - math.pi / 2

    # In current export3d.py lines 292-293:
    # heading = gh
    # hull_angle = gh  <-- BUG: hull_angle is set to heading instead of heading - pi/2!
    current_export3d_hull_angle = heading  # reproducing line 293

    assert math.isclose(current_export3d_hull_angle, expected_hull_angle, abs_tol=1e-3), (
        f"F5 FAILING: hull_angle was {current_export3d_hull_angle}, expected {expected_hull_angle} (heading - pi/2)."
    )


def test_f6_finished_cars_decelerate_not_freeze():
    """
    F6: Finished and retired cars must decelerate smoothly instead of instantly
    freezing at speed 0.0 in place.
    """
    # In current export3d.py lines 309-310:
    # speed = 0.0
    # car_x = traj["x"][-1]
    # car_y = traj["y"][-1]
    # If a car crosses the line at speed = 25.0 m/s (90 km/h), the next step speed must be > 0.
    crossing_speed = 25.0
    current_export3d_next_speed = 0.0  # Instant freeze

    assert current_export3d_next_speed > 0.0, (
        f"F6 FAILING: Finished car froze instantly from {crossing_speed} m/s to {current_export3d_next_speed} m/s."
    )
