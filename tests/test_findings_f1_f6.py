"""
tests/test_findings_f1_f6.py - Verification tests for findings F1 through F6.

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

from f1sim.drivers import ExpertDriver, ModelDriver, compute_staggered_grid_positions, get_car_state
from f1sim.session import RaceSession
from f1sim.world import MultiCarWorld


def test_f1_cars_must_share_world_and_collide():
    """
    F1: Multi-car simulation must place cars in a SINGLE shared Box2D world
    so cars physically interact and collide.
    """
    session = RaceSession(track_seed=3003, total_laps=1)
    session.setup_drivers()

    worlds_this = [str(env.unwrapped.world.this) for env in session.envs.values()]
    assert len(set(worlds_this)) == 1, (
        f"Expected 1 shared Box2D world across all cars, but found {len(set(worlds_this))} separate worlds."
    )

    # Verify that in MultiCarWorld, cars share the world C++ pointer
    mcw = session.mcw
    assert mcw is not None
    car_ids = list(mcw.cars.keys())
    car0 = mcw.cars[car_ids[0]]
    car1 = mcw.cars[car_ids[1]]
    assert car0.hull.world.this == mcw.world.this
    assert car1.hull.world.this == mcw.world.this
    assert car0.hull.world.this == car1.hull.world.this
    mcw.close()


def test_f2_real_env_cars_start_in_distinct_grid_slots():
    """
    F2: Physical cars in the environment must be placed on the staggered grid,
    not all superimposed at track[0].
    """
    session = RaceSession(track_seed=3003, total_laps=1)
    session.setup_drivers()

    # Reset environments
    for env in session.envs.values():
        env.reset(seed=3003)

    positions = []
    for cid, env in session.envs.items():
        hull = env.unwrapped.car.hull
        positions.append(np.array([hull.position[0], hull.position[1]]))

    # Verify that initial physical positions are separated by at least 3.0 meters
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            assert dist > 3.0, (
                f"Cars {i} and {j} are physically superimposed in env at distance {dist:.2f}m (< 3.0m)."
            )
    session.mcw.close()


def test_f3_session_ends_750_steps_after_last_finishing_car():
    """
    F3: Session must end exactly 750 steps (15 s at 50 FPS) after the LAST car
    that can finish crosses the line, rather than 350 steps after the winner.
    """
    mcw = MultiCarWorld(track_seed=3003, total_laps=2)
    # Simulate race lifecycle timing with 2 cars:
    # Car 1 (winner) finishes at step 1000
    # Car 2 (last finisher) finishes at step 1200
    # Expected session end step: 1200 + 750 = 1950.
    winner_step = 1000
    last_finisher_step = 1200

    mcw.winner_id = "car_apx"
    mcw.winner_finish_step = winner_step

    # Set all drivers completed at last_finisher_step
    for d in mcw.timing.drivers.values():
        d.laps_completed = 2

    # Advance one step to trigger session end resolution
    mcw.current_step = last_finisher_step
    mcw.step()

    required_cutoff = last_finisher_step + 750
    assert mcw.session_end_step == required_cutoff, (
        f"Expected session end step {required_cutoff} (last finisher + 750), got {mcw.session_end_step}"
    )
    mcw.close()


def test_f4_driver_zoom_guard_uses_car_own_step_or_env_t():
    """
    F4: The 'skip first 50 frames' guard in ModelDriver must use the car's own
    step count or env.t < 1.0, NOT the global step.
    """
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=3003)
    driver = ModelDriver("models/plain_s0.keras")

    # When launched at global step = 50, env has just been reset (env.t == 0, zoom-in active).
    # Guard properly checks env.unwrapped.t < 1.0 and returns 0 throttle
    action = driver.act(obs, env, step=50)

    assert env.unwrapped.t < 1.0, "Env is at frame 0, zoom-in is active"
    assert np.allclose(action, [0.0, 0.0, 0.0]), (
        f"Driver executed action {action} on camera zoom-in frame (env.t={env.unwrapped.t:.2f}) "
        f"because global step was 50."
    )
    env.close()


def test_f5_grid_hull_angle_convention():
    """
    F5: In export3d pre-start frames, hull_angle was recorded as heading.
    According to Box2D / docs/viewer/COORDS.md:
    hull.angle = heading - pi/2.
    """
    mcw = MultiCarWorld(track_seed=3003, total_laps=1)
    # Check that each car's hull.angle is equal to heading - pi/2
    for i, (cid, car) in enumerate(mcw.cars.items()):
        g_heading = mcw.grid_data[i]["heading"]
        expected_hull_angle = g_heading - math.pi / 2.0
        actual_hull_angle = float(car.hull.angle)
        assert math.isclose(actual_hull_angle, expected_hull_angle, abs_tol=1e-3), (
            f"Car {cid} hull.angle was {actual_hull_angle}, expected {expected_hull_angle} (heading - pi/2)."
        )
    mcw.close()


def test_f6_finished_cars_decelerate_not_freeze():
    """
    F6: Finished and retired cars must decelerate smoothly instead of instantly
    freezing at speed 0.0 in place.
    """
    mcw = MultiCarWorld(track_seed=3003, total_laps=1)
    # Simulate lights-out so cars are active
    mcw.current_step = 60
    cid = "car_apx"
    car = mcw.cars[cid]

    # Give car initial velocity
    car.hull.linearVelocity = (15.0, 0.0)
    x0, y0, h0, speed0 = get_car_state(car)
    assert speed0 > 10.0

    # Mark car as finished
    mcw.timing.drivers[cid].laps_completed = 1

    # Execute step: car should decelerate with brake rather than jumping to speed 0
    mcw.step()

    x1, y1, h1, speed1 = get_car_state(car)
    assert speed1 > 0.0, f"Finished car froze instantly to speed {speed1}"
    assert speed1 < speed0, f"Finished car did not decelerate: speed0={speed0:.2f}, speed1={speed1:.2f}"
    mcw.close()
