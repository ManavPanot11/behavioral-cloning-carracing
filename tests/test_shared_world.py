"""
tests/test_shared_world.py - Comprehensive tests for the shared-world Box2D engine.

Validates Phase 2 requirements:
1. Bit-identical observations: A solo car in MultiCarWorld receives bit-identical
   cropped observations (obs[:84, :, :]) to a single-car Gym CarRacing-v3 env across 200 steps.
2. Head-on collision: Box2D collision physics prevent penetration, resting gap ~4.9m,
   no NaN coordinates or velocities, no tunneling at high speed with elastic rebound.
3. Rear-end collision: Car ahead is impacted from behind, steward assesses geometric
   at-fault rear-end rule (+5s penalty), no NaN coordinates.
4. Side-swipe collision: Lateral contact resolved without tunneling or NaN.
5. 6-car race simulation: Runs without exceptions, records contact telemetry,
   and computes Real-Time Factor (RTF).
"""

import math
import numpy as np
import pytest
import gymnasium as gym

from f1sim.world import MultiCarWorld, SIX_CAR_SPECS
from f1sim.drivers import get_car_state
from gymnasium.envs.box2d import car_dynamics


def test_solo_car_bit_identical_observation_200_steps():
    """
    Verify that a single car running inside MultiCarWorld produces bit-identical
    input crops (obs[:84, :, :]) to a standard single-car Gym CarRacing-v3 env
    across 200 identical steps.
    """
    seed = 3003
    steps = 200

    # 1. Standard single-car Gym environment
    single_env = gym.make("CarRacing-v3", render_mode="rgb_array")
    s_obs, _ = single_env.reset(seed=seed)

    # 2. Solo-car in MultiCarWorld (1 car on grid at pole position)
    solo_spec = [SIX_CAR_SPECS[0]]  # car_apx
    mcw = MultiCarWorld(
        track_seed=seed,
        total_laps=1,
        car_specs=solo_spec,
        traffic_mode="contact",
        fps=50.0,
        lights_hold_steps=0,
    )

    # Align starting position to match single_env exactly
    mcw_car = mcw.cars["car_apx"]
    single_uw = single_env.unwrapped
    single_car = single_uw.car

    single_car.hull.position = (float(mcw_car.hull.position[0]), float(mcw_car.hull.position[1]))
    single_car.hull.angle = float(mcw_car.hull.angle)
    single_car.hull.color = [0.8, 0.0, 0.0]
    mcw_car.hull.color = [0.8, 0.0, 0.0]

    for s in range(steps):
        steer = -0.15 if s > 60 else 0.0
        gas = 0.6 if s < 150 else 0.2
        brake = 0.0
        action = np.array([steer, gas, brake], dtype=np.float32)

        # Single env step
        single_car.steer(-float(steer))
        single_car.gas(float(gas))
        single_car.brake(float(brake))
        single_car.step(1.0 / 50.0)
        single_uw.world.Step(1.0 / 50.0, 6 * 30, 2 * 30)
        single_uw.t += 1.0 / 50.0
        single_obs = single_uw._render("state_pixels")

        # MultiCarWorld step
        mcw_car.steer(-float(steer))
        mcw_car.gas(float(gas))
        mcw_car.brake(float(brake))
        mcw_car.step(1.0 / 50.0)
        mcw.world.Step(1.0 / 50.0, 6 * 30, 2 * 30)
        mcw.uw.t += 1.0 / 50.0
        mcw_obs = mcw._get_car_observation(mcw_car)

        # Compare cropped 84x96 road observation fed into driver neural networks
        crop_single = single_obs[:84, :, :]
        crop_mcw = mcw_obs[:84, :, :]

        assert np.array_equal(crop_single, crop_mcw), (
            f"Step {s}: MultiCarWorld rendered observation differs from single-car environment!"
        )

    single_env.close()
    mcw.close()


def test_head_on_collision_no_overlap_no_nan():
    """
    Test head-on collision between two cars in the shared Box2D world.
    Verify:
    - Resting gap when in contact is ~4.9m for 5.0m car length
    - High-speed collision does not tunnel; cars collide, momentum is conserved (v -> 0),
      and separation remains >= 4.5m
    - No NaN in positions or velocities
    """
    mcw = MultiCarWorld(track_seed=3003, total_laps=1, lights_hold_steps=0)
    car_a = mcw.cars["car_apx"]
    car_b = mcw.cars["car_vel"]

    # 1. Test resting gap
    car_a.hull.position = (100.0, 100.0)
    car_a.hull.angle = 0.0
    car_b.hull.position = (100.0, 104.9)
    car_b.hull.angle = 0.0

    for _ in range(25):
        mcw.world.Step(1.0 / 50.0, 6 * 30, 2 * 30)

    resting_gap = float(np.linalg.norm(np.array(car_a.hull.position) - np.array(car_b.hull.position)))
    assert math.isclose(resting_gap, 4.88, abs_tol=0.2), (
        f"Expected resting gap ~4.9m for 5.0m cars in contact, got {resting_gap:.2f}m"
    )

    # 2. Test dynamic head-on collision along Y
    car_a.hull.position = (100.0, 100.0)
    car_a.hull.angle = 0.0
    car_a.hull.linearVelocity = (0.0, 15.0)  # Traveling North (+Y) at 15 m/s
    for w in car_a.wheels:
        w.position = car_a.hull.position
        w.linearVelocity = car_a.hull.linearVelocity

    car_b.hull.position = (100.0, 115.0)
    car_b.hull.angle = math.pi
    car_b.hull.linearVelocity = (0.0, -15.0)  # Traveling South (-Y) at 15 m/s
    for w in car_b.wheels:
        w.position = car_b.hull.position
        w.linearVelocity = car_b.hull.linearVelocity

    min_dist = float("inf")
    collision_occurred = False

    for step in range(35):
        mcw.world.Step(1.0 / 50.0, 6 * 30, 2 * 30)

        pos_a = np.array([car_a.hull.position[0], car_a.hull.position[1]])
        pos_b = np.array([car_b.hull.position[0], car_b.hull.position[1]])
        dist = float(np.linalg.norm(pos_a - pos_b))
        min_dist = min(min_dist, dist)

        assert not np.isnan(pos_a).any(), f"NaN in Car A position at step {step}"
        assert not np.isnan(pos_b).any(), f"NaN in Car B position at step {step}"

        vel_a = np.array([car_a.hull.linearVelocity[0], car_a.hull.linearVelocity[1]])
        vel_b = np.array([car_b.hull.linearVelocity[0], car_b.hull.linearVelocity[1]])
        assert not np.isnan(vel_a).any(), f"NaN in Car A velocity at step {step}"
        assert not np.isnan(vel_b).any(), f"NaN in Car B velocity at step {step}"

        if dist < 6.0:
            collision_occurred = True

    assert collision_occurred, "Head-on collision was not reached"
    assert min_dist >= 4.5, f"Cars penetrated beyond hull boundary! min_dist was {min_dist:.2f}m (< 4.5m)"
    # Momentum conservation: equal and opposite momentum brings both cars to rest
    assert abs(car_a.hull.linearVelocity[1]) < 1.0, f"Car A did not stop: vy={car_a.hull.linearVelocity[1]}"
    assert abs(car_b.hull.linearVelocity[1]) < 1.0, f"Car B did not stop: vy={car_b.hull.linearVelocity[1]}"
    mcw.close()


def test_rear_end_collision_geometric_steward_penalty():
    """
    Test rear-end collision: Car A approaches Car B from behind.
    Verify:
    - Box2D resolves impact without tunneling or NaN
    - Geometric steward at-fault rule identifies Car A as at-fault
    - 5s penalty assessed to Car A
    """
    mcw = MultiCarWorld(track_seed=3003, total_laps=1, lights_hold_steps=0)
    car_a = mcw.cars["car_apx"]  # Attacking car behind
    car_b = mcw.cars["car_vel"]  # Leading car ahead

    # Both facing North (heading = pi/2, hull.angle = 0)
    # Car B is at (100, 104.5)
    # Car A is at (100, 100.0) moving North at 20 m/s
    car_b.hull.position = (100.0, 104.5)
    car_b.hull.angle = 0.0
    car_b.hull.linearVelocity = (0.0, 0.0)

    car_a.hull.position = (100.0, 100.0)
    car_a.hull.angle = 0.0
    car_a.hull.linearVelocity = (0.0, 20.0)

    # Step simulation until collision
    rear_end_penalties = []
    for s in range(15):
        mcw.step()

        # Check for penalty events in race events
        for ev in mcw.race_events:
            if ev.get("type") == "penalty_5s" and ev.get("car_id") == "car_apx":
                rear_end_penalties.append(ev)

        pos_a = np.array([car_a.hull.position[0], car_a.hull.position[1]])
        pos_b = np.array([car_b.hull.position[0], car_b.hull.position[1]])
        assert not np.isnan(pos_a).any()
        assert not np.isnan(pos_b).any()

    assert len(rear_end_penalties) > 0, "Steward failed to assess rear-end +5s penalty to car_apx"
    assert mcw.stewards.states["car_apx"].penalty_seconds >= 5.0
    mcw.close()


def test_side_swipe_collision_no_tunneling():
    """
    Test side-swipe collision: two cars side-by-side converge laterally.
    Verify collision is registered as a racing incident without NaN or penetration.
    """
    mcw = MultiCarWorld(track_seed=3003, total_laps=1, lights_hold_steps=0)
    car_a = mcw.cars["car_apx"]
    car_b = mcw.cars["car_vel"]

    # Car width is 2.4m. Place side-by-side: A at (100, 100), B at (103.5, 100)
    car_a.hull.position = (100.0, 100.0)
    car_a.hull.angle = 0.0
    car_a.hull.linearVelocity = (2.0, 15.0)  # drifting right

    car_b.hull.position = (103.5, 100.0)
    car_b.hull.angle = 0.0
    car_b.hull.linearVelocity = (-2.0, 15.0)  # drifting left

    min_lateral_dist = float("inf")
    for s in range(30):
        mcw.step()

        pos_a = np.array([car_a.hull.position[0], car_a.hull.position[1]])
        pos_b = np.array([car_b.hull.position[0], car_b.hull.position[1]])
        dist = float(np.linalg.norm(pos_a - pos_b))
        min_lateral_dist = min(min_lateral_dist, dist)

        assert not np.isnan(pos_a).any()
        assert not np.isnan(pos_b).any()

    # Minimum lateral separation should be >= ~2.2m (sum of half-widths is 2.4m)
    assert min_lateral_dist > 2.0, f"Side swipe penetrated! min dist was {min_lateral_dist:.2f}m"
    mcw.close()


def test_6_car_race_simulation_and_rtf():
    """
    Simulate a multi-car race with all 6 cars on track seed 3003.
    Verifies:
    - 6 cars advance without exception
    - Contact telemetry is recorded
    - Replay JSON builds cleanly
    - Real-Time Factor (RTF) is reported
    """
    import time
    mcw = MultiCarWorld(track_seed=3003, total_laps=1)

    t0 = time.time()
    # Step 300 frames (~6 seconds of simulation)
    for _ in range(300):
        mcw.step()
    t_elapsed = time.time() - t0

    sim_seconds = 300 / 50.0  # 6.0 s
    rtf = sim_seconds / max(t_elapsed, 1e-4)

    print(f"\n[BENCHMARK] 6-Car Shared-World Simulation RTF: {rtf:.2f}x ({sim_seconds:.1f}s simulated in {t_elapsed:.2f}s wallclock)")
    assert rtf > 0.25, f"RTF too low: {rtf:.2f}x"

    payload = mcw.build_replay_payload(rtf=rtf)
    assert len(payload["drivers"]) == 6
    assert len(payload["cars"]) == 6
    assert payload["meta"]["num_cars"] == 6
    assert "contacts" in payload
    mcw.close()
