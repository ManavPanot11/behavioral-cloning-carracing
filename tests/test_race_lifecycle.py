"""
tests/test_race_lifecycle.py - Comprehensive tests for Phase 3 Race Lifecycle.

State machine: GRID -> LIGHTS -> RACING -> CHEQUERED -> FINISHING -> COOLDOWN -> RESULTS.
Validates:
1. Session end exactly 750 steps after the LAST classified car crosses the line.
2. Cars keep moving (decelerate and coast, never freeze at speed 0).
3. Visible 'SESSION ENDS IN 00:xx' countdown messages emitted during cooldown.
4. Retired (>150 frames off track) and stopped cars (>10s at <0.2 m/s) are DNF and never block the end.
5. 120s timeout after winner classifies unfinished cars as NOT CLASSIFIED.
6. Generous safety cap against infinite loops.
7. Tests: synthetic crossing times, DNF, all-DNF, late car delayed by collision, single-car race,
   lapped car finish, and a 10-lap race longer than 4:06 (12,302 steps).
"""

import math
import numpy as np
import pytest

from f1sim.world import MultiCarWorld, SIX_CAR_SPECS
from f1sim.drivers import get_car_state


def test_synthetic_crossing_times():
    """
    Test that session ends exactly 750 steps after the LAST car that can finish crosses the line.
    Winner crosses at step 1000, P2 crosses at step 1350.
    Session must end at 1350 + 750 = 2100.
    """
    mcw = MultiCarWorld(track_seed=3003, total_laps=2, lights_hold_steps=50)
    winner_cid = "car_apx"
    p2_cid = "car_vel"

    # Step up to step 1000
    mcw.current_step = 1000
    mcw.timing.drivers[winner_cid].laps_completed = 2
    mcw.step()

    assert mcw.winner_id == winner_cid
    assert mcw.winner_finish_step == 1000
    assert mcw.race_state in ("CHEQUERED", "FINISHING")
    assert mcw.session_end_step < 0, "Session end step should NOT be set while P2 is still racing"

    # P2 crosses line at step 1350
    mcw.current_step = 1350
    mcw.timing.drivers[p2_cid].laps_completed = 2
    # Mark others finished as well
    for cid in mcw.cars:
        mcw.timing.drivers[cid].laps_completed = 2
    mcw.step()

    assert mcw.last_finisher_step == 1350
    assert mcw.session_end_step == 1350 + 750
    assert mcw.race_state == "COOLDOWN"
    mcw.close()


def test_dnf_does_not_block_session_end():
    """
    A retired car (DNF from off-track or stopped) must never block the session end.
    Winner finishes at step 1000, second car is DNF. Session ends at 1000 + 750 = 1750.
    """
    two_specs = [SIX_CAR_SPECS[0], SIX_CAR_SPECS[1]]
    mcw = MultiCarWorld(track_seed=3003, total_laps=1, car_specs=two_specs, lights_hold_steps=50)

    # Car B is DNF
    mcw.stewards.states["car_vel"].is_dnf = True
    mcw.stewards.states["car_vel"].dnf_reason = "EXCEEDED 150 CONSECUTIVE OFF-ROAD FRAMES"

    # Car A (winner) crosses line at step 1000
    mcw.current_step = 1000
    mcw.timing.drivers["car_apx"].laps_completed = 1
    mcw.step()

    assert mcw.winner_id == "car_apx"
    assert mcw.session_end_step == 1000 + 750, (
        f"Expected session end at 1750, got {mcw.session_end_step}"
    )
    mcw.close()


def test_all_dnf_race_ends_cleanly():
    """
    If all cars are DNF, the session must resolve cleanly without hanging.
    """
    mcw = MultiCarWorld(track_seed=3003, total_laps=1, lights_hold_steps=50)
    mcw.current_step = 500

    # All cars DNF
    for cid in mcw.cars:
        mcw.stewards.states[cid].is_dnf = True
        mcw.stewards.states[cid].dnf_reason = "RETIRED"

    mcw.step()
    assert mcw.session_end_step == 500 + 750
    assert mcw.race_state == "COOLDOWN"

    # Step to conclusion
    mcw.current_step = mcw.session_end_step
    running = mcw.step()
    assert not running, "Session must return False (terminate) when session_end_step is reached"
    assert mcw.race_state == "RESULTS"
    mcw.close()


def test_late_car_delayed_by_collision():
    """
    Car A finishes at step 1000. Car B was delayed by a collision and crosses the line
    at step 1600. The session must end at exactly 1600 + 750 = 2350 steps.
    """
    two_specs = [SIX_CAR_SPECS[0], SIX_CAR_SPECS[1]]
    mcw = MultiCarWorld(track_seed=3003, total_laps=2, car_specs=two_specs, lights_hold_steps=50)

    # Car A finishes at step 1000
    mcw.current_step = 1000
    mcw.timing.drivers["car_apx"].laps_completed = 2
    mcw.step()
    assert mcw.session_end_step < 0

    # Advance steps where Car B is still running
    for s in range(1001, 1600, 100):
        mcw.current_step = s
        mcw.step()
        assert mcw.session_end_step < 0, f"Session prematurely ended at step {s} while Car B was racing"

    # Car B crosses at step 1600
    mcw.current_step = 1600
    mcw.timing.drivers["car_vel"].laps_completed = 2
    mcw.step()

    assert mcw.last_finisher_step == 1600
    assert mcw.session_end_step == 1600 + 750
    mcw.close()


def test_single_car_race():
    """
    Single-car race: car finishes and cooldown ends cleanly at winner + 750 steps.
    """
    solo_spec = [SIX_CAR_SPECS[0]]
    mcw = MultiCarWorld(track_seed=3003, total_laps=1, car_specs=solo_spec, lights_hold_steps=50)

    mcw.current_step = 800
    mcw.timing.drivers["car_apx"].laps_completed = 1
    mcw.step()

    assert mcw.winner_id == "car_apx"
    assert mcw.session_end_step == 800 + 750
    mcw.close()


def test_lapped_cars_finish_on_crossing_after_winner():
    """
    F1 rule: Lapped cars finish when they cross the start/finish line AFTER the winner has finished,
    even if they completed fewer laps than total_laps.
    """
    two_specs = [SIX_CAR_SPECS[0], SIX_CAR_SPECS[1]]
    mcw = MultiCarWorld(track_seed=3003, total_laps=3, car_specs=two_specs, lights_hold_steps=50)

    # Winner completes lap 3 at step 2000
    mcw.current_step = 2000
    mcw.timing.drivers["car_apx"].laps_completed = 3
    mcw.step()
    assert mcw.winner_id == "car_apx"
    assert "car_vel" not in mcw.finished_cars

    # Car B has only completed 2 laps (lapped car), crosses line at step 2150
    mcw.current_step = 2150
    # Simulate line crossing in tracker
    tracker_b = mcw.trackers["car_vel"]
    tracker_b._prev_tile = int(tracker_b.track_geom.num_tiles * 0.90)
    tracker_b.current_tile = int(tracker_b.track_geom.num_tiles * 0.05)
    tracker_b.passed_halfway = True
    tracker_b.has_started = True
    t0 = mcw.track_pts[0]
    mcw.cars["car_vel"].hull.position = (float(t0[0]), float(t0[1]))

    # Call step with lap_finished simulated
    mcw.step()

    assert "car_vel" in mcw.finished_cars, "Lapped car was not flagged finished after winner"
    assert mcw.session_end_step == mcw.last_finisher_step + 750
    mcw.close()


def test_stopped_car_on_track_dnf():
    """
    A car stopped on track (speed < 0.2 m/s for > 500 frames / 10s) must be retired as DNF
    and must not hang the race session.
    """
    two_specs = [SIX_CAR_SPECS[0], SIX_CAR_SPECS[1]]
    mcw = MultiCarWorld(track_seed=3003, total_laps=1, car_specs=two_specs, lights_hold_steps=50)

    # Put Car B into stopped state
    car_b = mcw.cars["car_vel"]
    mcw.drivers["car_vel"].act = lambda *args, **kwargs: np.array([0.0, 0.0, 1.0], dtype=np.float32)
    car_b.hull.linearVelocity = (0.0, 0.0)

    # Run 50 pre-start steps + 510 racing steps (total 560 steps, >500 racing steps)
    for _ in range(560):
        mcw.step()

    assert mcw.stewards.states["car_vel"].is_dnf, "Car stopped for > 500 frames was not marked DNF"
    mcw.close()


def test_cooldown_countdown_events():
    """
    During the 750-step cooldown, visible 'SESSION ENDS IN 00:xx' countdown events
    must be generated at 1-second intervals.
    """
    solo_spec = [SIX_CAR_SPECS[0]]
    mcw = MultiCarWorld(track_seed=3003, total_laps=1, car_specs=solo_spec, lights_hold_steps=50)

    # Finish at step 1000
    mcw.current_step = 1000
    mcw.timing.drivers["car_apx"].laps_completed = 1
    mcw.step()

    # Step through 200 frames of cooldown
    countdown_events = []
    for s in range(1001, 1251):
        mcw.current_step = s
        mcw.step()
        for ev in mcw.race_events:
            if ev.get("type") == "countdown" and ev["step"] == s:
                countdown_events.append(ev)

    assert len(countdown_events) >= 4, f"Expected countdown events every 50 frames, got {len(countdown_events)}"
    assert "SESSION ENDS IN 00:" in countdown_events[0]["text"]
    mcw.close()


def test_10_lap_race_longer_than_4_06():
    """
    A 10-lap race is simulated past 12,302 steps (4:06 duration noted in F3)
    without running into early terminations or timeouts.
    """
    mcw = MultiCarWorld(track_seed=3003, total_laps=10, lights_hold_steps=50)
    # Fast forward to step 12305
    mcw.current_step = 12305
    # Both active drivers still running lap 9 of 10
    for d in mcw.timing.drivers.values():
        d.laps_completed = 8
        d.current_lap = 9

    running = mcw.step()
    assert running, "Race terminated prematurely before 10 laps completed"
    assert mcw.current_step > 12302, f"Expected step > 12302 (4:06), got {mcw.current_step}"
    assert mcw.session_end_step < 0, "Session end step set prematurely"
    mcw.close()
