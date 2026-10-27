"""
eval_suite.py - Unified evaluation suite implementing the protocol in results/protocol.md.
Used for:
- Stage 3: Dev comparison (seeds 1000-1019)
- Stage 4: Disturbance / Recovery test
- Stage 5: Speed stress test (1.0x, 1.25x, 1.50x)
- Stage 6: Final held-out test evaluation (seeds 2000-2019)
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import gymnasium as gym

from expert import (expert_steer, speed_controller, get_car_state,
                    wheels_on_track, LOOKAHEAD, GAIN,
                    TARGET_SPEED_STRAIGHT, TARGET_SPEED_TURN, MIN_TARGET_SPEED)

DEV_SEEDS = list(range(1000, 1020))
TEST_SEEDS = list(range(2000, 2020))
SKIP_FRAMES = 50
MAX_STEPS = 2000
MAX_OFFROAD_STREAK = 150
DISTURBANCE_MAGNITUDE = 0.6
DISTURBANCE_DURATION = 20
RECOVERY_WINDOW = 60
RECOVERY_THRESHOLD = 3.0


def run_episode(
    driver_type,
    model=None,
    seed=1000,
    speed_factor=1.0,
    disturb=False,
    record_frames=False,
    max_steps=MAX_STEPS
):
    """
    Run an episode strictly according to results/protocol.md rules.
    driver_type: 'expert', 'zero', or 'model'
    speed_factor: multiplier on target speeds (straight, turn, min)
    disturb: boolean, applies pre-registered disturbance protocol
    """
    v_straight = TARGET_SPEED_STRAIGHT * speed_factor
    v_turn = TARGET_SPEED_TURN * speed_factor
    v_min = MIN_TARGET_SPEED * speed_factor

    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        max_episode_steps=max_steps
    )
    obs, _ = env.reset(seed=seed)
    uw = env.unwrapped
    total_tiles = len(uw.track)
    waypoints = np.array([[p[2], p[3]] for p in uw.track], dtype=np.float32)

    # Disturbance setup: triggers at waypoint fractions [0.25, 0.50, 0.75]
    triggers = []
    if disturb:
        fractions = [0.25, 0.50, 0.75]
        for idx, frac in enumerate(fractions):
            target_wp = int(frac * total_tiles)
            rng = np.random.RandomState(seed * 1000 + target_wp)
            sign = float(rng.choice([-1.0, 1.0]))
            triggers.append({
                "target_wp": target_wp,
                "offset": sign * DISTURBANCE_MAGNITUDE,
                "triggered": False,
                "trigger_step": -1,
                "end_step": -1,
                "recovered": False,
            })

    speeds = []
    steers = []
    centre_dists = []
    frames = [] if record_frames else None

    offroad_events = 0
    was_on_track = True
    offroad_streak = 0
    longest_offroad_streak = 0
    lap_completed = False
    lost = False
    truncated_limit = False

    active_disturbance_offset = 0.0
    active_disturbance_end = -1

    for step in range(max_steps):
        car_x, car_y, heading, speed = get_car_state(env)
        speeds.append(speed)

        # Centre-line distance
        dists = np.linalg.norm(waypoints - np.array([car_x, car_y], dtype=np.float32), axis=1)
        nearest_idx = int(np.argmin(dists))
        centre_dist = float(dists[nearest_idx])
        if step >= SKIP_FRAMES:
            centre_dists.append(centre_dist)

        # Disturbance trigger check
        if disturb and step >= SKIP_FRAMES:
            for trig in triggers:
                if not trig["triggered"] and nearest_idx >= trig["target_wp"]:
                    trig["triggered"] = True
                    trig["trigger_step"] = step
                    trig["end_step"] = step + DISTURBANCE_DURATION
                    active_disturbance_offset = trig["offset"]
                    active_disturbance_end = trig["end_step"]
                    break

            # Disturbance expiry
            if step >= active_disturbance_end:
                active_disturbance_offset = 0.0

            # Recovery check: within 60 frames after end_step
            for trig in triggers:
                if trig["triggered"] and not trig["recovered"]:
                    if trig["end_step"] <= step <= trig["end_step"] + RECOVERY_WINDOW:
                        if centre_dist <= RECOVERY_THRESHOLD:
                            trig["recovered"] = True

        # Policy computation
        if step < SKIP_FRAMES:
            raw_steer = 0.0
        elif driver_type == "expert":
            raw_steer = expert_steer(env, lookahead=LOOKAHEAD, gain=GAIN)
        elif driver_type == "zero":
            raw_steer = 0.0
        else:
            cropped = obs[:84, :, :].astype(np.float32)[np.newaxis]
            pred = model.predict(cropped, verbose=0)[0, 0]
            raw_steer = float(np.clip(pred, -1.0, 1.0))

        steers.append(raw_steer)

        # Executed steering (with disturbance if active)
        exec_steer = float(np.clip(raw_steer + active_disturbance_offset, -1.0, 1.0))

        # Speed controller
        gas, brake = speed_controller(env, exec_steer, v_straight, v_turn, v_min)
        action = np.array([exec_steer, gas, brake], dtype=np.float32)

        if record_frames:
            rendered = env.render()
            if rendered is not None:
                frames.append(rendered)

        obs, reward, terminated, truncated, info = env.step(action)

        # Track off-road
        n_wheels = wheels_on_track(env)
        currently_on = n_wheels > 0

        if not currently_on:
            offroad_streak += 1
            longest_offroad_streak = max(longest_offroad_streak, offroad_streak)
            if was_on_track:
                offroad_events += 1
        else:
            offroad_streak = 0
        was_on_track = currently_on

        # Lost detection rule
        if offroad_streak >= MAX_OFFROAD_STREAK:
            lost = True
            break

        if terminated:
            tiles_visited = uw.tile_visited_count
            lap_completed = info.get("lap_finished", False) or (tiles_visited / total_tiles >= 0.95 and uw.new_lap)
            break

        if truncated:
            truncated_limit = True
            break

    tiles_visited = uw.tile_visited_count
    track_progress = (tiles_visited / total_tiles) * 100.0
    total_steps = len(speeds)
    env.close()

    # Steering jitter: mean |steer[t] - steer[t-1]| for t >= SKIP_FRAMES
    steers_post = np.array(steers[SKIP_FRAMES:], dtype=np.float32) if len(steers) > SKIP_FRAMES else np.array(steers, dtype=np.float32)
    if len(steers_post) > 1:
        jitter = float(np.mean(np.abs(np.diff(steers_post))))
        mean_abs_steer = float(np.mean(np.abs(steers_post)))
    else:
        jitter = 0.0
        mean_abs_steer = 0.0

    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    mean_centre = float(np.mean(centre_dists)) if centre_dists else 0.0
    max_centre = float(np.max(centre_dists)) if centre_dists else 0.0

    # Ending type
    if lap_completed:
        ending = "LAP"
    elif lost:
        ending = "LOST"
    elif truncated_limit or total_steps >= max_steps:
        ending = "TRUNCATED"
    else:
        ending = "OTHER"

    # Autonomy: (1 - takeovers * 6 / seconds) * 100
    seconds = total_steps / 50.0
    autonomy = float(max(0.0, min(100.0, (1.0 - offroad_events * 6.0 / max(seconds, 1.0)) * 100.0)))

    # Disturbance recovery metric
    if disturb:
        triggered_list = [t for t in triggers if t["triggered"]]
        if triggered_list:
            recovered_all = all(t["recovered"] for t in triggered_list)
        else:
            recovered_all = True
    else:
        recovered_all = True

    return {
        "seed": seed,
        "lap_completed": lap_completed,
        "steps": total_steps,
        "track_progress": track_progress,
        "offroad_events": offroad_events,
        "longest_offroad_streak": longest_offroad_streak,
        "mean_speed": mean_speed,
        "mean_abs_steer": mean_abs_steer,
        "steer_jitter": jitter,
        "mean_centre_distance": mean_centre,
        "max_centre_distance": max_centre,
        "ending_type": ending,
        "autonomy": autonomy,
        "recovered_each_disturbance": recovered_all,
        "frames": frames,
    }
