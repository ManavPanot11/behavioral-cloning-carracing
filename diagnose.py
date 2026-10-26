"""
diagnose.py
Detailed diagnostic script for CarRacing-v3 behavioral cloning.
Evaluates expert, zero, plain_s0, aug_s0, dagger_s0 on DEV TRACKS (seeds 1000-1019) ONLY.
Evaluates both at max_steps=1000 and max_steps=1500.
Computes track progress (% of tiles visited), ending type, speed, steering jitter, etc.
"""
import os, sys, math, time
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import gymnasium as gym
import keras

from expert import (expert_steer, speed_controller, get_car_state,
                    wheels_on_track, LOOKAHEAD, GAIN,
                    TARGET_SPEED_STRAIGHT, TARGET_SPEED_TURN, MIN_TARGET_SPEED)

DEV_SEEDS = list(range(1000, 1020))
SKIP_FRAMES = 50
MAX_OFFROAD_STREAK = 150


def run_episode(driver_type, model, seed, max_steps):
    """
    Run single episode on track seed with specified max_steps.
    driver_type: 'expert', 'zero', 'model'
    """
    # Create env with explicit max_episode_steps to allow testing 1000 and 1500
    env = gym.make("CarRacing-v3", render_mode="rgb_array",
                   lap_complete_percent=0.95, max_episode_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    
    uw = env.unwrapped
    total_tiles = len(uw.track)
    
    speeds = []
    steers = []
    offroad_events = 0
    was_on_track = True
    offroad_streak = 0
    max_offroad = 0
    lap_completed = False
    lost = False
    truncated_limit = False
    
    for step in range(max_steps):
        _, _, _, speed = get_car_state(env)
        speeds.append(speed)
        
        if step < SKIP_FRAMES:
            steer = 0.0
        elif driver_type == "expert":
            steer = expert_steer(env, lookahead=LOOKAHEAD, gain=GAIN)
        elif driver_type == "zero":
            steer = 0.0
        else:
            cropped = obs[:84, :, :].astype(np.float32)[np.newaxis]
            pred = model.predict(cropped, verbose=0)[0, 0]
            steer = float(np.clip(pred, -1.0, 1.0))
        
        steers.append(steer)
        
        gas, brake = speed_controller(env, steer,
                                    TARGET_SPEED_STRAIGHT, TARGET_SPEED_TURN,
                                    MIN_TARGET_SPEED)
        action = np.array([steer, gas, brake], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Track off-road
        n_wheels = wheels_on_track(env)
        currently_on = n_wheels > 0
        
        if not currently_on:
            offroad_streak += 1
            max_offroad = max(max_offroad, offroad_streak)
            if was_on_track:
                offroad_events += 1
        else:
            offroad_streak = 0
        was_on_track = currently_on
        
        if offroad_streak >= MAX_OFFROAD_STREAK:
            lost = True
            break
        
        if terminated:
            # Check lap finish
            lap_completed = info.get("lap_finished", False) or (uw.tile_visited_count / total_tiles >= 0.95 and uw.new_lap)
            break
        
        if truncated:
            truncated_limit = True
            break
    
    tiles_visited = uw.tile_visited_count
    track_progress = (tiles_visited / total_tiles) * 100.0
    
    env.close()
    
    # Calculate steering jitter: mean |steer[t] - steer[t-1]|
    steers_arr = np.array(steers[SKIP_FRAMES:]) if len(steers) > SKIP_FRAMES else np.array(steers)
    if len(steers_arr) > 1:
        jitter = float(np.mean(np.abs(np.diff(steers_arr))))
        mean_abs_steer = float(np.mean(np.abs(steers_arr)))
    else:
        jitter = 0.0
        mean_abs_steer = 0.0
    
    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    
    # Ending type
    if lap_completed:
        ending = "LAP"
    elif lost:
        ending = "LOST"
    elif truncated_limit or len(speeds) >= max_steps:
        ending = "TRUNCATED"
    else:
        ending = "OTHER"
    
    return {
        "seed": seed,
        "steps": len(speeds),
        "lap_completed": lap_completed,
        "ending": ending,
        "tiles_visited": tiles_visited,
        "total_tiles": total_tiles,
        "track_progress": track_progress,
        "mean_speed": mean_speed,
        "mean_abs_steer": mean_abs_steer,
        "steer_jitter": jitter,
        "offroad_events": offroad_events,
        "max_offroad": max_offroad,
    }


def evaluate_driver(name, driver_type, model_path=None, max_steps=1000):
    model = None
    if driver_type == "model":
        model = keras.models.load_model(model_path)
    
    results = []
    for seed in DEV_SEEDS:
        res = run_episode(driver_type, model, seed, max_steps)
        results.append(res)
    return results


def print_summary(name, max_steps, results):
    laps = sum(1 for r in results if r["lap_completed"])
    mean_prog = np.mean([r["track_progress"] for r in results])
    mean_speed = np.mean([r["mean_speed"] for r in results])
    mean_abs_s = np.mean([r["mean_abs_steer"] for r in results])
    mean_jit = np.mean([r["steer_jitter"] for r in results])
    mean_off = np.mean([r["offroad_events"] for r in results])
    
    endings = {}
    for r in results:
        e = r["ending"]
        endings[e] = endings.get(e, 0) + 1
    
    print(f"\n--- {name} (max_steps={max_steps}) ---")
    print(f"  Laps completed: {laps}/20 ({laps/20*100:.1f}%)")
    print(f"  Mean track progress: {mean_prog:.1f}%")
    print(f"  Mean speed: {mean_speed:.2f}")
    print(f"  Mean |steer|: {mean_abs_s:.3f}")
    print(f"  Steer jitter: {mean_jit:.4f}")
    print(f"  Mean offroad events: {mean_off:.1f}")
    print(f"  Ending breakdown: {endings}")
    
    # Detailed non-lap endings
    non_laps = [r for r in results if not r["lap_completed"]]
    if non_laps:
        print(f"  Failed episodes ({len(non_laps)}):")
        for r in non_laps:
            print(f"    Seed {r['seed']}: ending={r['ending']:9s} steps={r['steps']:4d} "
                  f"progress={r['track_progress']:5.1f}% ({r['tiles_visited']}/{r['total_tiles']}) "
                  f"speed={r['mean_speed']:4.1f} offroad={r['offroad_events']}")


def main():
    print("="*70)
    print("RUNNING DEV TRACK DIAGNOSTICS (SEEDS 1000-1019)")
    print("="*70)
    
    drivers = [
        ("expert", "expert", None),
        ("zero", "zero", None),
        ("plain_s0", "model", "models/plain_s0.keras"),
        ("aug_s0", "model", "models/aug_s0.keras"),
        ("dagger_s0", "model", "models/dagger_s0.keras"),
    ]
    
    all_results = {}
    
    # Test at max_steps = 1000
    print("\n" + "#"*70)
    print("EVALUATION AT max_steps = 1000")
    print("#"*70)
    for name, dtype, path in drivers:
        t0 = time.time()
        res = evaluate_driver(name, dtype, path, max_steps=1000)
        all_results[(name, 1000)] = res
        print_summary(name, 1000, res)
        print(f"  (Took {time.time()-t0:.1f}s)")
    
    # Test at max_steps = 1500
    print("\n" + "#"*70)
    print("EVALUATION AT max_steps = 1500")
    print("#"*70)
    for name, dtype, path in drivers:
        t0 = time.time()
        res = evaluate_driver(name, dtype, path, max_steps=1500)
        all_results[(name, 1500)] = res
        print_summary(name, 1500, res)
        print(f"  (Took {time.time()-t0:.1f}s)")
    
    # Save results to a json/csv for full recording
    import json
    serializable = {}
    for (name, ms), res in all_results.items():
        serializable[f"{name}_{ms}"] = res
    with open("results/dev_diagnostics.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print("\nSaved full diagnostics to results/dev_diagnostics.json")


if __name__ == "__main__":
    main()
