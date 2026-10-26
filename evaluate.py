"""
evaluate.py <name> [dev|test]
Evaluate a trained model (or reference driver) on unseen tracks.
  name: model name (e.g., "aug_s0") or "expert" or "zero"
  dev:  seeds 1000-1019 (used inside DAgger loop)
  test: seeds 2000-2019 (default, final numbers only)
"""
import os, sys, math, time, csv
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import gymnasium as gym
import keras

from expert import (expert_steer, speed_controller, get_car_state,
                    wheels_on_track, LOOKAHEAD, GAIN,
                    TARGET_SPEED_STRAIGHT, TARGET_SPEED_TURN, MIN_TARGET_SPEED)

# ── Configuration ────────────────────────────────────────────────────
SKIP_FRAMES = 50
MAX_STEPS = 2000
MAX_OFFROAD_STREAK = 150  # End episode if off-track this long


def evaluate(name, split="test", record_episodes=3):
    """
    Evaluate a driver on 20 tracks.
    name: "expert", "zero", or a model name like "plain_s0"
    """
    if split == "test" and not os.environ.get("ALLOW_TEST_EVAL"):
        print(f"SKIPPING: Test split evaluation for {name} is blocked until Stage 6.")
        return [], 0, 100.0, 0.0
    
    if split == "dev":
        seeds = list(range(1000, 1020))
    else:
        seeds = list(range(2000, 2020))
    
    # Load model if needed
    model = None
    mode = None
    seed_used = None
    
    if name == "expert":
        mode = "expert"
    elif name == "zero":
        mode = "zero"
    else:
        model_path = f"models/{name}.keras"
        if not os.path.exists(model_path):
            print(f"ERROR: Model {model_path} not found!")
            sys.exit(1)
        model = keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
        # Parse mode and seed from name
        parts = name.split("_s")
        mode = parts[0]
        seed_used = int(parts[1]) if len(parts) > 1 else 0
    
    # Setup recording
    record_dir = f"videos/{name}"
    os.makedirs(record_dir, exist_ok=True)
    
    results = []
    
    for i, track_seed in enumerate(seeds):
        env = gym.make("CarRacing-v3", render_mode="rgb_array",
                       lap_complete_percent=0.95, max_episode_steps=MAX_STEPS)
        
        # Record first few episodes
        do_record = i < record_episodes
        frames_for_video = [] if do_record else None
        
        obs, _ = env.reset(seed=track_seed)
        
        total_reward = 0.0
        steps = 0
        offroad_events = 0
        was_on_track = True
        offroad_streak = 0
        max_offroad = 0
        lap_completed = False
        lost = False
        
        for step in range(MAX_STEPS):
            if step < SKIP_FRAMES:
                steer = 0.0
            elif name == "expert":
                steer = expert_steer(env, lookahead=LOOKAHEAD, gain=GAIN)
            elif name == "zero":
                steer = 0.0
            else:
                # Model prediction
                cropped = obs[:84, :, :].astype(np.float32)[np.newaxis]
                pred = model.predict(cropped, verbose=0)[0, 0]
                steer = float(np.clip(pred, -1.0, 1.0))
            
            gas, brake = speed_controller(env, steer,
                                        TARGET_SPEED_STRAIGHT, TARGET_SPEED_TURN,
                                        MIN_TARGET_SPEED)
            action = np.array([steer, gas, brake], dtype=np.float32)
            
            if do_record:
                # Get the larger render for video
                rendered = env.render()
                if rendered is not None:
                    frames_for_video.append(rendered)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
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
            
            # Lost detection: all wheels off for too long
            if offroad_streak >= MAX_OFFROAD_STREAK:
                lost = True
                break
            
            if terminated or truncated:
                uw = env.unwrapped if hasattr(env, 'unwrapped') else env
                total_tiles = len(uw.track) if hasattr(uw, 'track') else 1
                tiles_visited = uw.tile_visited_count if hasattr(uw, 'tile_visited_count') else 0
                lap_completed = info.get("lap_finished", False) or (hasattr(uw, 'new_lap') and uw.new_lap and tiles_visited / total_tiles >= 0.95)
                break
        
        uw = env.unwrapped if hasattr(env, 'unwrapped') else env
        total_tiles = len(uw.track) if hasattr(uw, 'track') else 1
        tiles_visited = uw.tile_visited_count if hasattr(uw, 'tile_visited_count') else 0
        progress = (tiles_visited / total_tiles) * 100.0
        
        # Autonomy metric
        lap_time_seconds = steps / 50.0
        takeovers = offroad_events
        autonomy = max(0.0, min(100.0, (1 - takeovers * 6 / max(lap_time_seconds, 1)) * 100))
        
        result = {
            "seed": track_seed,
            "lap_completed": lap_completed,
            "score": total_reward,
            "steps": steps,
            "track_progress": progress,
            "tiles_visited": tiles_visited,
            "total_tiles": total_tiles,
            "offroad_events": offroad_events,
            "max_offroad_streak": max_offroad,
            "autonomy": autonomy,
            "lost": lost,
        }
        results.append(result)
        
        status = "LAP" if lap_completed else ("LOST" if lost else "DONE")
        print(f"  Seed {track_seed}: {status:4s}  score={total_reward:7.1f}  "
              f"progress={progress:5.1f}%  steps={steps:4d}  offroad={offroad_events}  autonomy={autonomy:.1f}%")
        
        # Save video
        if do_record and frames_for_video:
            try:
                import imageio
                video_path = os.path.join(record_dir, f"episode_{track_seed}.mp4")
                writer = imageio.get_writer(video_path, fps=50, codec='libx264',
                                          output_params=['-pix_fmt', 'yuv420p'])
                for frame in frames_for_video:
                    writer.append_data(frame)
                writer.close()
                print(f"    Saved video: {video_path}")
            except Exception as e:
                print(f"    Video save error: {e}")
        
        env.close()
    
    # Summary
    laps = sum(1 for r in results if r["lap_completed"])
    mean_score = np.mean([r["score"] for r in results])
    mean_autonomy = np.mean([r["autonomy"] for r in results])
    mean_offroad = np.mean([r["offroad_events"] for r in results])
    mean_progress = np.mean([r["track_progress"] for r in results])
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {name} on {split} tracks")
    print(f"  Laps completed: {laps}/{len(seeds)}")
    print(f"  Mean track progress: {mean_progress:.1f}%")
    print(f"  Mean score: {mean_score:.1f}")
    print(f"  Mean autonomy: {mean_autonomy:.1f}%")
    print(f"  Mean off-road events: {mean_offroad:.1f}")
    print(f"{'='*60}")
    
    # Append to results CSV
    csv_path = "results/results.csv"
    write_header = not os.path.exists(csv_path)
    
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["name", "mode", "seed", "split", "laps_completed",
                           "n_episodes", "mean_autonomy", "mean_score",
                           "mean_offroad_events", "mean_track_progress"])
        writer.writerow([name, mode, seed_used if seed_used is not None else "",
                        split, laps, len(seeds), f"{mean_autonomy:.1f}",
                        f"{mean_score:.1f}", f"{mean_offroad:.1f}", f"{mean_progress:.1f}"])
    
    return results, laps, mean_autonomy, mean_score


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <name> [dev|test]")
        sys.exit(1)
    
    name = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
    
    evaluate(name, split)
