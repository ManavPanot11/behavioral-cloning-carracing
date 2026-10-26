"""
collect.py - Data collection for behavioral cloning
Collects clean and noisy (with recovery perturbations) datasets.
"""
import os, sys, time, math
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from expert import expert_steer, speed_controller, wheels_on_track

# ── Configuration ────────────────────────────────────────────────────
TRAIN_SEEDS = list(range(0, 40))
VAL_SEEDS = list(range(100, 108))
SKIP_FRAMES = 50
MAX_STEPS = 1500

# Expert parameters (tuned for optimal track following)
LOOKAHEAD = 5
GAIN = 2.8
TARGET_SPEED_STRAIGHT = 70.0
TARGET_SPEED_TURN = 38.0
MIN_TARGET_SPEED = 26.0


def collect_dataset(seeds, noisy=False, desc="clean"):
    """
    Collect demonstrations from expert on given seeds.
    If noisy=True, add steering perturbations for recovery data.
    """
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95)
    
    all_obs = []
    all_steers = []
    all_seeds_arr = []
    all_frames = []
    
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        
        # Noisy perturbation state
        perturb_active = False
        perturb_steer = 0.0
        perturb_remaining = 0
        perturb_cooldown = 0
        
        for step in range(MAX_STEPS):
            if step < SKIP_FRAMES:
                # Intro: gentle gas, no recording
                steer = 0.0
                gas, brake = speed_controller(env, steer, TARGET_SPEED_STRAIGHT,
                                            TARGET_SPEED_TURN, MIN_TARGET_SPEED)
                action = np.array([steer, gas, brake], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                continue
            
            # Expert's clean steering (the label)
            expert_s = expert_steer(env, lookahead=LOOKAHEAD, gain=GAIN)
            
            if noisy:
                # Manage perturbation
                if perturb_active:
                    perturb_remaining -= 1
                    if perturb_remaining <= 0:
                        perturb_active = False
                        perturb_cooldown = np.random.randint(30, 80)
                elif perturb_cooldown > 0:
                    perturb_cooldown -= 1
                else:
                    # Randomly start a perturbation
                    if np.random.random() < 0.05:  # ~5% chance each frame
                        perturb_active = True
                        sign = np.random.choice([-1, 1])
                        magnitude = np.random.uniform(0.3, 0.8)
                        perturb_steer = sign * magnitude
                        perturb_remaining = np.random.randint(10, 26)
                
                # Execute: expert steer + perturbation
                exec_steer = float(np.clip(expert_s + (perturb_steer if perturb_active else 0.0), -1.0, 1.0))
            else:
                exec_steer = expert_s
            
            # Record: cropped observation (top 84 rows) + expert's CLEAN steering
            cropped = obs[:84, :, :]  # 84x96x3
            all_obs.append(cropped)
            all_steers.append(expert_s)
            all_seeds_arr.append(seed)
            all_frames.append(step)
            
            # Execute action
            gas, brake = speed_controller(env, exec_steer, TARGET_SPEED_STRAIGHT,
                                        TARGET_SPEED_TURN, MIN_TARGET_SPEED)
            action = np.array([exec_steer, gas, brake], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        lap = info.get("lap_finished", False) if (terminated or truncated) else False
        n_frames = sum(1 for s in all_seeds_arr if s == seed)
        print(f"  Seed {seed:4d}: {'LAP' if lap else 'DONE':4s}  {n_frames:4d} frames  "
              f"({'noisy' if noisy else 'clean'})")
    
    env.close()
    
    obs_arr = np.array(all_obs, dtype=np.uint8)
    steer_arr = np.array(all_steers, dtype=np.float32)
    seed_arr = np.array(all_seeds_arr, dtype=np.int32)
    frame_arr = np.array(all_frames, dtype=np.int32)
    
    return obs_arr, steer_arr, seed_arr, frame_arr


def main():
    np.random.seed(42)
    
    # Check if data already exists
    if os.path.exists("data/clean.npz") and os.path.exists("data/noisy.npz"):
        print("Data already exists, skipping collection.")
        print("Loading for statistics...")
        clean = np.load("data/clean.npz")
        noisy = np.load("data/noisy.npz")
        print_stats(clean["steers"], "clean")
        print_stats(noisy["steers"], "noisy")
        return
    
    all_seeds = TRAIN_SEEDS + VAL_SEEDS
    
    # ── Clean dataset ────────────────────────────────────────────
    print("=" * 60)
    print("Collecting CLEAN dataset (training + validation)")
    print("=" * 60)
    t0 = time.time()
    obs_c, steer_c, seed_c, frame_c = collect_dataset(all_seeds, noisy=False, desc="clean")
    print(f"\nClean dataset: {len(obs_c)} frames, {obs_c.nbytes/1e6:.1f} MB")
    print(f"Time: {time.time()-t0:.1f}s")
    
    if not os.path.exists("data/clean.npz"):
        np.savez_compressed("data/clean.npz",
                           observations=obs_c, steers=steer_c,
                           seeds=seed_c, frames=frame_c)
        print(f"Saved data/clean.npz")
    
    # ── Noisy dataset ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Collecting NOISY dataset (with recovery perturbations)")
    print("=" * 60)
    t0 = time.time()
    obs_n, steer_n, seed_n, frame_n = collect_dataset(all_seeds, noisy=True, desc="noisy")
    print(f"\nNoisy dataset: {len(obs_n)} frames, {obs_n.nbytes/1e6:.1f} MB")
    print(f"Time: {time.time()-t0:.1f}s")
    
    if not os.path.exists("data/noisy.npz"):
        np.savez_compressed("data/noisy.npz",
                           observations=obs_n, steers=steer_n,
                           seeds=seed_n, frames=frame_n)
        print(f"Saved data/noisy.npz")
    
    # ── Label statistics ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Label Statistics")
    print("=" * 60)
    print_stats(steer_c, "clean")
    print_stats(steer_n, "noisy")
    
    # ── Histogram ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(steer_c, bins=100, range=(-1, 1), alpha=0.7, color='blue')
    axes[0].set_title("Clean steering labels")
    axes[0].set_xlabel("Steering")
    axes[0].set_ylabel("Count")
    
    axes[1].hist(steer_n, bins=100, range=(-1, 1), alpha=0.7, color='orange')
    axes[1].set_title("Noisy steering labels")
    axes[1].set_xlabel("Steering")
    axes[1].set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig("results/steer_hist.png", dpi=100)
    print("Saved results/steer_hist.png")


def print_stats(steers, name):
    straight_frac = (np.abs(steers) < 0.05).mean()
    print(f"  {name}: {len(steers)} frames, "
          f"|steer|<0.05 = {straight_frac:.1%}, "
          f"mean={steers.mean():.4f}, std={steers.std():.4f}, "
          f"min={steers.min():.3f}, max={steers.max():.3f}")


if __name__ == "__main__":
    main()
