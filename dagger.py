"""
dagger.py <seed>
DAgger (Dataset Aggregation) loop:
  Round 0: start from the aug model
  Rounds 1-6: collect more data with learner/expert mixing, retrain
"""
import os, sys, json, time
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import gymnasium as gym
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from expert import (expert_steer, speed_controller, get_car_state,
                    wheels_on_track)
from train import train, set_seeds, load_data, SteeringDataset, VAL_SEEDS
from model import build_model
from evaluate import evaluate

# ── Configuration ────────────────────────────────────────────────────
SKIP_FRAMES = 50
MAX_STEPS = 1500
MAX_OFFROAD_STREAK = 150
TRACKS_PER_ROUND = 15
MAX_ROUNDS = 6
LOOKAHEAD = 5
GAIN = 2.8
TARGET_SPEED_STRAIGHT = 70.0
TARGET_SPEED_TURN = 38.0
MIN_TARGET_SPEED = 26.0


def collect_dagger_round(model, round_num, seed_base, beta):
    """
    Collect data for one DAgger round.
    Each 30-frame block is driven by learner with prob (1-beta) or expert with prob beta.
    Labels are always the expert's clean steering.
    """
    env = gym.make("CarRacing-v3", render_mode="rgb_array",
                   lap_complete_percent=0.95, max_episode_steps=MAX_STEPS)
    
    all_obs = []
    all_steers = []
    
    seeds = [seed_base + i for i in range(TRACKS_PER_ROUND)]
    
    for track_seed in seeds:
        obs, _ = env.reset(seed=track_seed)
        
        offroad_streak = 0
        use_expert_block = True
        block_counter = 0
        
        for step in range(MAX_STEPS):
            if step < SKIP_FRAMES:
                steer = 0.0
                gas, brake = speed_controller(env, steer, TARGET_SPEED_STRAIGHT,
                                            TARGET_SPEED_TURN, MIN_TARGET_SPEED)
                action = np.array([steer, gas, brake], dtype=np.float32)
                obs, _, terminated, truncated, _ = env.step(action)
                continue
            
            # Decide block driver every 30 frames
            if block_counter % 30 == 0:
                use_expert_block = np.random.random() < beta
            block_counter += 1
            
            # Expert's clean steering (always the label)
            expert_s = expert_steer(env, lookahead=LOOKAHEAD, gain=GAIN)
            
            # Decide who drives
            if use_expert_block:
                exec_steer = expert_s
            else:
                # Model prediction
                cropped = obs[:84, :, :].astype(np.float32)[np.newaxis]
                pred = model.predict(cropped, verbose=0)[0, 0]
                exec_steer = float(np.clip(pred, -1.0, 1.0))
            
            # Record frame
            cropped = obs[:84, :, :]
            all_obs.append(cropped)
            all_steers.append(expert_s)
            
            # Execute
            gas, brake = speed_controller(env, exec_steer, TARGET_SPEED_STRAIGHT,
                                        TARGET_SPEED_TURN, MIN_TARGET_SPEED)
            action = np.array([exec_steer, gas, brake], dtype=np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            
            # Track off-road for early termination
            n_wheels = wheels_on_track(env)
            if n_wheels == 0:
                offroad_streak += 1
            else:
                offroad_streak = 0
            
            if offroad_streak >= MAX_OFFROAD_STREAK:
                break
            
            if terminated or truncated:
                break
        
        print(f"    Round {round_num}, Seed {track_seed}: "
              f"{'expert' if use_expert_block else 'learner'} driving, "
              f"{sum(1 for s in all_steers[-MAX_STEPS:])} frames this episode")
    
    env.close()
    
    return np.array(all_obs, dtype=np.uint8), np.array(all_steers, dtype=np.float32)


def run_dagger(seed):
    """Run the full DAgger loop."""
    set_seeds(seed)
    
    print(f"\n{'='*60}")
    print(f"DAgger loop, seed={seed}")
    print(f"{'='*60}")
    
    # Round 0: start from aug model
    aug_model_path = f"models/aug_s{seed}.keras"
    if not os.path.exists(aug_model_path):
        print(f"Training initial aug model...")
        from train import train as train_model
        train_model("aug", seed)
    
    model = keras.models.load_model(aug_model_path)
    print(f"Loaded initial model from {aug_model_path}")
    
    # Evaluate round 0 on dev
    print(f"\n--- Round 0: Evaluating initial aug model ---")
    _, laps_0, autonomy_0, _ = evaluate(f"aug_s{seed}", "dev", record_episodes=0)
    
    curve_data = [{
        "round": 0,
        "laps_completed": laps_0,
        "mean_autonomy": autonomy_0,
    }]
    
    # Aggregate extra data across rounds
    all_extra_obs = []
    all_extra_steers = []
    
    consecutive_good = 0
    
    for round_num in range(1, MAX_ROUNDS + 1):
        beta = 0.5 / (2 ** (round_num - 1))  # 0.5, 0.25, 0.125, ...
        seed_base = 300 + 15 * round_num
        
        print(f"\n--- DAgger Round {round_num} (beta={beta:.3f}) ---")
        
        # Collect new data
        new_obs, new_steers = collect_dagger_round(model, round_num, seed_base, beta)
        print(f"  Collected {len(new_obs)} new frames")
        
        all_extra_obs.append(new_obs)
        all_extra_steers.append(new_steers)
        
        # Aggregate
        extra_obs = np.concatenate(all_extra_obs, axis=0)
        extra_steers = np.concatenate(all_extra_steers, axis=0)
        print(f"  Total extra data: {len(extra_obs)} frames")
        
        # Retrain from scratch with aggregated data
        set_seeds(seed + round_num * 1000)
        
        # Load base data
        base_data = np.load("data/noisy.npz")
        base_seeds = base_data["seeds"]
        train_mask = ~np.isin(base_seeds, list(VAL_SEEDS))
        
        X_train = base_data["observations"][train_mask]
        y_train = base_data["steers"][train_mask]
        X_val = base_data["observations"][~train_mask]
        y_val = base_data["steers"][~train_mask]
        
        # Add DAgger data to training
        X_train = np.concatenate([X_train, extra_obs], axis=0)
        y_train = np.concatenate([y_train, extra_steers], axis=0)
        
        # Apply balancing
        straight_mask = np.abs(y_train) < 0.05
        keep_mask = ~straight_mask | (np.random.random(len(y_train)) < 0.3)
        X_train = X_train[keep_mask]
        y_train = y_train[keep_mask]
        
        print(f"  Training data after balancing: {len(X_train)} frames")
        
        # Build and train
        model = build_model()
        
        train_ds = SteeringDataset(X_train, y_train, batch_size=64, augment=True)
        val_ds = SteeringDataset(X_val, y_val, batch_size=64, augment=False)
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=1
            ),
        ]
        
        history = model.fit(
            train_ds, validation_data=val_ds,
            epochs=30, callbacks=callbacks, verbose=1
        )
        
        # Save intermediate model
        round_model_path = f"models/dagger_s{seed}_r{round_num}.keras"
        model.save(round_model_path)
        
        # Evaluate on dev
        # Save model temporarily for evaluation
        temp_path = f"models/dagger_s{seed}.keras"
        model.save(temp_path)
        
        print(f"\n  Evaluating round {round_num}...")
        _, laps_r, autonomy_r, _ = evaluate(f"dagger_s{seed}", "dev", record_episodes=0)
        
        curve_data.append({
            "round": round_num,
            "laps_completed": laps_r,
            "mean_autonomy": autonomy_r,
        })
        
        # Check early stopping
        if laps_r >= 19:
            consecutive_good += 1
        else:
            consecutive_good = 0
        
        if consecutive_good >= 2:
            print(f"\n  Early stop: {laps_r}/20 laps in 2 consecutive rounds!")
            break
    
    # Save final model
    final_path = f"models/dagger_s{seed}.keras"
    model.save(final_path)
    print(f"\nSaved final DAgger model to {final_path}")
    
    # Save curve data
    curve_csv = f"results/dagger_curve_s{seed}.csv"
    with open(curve_csv, "w", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=["round", "laps_completed", "mean_autonomy"])
        writer.writeheader()
        writer.writerows(curve_data)
    
    # Plot curve
    rounds = [d["round"] for d in curve_data]
    laps = [d["laps_completed"] for d in curve_data]
    autonomies = [d["mean_autonomy"] for d in curve_data]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(rounds, laps, 'b-o', label="Laps completed")
    ax1.set_xlabel("DAgger Round")
    ax1.set_ylabel("Laps Completed (/20)", color='b')
    ax1.set_ylim(0, 21)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(rounds, autonomies, 'r-s', label="Mean Autonomy")
    ax2.set_ylabel("Mean Autonomy (%)", color='r')
    ax2.set_ylim(0, 105)
    
    plt.title(f"DAgger Progress (seed {seed})")
    fig.tight_layout()
    plt.savefig(f"results/dagger_curve_s{seed}.png", dpi=100)
    plt.close()
    print(f"Saved results/dagger_curve_s{seed}.png")
    
    return model, curve_data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dagger.py <seed>")
        sys.exit(1)
    
    seed = int(sys.argv[1])
    run_dagger(seed)
