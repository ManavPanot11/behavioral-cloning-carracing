"""
run_stage4_disturbance.py - Stage 4 Disturbance (Recovery) Test
Pre-registered protocol:
- Dev tracks 1000-1019
- Drivers: expert, zero, plain_s0-2, aug_s0-2, dagger_s0-2 (11 drivers, 20 tracks each = 220 episodes)
- 3 disturbance triggers per track at waypoint fractions [0.25, 0.50, 0.75]
- Steering offset magnitude 0.6 for 20 frames, sign chosen via seeded RNG per (track, trigger)
- Metrics: recovered_each_disturbance, offroad_events, lap_completed, max_centre_distance, etc.
- Saves results/dev_disturbed.csv
"""
import os, time, csv
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras

from eval_suite import run_episode, DEV_SEEDS

DRIVERS = [
    ("expert", "expert", None),
    ("zero", "zero", None),
    ("plain_s0", "plain", 0),
    ("plain_s1", "plain", 1),
    ("plain_s2", "plain", 2),
    ("aug_s0", "aug", 0),
    ("aug_s1", "aug", 1),
    ("aug_s2", "aug", 2),
    ("dagger_s0", "dagger", 0),
    ("dagger_s1", "dagger", 1),
    ("dagger_s2", "dagger", 2),
]


def main():
    t_start = time.time()
    csv_path = "results/dev_disturbed.csv"
    fieldnames = [
        "driver", "mode", "model_seed", "track_seed",
        "lap_completed", "steps", "track_progress",
        "offroad_events", "longest_offroad_streak",
        "mean_speed", "mean_abs_steer", "steer_jitter",
        "mean_centre_distance", "max_centre_distance",
        "ending_type", "autonomy", "recovered_each_disturbance"
    ]

    all_records = []

    print("=" * 70)
    print("STAGE 4: DISTURBANCE (RECOVERY) TEST (DEV TRACKS 1000-1019)")
    print(f"Unified 2000-step limit, {len(DRIVERS)} drivers, {len(DEV_SEEDS)} tracks each")
    print("Disturbance: magnitude 0.6 for 20 frames at fractions [0.25, 0.50, 0.75]")
    print("Recovery threshold: within 3 world units within 60 frames after disturbance")
    print("=" * 70)

    for driver_name, mode, seed_num in DRIVERS:
        print(f"\nEvaluating disturbed {driver_name} (mode={mode}, seed={seed_num})...")
        t0 = time.time()

        model = None
        driver_type = mode
        if mode in ["plain", "aug", "dagger"]:
            driver_type = "model"
            model_path = f"models/{driver_name}.keras"
            model = keras.models.load_model(model_path)

        for track_seed in DEV_SEEDS:
            res = run_episode(
                driver_type=driver_type,
                model=model,
                seed=track_seed,
                speed_factor=1.0,
                disturb=True,
                record_frames=False,
                max_steps=2000
            )

            rec = {
                "driver": driver_name,
                "mode": mode,
                "model_seed": seed_num if seed_num is not None else "",
                "track_seed": track_seed,
                "lap_completed": res["lap_completed"],
                "steps": res["steps"],
                "track_progress": round(res["track_progress"], 2),
                "offroad_events": res["offroad_events"],
                "longest_offroad_streak": res["longest_offroad_streak"],
                "mean_speed": round(res["mean_speed"], 2),
                "mean_abs_steer": round(res["mean_abs_steer"], 4),
                "steer_jitter": round(res["steer_jitter"], 4),
                "mean_centre_distance": round(res["mean_centre_distance"], 2),
                "max_centre_distance": round(res["max_centre_distance"], 2),
                "ending_type": res["ending_type"],
                "autonomy": round(res["autonomy"], 2),
                "recovered_each_disturbance": res["recovered_each_disturbance"],
            }
            all_records.append(rec)

        elapsed = time.time() - t0
        driver_recs = [r for r in all_records if r["driver"] == driver_name]
        laps = sum(1 for r in driver_recs if r["lap_completed"])
        recov = sum(1 for r in driver_recs if r["recovered_each_disturbance"])
        mean_prog = np.mean([r["track_progress"] for r in driver_recs])
        mean_off = np.mean([r["offroad_events"] for r in driver_recs])
        mean_cd = np.mean([r["mean_centre_distance"] for r in driver_recs])
        max_cd = np.max([r["max_centre_distance"] for r in driver_recs])
        print(f"  {driver_name}: {laps}/20 laps | recovered_all={recov}/20 | progress={mean_prog:.1f}% | offroad={mean_off:.1f} | mean_dist={mean_cd:.2f} | max_dist={max_cd:.2f} | took {elapsed:.1f}s")

    # Save CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n[Stage 4] Saved {len(all_records)} disturbed dev episode records to {csv_path}")

    # Aggregated Mode Summary Table (mean +/- std across seeds)
    print("\n" + "=" * 90)
    print("STAGE 4: DEV DISTURBANCE COMPARISON SUMMARY")
    print(f"{'Mode':<10} | {'Laps/20':<15} | {'Recovered All':<15} | {'Progress %':<15} | {'Offroad Events':<15} | {'Max Centre Dist':<15}")
    print("-" * 90)

    # Reference rows: expert and zero
    for ref_name in ["expert", "zero"]:
        r_list = [r for r in all_records if r["driver"] == ref_name]
        laps = sum(1 for r in r_list if r["lap_completed"])
        recov = sum(1 for r in r_list if r["recovered_each_disturbance"])
        prog = np.mean([r["track_progress"] for r in r_list])
        off = np.mean([r["offroad_events"] for r in r_list])
        max_dist = np.mean([r["max_centre_distance"] for r in r_list])
        print(f"{ref_name:<10} | {laps:<15} | {recov:<15} | {prog:<15.1f} | {off:<15.2f} | {max_dist:<15.2f}")

    # Learned modes: plain, aug, dagger
    for mode in ["plain", "aug", "dagger"]:
        mode_laps = []
        mode_recov = []
        mode_prog = []
        mode_off = []
        mode_max_dist = []
        for s in [0, 1, 2]:
            s_recs = [r for r in all_records if r["mode"] == mode and r["model_seed"] == s]
            mode_laps.append(sum(1 for r in s_recs if r["lap_completed"]))
            mode_recov.append(sum(1 for r in s_recs if r["recovered_each_disturbance"]))
            mode_prog.append(np.mean([r["track_progress"] for r in s_recs]))
            mode_off.append(np.mean([r["offroad_events"] for r in s_recs]))
            mode_max_dist.append(np.mean([r["max_centre_distance"] for r in s_recs]))

        lap_str = f"{np.mean(mode_laps):.1f} +/- {np.std(mode_laps):.1f}"
        recov_str = f"{np.mean(mode_recov):.1f} +/- {np.std(mode_recov):.1f}"
        prog_str = f"{np.mean(mode_prog):.1f}% +/- {np.std(mode_prog):.1f}%"
        off_str = f"{np.mean(mode_off):.2f} +/- {np.std(mode_off):.2f}"
        dist_str = f"{np.mean(mode_max_dist):.2f} +/- {np.std(mode_max_dist):.2f}"
        print(f"{mode:<10} | {lap_str:<15} | {recov_str:<15} | {prog_str:<15} | {off_str:<15} | {dist_str:<15}")

    print("=" * 90)
    print(f"Total time elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
