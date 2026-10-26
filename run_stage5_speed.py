"""
run_stage5_speed.py - Stage 5 Speed Stress Test (Pre-registered, secondary)
Hypothesis (untested): the speed controller slows the car in curves, which may be hiding errors that compound.
Test by scaling ALL controller target speeds (straight, turn, minimum) by a factor:
  - 1.0 (baseline)
  - 1.25
  - 1.5
Evaluated on DEV tracks 1000-1019 for:
  - expert
  - plain_s0, plain_s1, plain_s2
  - aug_s0, aug_s1, aug_s2
  - dagger_s0, dagger_s1, dagger_s2
(Expert is run FIRST at each factor to verify feasibility.)
Saves results/dev_speed_stress.csv.
"""
import os, sys, time, csv
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras

from eval_suite import run_episode, DEV_SEEDS

FACTORS = [1.0, 1.25, 1.5]

DRIVERS = [
    ("expert", "expert", None),
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
    csv_path = "results/dev_speed_stress.csv"
    fieldnames = [
        "speed_factor", "driver", "mode", "model_seed", "track_seed",
        "lap_completed", "steps", "track_progress",
        "offroad_events", "longest_offroad_streak",
        "mean_speed", "mean_abs_steer", "steer_jitter",
        "mean_centre_distance", "max_centre_distance",
        "ending_type", "autonomy"
    ]

    all_records = []

    print("=" * 75)
    print("STAGE 5: SPEED STRESS TEST (DEV TRACKS 1000-1019)")
    print(f"Factors: {FACTORS} | Drivers: {len(DRIVERS)} | Tracks: {len(DEV_SEEDS)}")
    print("=" * 75)

    for factor in FACTORS:
        print(f"\n==================== SPEED FACTOR {factor:.2f} ====================")
        # For baseline factor 1.0, reuse dev_final.csv if present
        if factor == 1.0 and os.path.exists("results/dev_final.csv"):
            print("  [Stage 5] Ingesting measured factor 1.0 baseline from results/dev_final.csv...")
            with open("results/dev_final.csv", "r", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r["driver"] == "zero":
                        continue
                    rec = dict(r)
                    rec["speed_factor"] = 1.0
                    all_records.append(rec)
            for driver_name, mode, seed_num in DRIVERS:
                drecs = [r for r in all_records if r["speed_factor"] == 1.0 and r["driver"] == driver_name]
                laps = sum(1 for r in drecs if r["lap_completed"] in [True, "True", 1, "1"])
                mean_spd = np.mean([float(r["mean_speed"]) for r in drecs])
                mean_prog = np.mean([float(r["track_progress"]) for r in drecs])
                mean_off = np.mean([float(r["offroad_events"]) for r in drecs])
                mean_cd = np.mean([float(r["mean_centre_distance"]) for r in drecs])
                print(f"  [Factor 1.00] {driver_name:10s}: {laps}/20 laps | speed={mean_spd:.1f} | progress={mean_prog:.1f}% | offroad={mean_off:.1f} | centre_dist={mean_cd:.2f} | (from dev_final.csv)")
            continue

        for driver_name, mode, seed_num in DRIVERS:
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
                    speed_factor=factor,
                    disturb=False,
                    record_frames=False,
                    max_steps=2000
                )

                rec = {
                    "speed_factor": factor,
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
                }
                all_records.append(rec)

            elapsed = time.time() - t0
            driver_recs = [r for r in all_records if r["speed_factor"] == factor and r["driver"] == driver_name]
            laps = sum(1 for r in driver_recs if r["lap_completed"])
            mean_prog = np.mean([r["track_progress"] for r in driver_recs])
            mean_off = np.mean([r["offroad_events"] for r in driver_recs])
            mean_spd = np.mean([r["mean_speed"] for r in driver_recs])
            mean_cd = np.mean([r["mean_centre_distance"] for r in driver_recs])
            print(f"  [Factor {factor:.2f}] {driver_name:10s}: {laps}/20 laps | speed={mean_spd:.1f} | progress={mean_prog:.1f}% | offroad={mean_off:.1f} | centre_dist={mean_cd:.2f} | took {elapsed:.1f}s")

    # Save CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n[Stage 5] Saved {len(all_records)} speed stress episode records to {csv_path}")

    # Summary Table per factor
    print("\n" + "=" * 85)
    print("STAGE 5: SPEED STRESS TEST SUMMARY TABLE")
    print(f"{'Factor':<8} | {'Mode':<10} | {'Laps/20':<15} | {'Progress %':<15} | {'Offroad Events':<15} | {'Mean Dist':<12}")
    print("-" * 85)

    for factor in FACTORS:
        # Expert row
        exp_recs = [r for r in all_records if float(r["speed_factor"]) == factor and r["driver"] == "expert"]
        laps = sum(1 for r in exp_recs if r["lap_completed"] in [True, "True", 1, "1"])
        prog = np.mean([float(r["track_progress"]) for r in exp_recs])
        off = np.mean([float(r["offroad_events"]) for r in exp_recs])
        cd = np.mean([float(r["mean_centre_distance"]) for r in exp_recs])
        print(f"{factor:<8.2f} | {'expert':<10} | {laps:<15} | {prog:<15.1f} | {off:<15.2f} | {cd:<12.2f}")

        # Learned modes: plain, aug, dagger
        for mode in ["plain", "aug", "dagger"]:
            mode_laps = []
            mode_prog = []
            mode_off = []
            mode_cd = []
            for s in [0, 1, 2]:
                s_recs = [r for r in all_records if float(r["speed_factor"]) == factor and r["mode"] == mode and str(r["model_seed"]) == str(s)]
                mode_laps.append(sum(1 for r in s_recs if r["lap_completed"] in [True, "True", 1, "1"]))
                mode_prog.append(np.mean([float(r["track_progress"]) for r in s_recs]))
                mode_off.append(np.mean([float(r["offroad_events"]) for r in s_recs]))
                mode_cd.append(np.mean([float(r["mean_centre_distance"]) for r in s_recs]))

            lap_str = f"{np.mean(mode_laps):.1f} +/- {np.std(mode_laps):.1f}"
            prog_str = f"{np.mean(mode_prog):.1f}% +/- {np.std(mode_prog):.1f}%"
            off_str = f"{np.mean(mode_off):.2f} +/- {np.std(mode_off):.2f}"
            cd_str = f"{np.mean(mode_cd):.2f} +/- {np.std(mode_cd):.2f}"
            print(f"{'':<8} | {mode:<10} | {lap_str:<15} | {prog_str:<15} | {off_str:<15} | {cd_str:<12}")
        print("-" * 85)

    print(f"Total time elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
