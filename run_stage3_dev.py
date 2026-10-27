"""
run_stage3_dev.py - Stage 3 Dev Comparison under Unified Protocol
Evaluates expert, zero, plain_s0-2, aug_s0-2, dagger_s0-2 on DEV tracks 1000-1019.
Saves results/dev_final.csv and records 1 video clip per mode into videos/dev/<mode>/.
"""
import os, time, csv
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
import imageio

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


def save_video(frames, path, fps=50):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec='libx264',
                                output_params=['-pix_fmt', 'yuv420p'])
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"  [Video] Saved {path} ({len(frames)} frames)")


def main():
    t_start = time.time()
    csv_path = "results/dev_final.csv"
    fieldnames = [
        "driver", "mode", "model_seed", "track_seed",
        "lap_completed", "steps", "track_progress",
        "offroad_events", "longest_offroad_streak",
        "mean_speed", "mean_abs_steer", "steer_jitter",
        "mean_centre_distance", "max_centre_distance",
        "ending_type", "autonomy"
    ]

    all_records = []
    recorded_modes = set()

    print("=" * 70)
    print("STAGE 3: DEV COMPARISON (DEV TRACKS 1000-1019)")
    print(f"Unified 2000-step limit, {len(DRIVERS)} drivers, {len(DEV_SEEDS)} tracks each")
    print("=" * 70)

    for driver_name, mode, seed_num in DRIVERS:
        print(f"\nEvaluating {driver_name} (mode={mode}, seed={seed_num})...")
        t0 = time.time()

        model = None
        driver_type = mode
        if mode in ["plain", "aug", "dagger"]:
            driver_type = "model"
            model_path = f"models/{driver_name}.keras"
            model = keras.models.load_model(model_path)

        for track_seed in DEV_SEEDS:
            # Record 1 clip per mode (on track 1000)
            record_video = (mode not in recorded_modes and track_seed == 1000)
            if record_video:
                recorded_modes.add(mode)

            res = run_episode(
                driver_type=driver_type,
                model=model,
                seed=track_seed,
                speed_factor=1.0,
                disturb=False,
                record_frames=record_video,
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
            }
            all_records.append(rec)

            if record_video and res["frames"]:
                video_file = f"videos/dev/{mode}/{driver_name}_track{track_seed}.mp4"
                save_video(res["frames"], video_file)

        elapsed = time.time() - t0
        driver_recs = [r for r in all_records if r["driver"] == driver_name]
        laps = sum(1 for r in driver_recs if r["lap_completed"])
        mean_prog = np.mean([r["track_progress"] for r in driver_recs])
        mean_off = np.mean([r["offroad_events"] for r in driver_recs])
        mean_cd = np.mean([r["mean_centre_distance"] for r in driver_recs])
        print(f"  {driver_name}: {laps}/20 laps | progress={mean_prog:.1f}% | offroad={mean_off:.1f} | centre_dist={mean_cd:.2f} | took {elapsed:.1f}s")

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nSaved {len(all_records)} episode rows to {csv_path}")

    # Compute and print aggregate table across seeds per mode
    print("\n" + "=" * 90)
    print("STAGE 3 SUMMARY TABLE: DEV TRACKS (MEAN +/- STD ACROSS SEEDS PER MODE)")
    print("=" * 90)
    print(f"{'Mode':10s} | {'Laps / 20':15s} | {'Progress %':12s} | {'Steps to Lap':14s} | {'Off-road Events':15s} | {'Centre Dist':12s} | {'Jitter':10s}")
    print("-" * 90)

    # For reference modes (expert, zero)
    for ref_mode in ["expert", "zero"]:
        recs = [r for r in all_records if r["mode"] == ref_mode]
        laps = sum(1 for r in recs if r["lap_completed"])
        prog = np.mean([r["track_progress"] for r in recs])
        lap_steps = [r["steps"] for r in recs if r["lap_completed"]]
        steps_str = f"{np.mean(lap_steps):.0f}" if lap_steps else "N/A"
        offroad = np.mean([r["offroad_events"] for r in recs])
        cd = np.mean([r["mean_centre_distance"] for r in recs])
        jit = np.mean([r["steer_jitter"] for r in recs])
        print(f"{ref_mode:10s} | {laps:2d}/20           | {prog:5.1f}%       | {steps_str:14s} | {offroad:15.2f} | {cd:10.2f}   | {jit:8.4f}")

    # For learned modes (plain, aug, dagger)
    for l_mode in ["plain", "aug", "dagger"]:
        seed_laps = []
        seed_progs = []
        seed_steps = []
        seed_offroads = []
        seed_cds = []
        seed_jits = []

        for s in [0, 1, 2]:
            s_recs = [r for r in all_records if r["mode"] == l_mode and r["model_seed"] == s]
            laps = sum(1 for r in s_recs if r["lap_completed"])
            seed_laps.append(laps)
            seed_progs.append(np.mean([r["track_progress"] for r in s_recs]))
            l_steps = [r["steps"] for r in s_recs if r["lap_completed"]]
            if l_steps:
                seed_steps.append(np.mean(l_steps))
            seed_offroads.append(np.mean([r["offroad_events"] for r in s_recs]))
            seed_cds.append(np.mean([r["mean_centre_distance"] for r in s_recs]))
            seed_jits.append(np.mean([r["steer_jitter"] for r in s_recs]))

        laps_str = f"{np.mean(seed_laps):.1f} +/- {np.std(seed_laps):.1f}"
        prog_str = f"{np.mean(seed_progs):.1f} +/- {np.std(seed_progs):.1f}%"
        steps_str = f"{np.mean(seed_steps):.0f} +/- {np.std(seed_steps):.0f}" if seed_steps else "N/A"
        off_str = f"{np.mean(seed_offroads):.2f} +/- {np.std(seed_offroads):.2f}"
        cd_str = f"{np.mean(seed_cds):.2f} +/- {np.std(seed_cds):.2f}"
        jit_str = f"{np.mean(seed_jits):.4f} +/- {np.std(seed_jits):.4f}"

        print(f"{l_mode:10s} | {laps_str:15s} | {prog_str:12s} | {steps_str:14s} | {off_str:15s} | {cd_str:12s} | {jit_str:10s}")

    print("=" * 90)
    total_time = (time.time() - t_start) / 60
    print(f"Stage 3 completed in {total_time:.1f} minutes.")


if __name__ == "__main__":
    main()
