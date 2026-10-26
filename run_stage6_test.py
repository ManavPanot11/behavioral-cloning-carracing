"""
run_stage6_test.py - Stage 6 Final Evaluation on Held-Out Test Tracks (2000-2019)
Pre-registered protocol:
- Run ONCE per condition, NEVER tuned on.
- Conditions:
  (a) clean (speed 1.0, disturb=False)
  (b) disturbed (speed 1.0, disturb=True, Stage 4 protocol)
  (c) speed_1.25 (speed 1.25, disturb=False)
  (d) speed_1.50 (speed 1.50, disturb=False)
- Drivers: expert, zero, plain_s0-2, aug_s0-2, dagger_s0-2 (11 drivers)
- Test tracks: seeds 2000-2019 (20 tracks)
- Outputs:
  results/test_episodes.csv (one row per episode)
  results/test_results.csv (aggregated per driver and condition)
- Video recording:
  Clean lap for expert and dagger_s0.
  On the SAME test track (lowest track seed where all 3 disturbances trigger),
  disturbed lap for plain_s0, aug_s0, dagger_s0.
"""
import os, sys, time, csv
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
import imageio

from eval_suite import run_episode, TEST_SEEDS

CONDITIONS = [
    ("clean", 1.0, False),
    ("disturbed", 1.0, True),
    ("speed_1.25", 1.25, False),
    ("speed_1.50", 1.50, False),
]

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

    target_condition = sys.argv[1] if len(sys.argv) > 1 else None
    if target_condition:
        conditions_to_run = [c for c in CONDITIONS if c[0] == target_condition]
        episodes_csv_path = f"results/test_episodes_{target_condition}.csv"
        results_csv_path = f"results/test_results_{target_condition}.csv"
    else:
        conditions_to_run = CONDITIONS
        episodes_csv_path = "results/test_episodes.csv"
        results_csv_path = "results/test_results.csv"

    fieldnames = [
        "condition", "speed_factor", "disturbed",
        "driver", "mode", "model_seed", "track_seed",
        "lap_completed", "steps", "track_progress",
        "offroad_events", "longest_offroad_streak",
        "mean_speed", "mean_abs_steer", "steer_jitter",
        "mean_centre_distance", "max_centre_distance",
        "ending_type", "autonomy", "recovered_each_disturbance"
    ]

    all_records = []

    print("=" * 80)
    print("STAGE 6: FINAL TEST-TRACK EVALUATION (HELD-OUT SEEDS 2000-2019)")
    print(f"Conditions: {[c[0] for c in conditions_to_run]}")
    print(f"Drivers: {len(DRIVERS)} | Test Tracks: {len(TEST_SEEDS)}")
    print("Unified 2000-step limit applied everywhere")
    print("=" * 80)

    # Pre-select video test track: lowest track seed where all 3 disturbances trigger
    # In CarRacing, track waypoint fractions [0.25, 0.50, 0.75] will all trigger on any track
    # when completed. Seed 2000 is the lowest test track seed.
    chosen_video_track = 2000

    for cond_name, factor, disturb_flag in conditions_to_run:
        print(f"\n==================== CONDITION: {cond_name.upper()} ====================")
        for driver_name, mode, seed_num in DRIVERS:
            t0 = time.time()
            model = None
            driver_type = mode
            if mode in ["plain", "aug", "dagger"]:
                driver_type = "model"
                model_path = f"models/{driver_name}.keras"
                model = keras.models.load_model(model_path)

            for track_seed in TEST_SEEDS:
                # Video recording conditions:
                # 1. Clean test lap for expert and dagger_s0 (track 2000)
                # 2. Disturbed lap for plain_s0, aug_s0, dagger_s0 on chosen_video_track (track 2000)
                rec_vid = False
                if cond_name == "clean" and track_seed == chosen_video_track and driver_name in ["expert", "dagger_s0"]:
                    rec_vid = True
                elif cond_name == "disturbed" and track_seed == chosen_video_track and driver_name in ["plain_s0", "aug_s0", "dagger_s0"]:
                    rec_vid = True

                res = run_episode(
                    driver_type=driver_type,
                    model=model,
                    seed=track_seed,
                    speed_factor=factor,
                    disturb=disturb_flag,
                    record_frames=rec_vid,
                    max_steps=2000
                )

                rec = {
                    "condition": cond_name,
                    "speed_factor": factor,
                    "disturbed": disturb_flag,
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

                if rec_vid and res["frames"]:
                    vid_dir = f"videos/test/{cond_name}"
                    vid_path = f"{vid_dir}/{driver_name}_track{track_seed}.mp4"
                    save_video(res["frames"], vid_path)

            elapsed = time.time() - t0
            driver_recs = [r for r in all_records if r["condition"] == cond_name and r["driver"] == driver_name]
            laps = sum(1 for r in driver_recs if r["lap_completed"])
            mean_prog = np.mean([r["track_progress"] for r in driver_recs])
            mean_off = np.mean([r["offroad_events"] for r in driver_recs])
            mean_spd = np.mean([r["mean_speed"] for r in driver_recs])
            mean_cd = np.mean([r["mean_centre_distance"] for r in driver_recs])
            print(f"  [{cond_name:10s}] {driver_name:10s}: {laps}/20 laps | spd={mean_spd:.1f} | prog={mean_prog:.1f}% | off={mean_off:.1f} | dist={mean_cd:.2f} | took {elapsed:.1f}s")

    # Save detailed episodes CSV
    os.makedirs(os.path.dirname(episodes_csv_path), exist_ok=True)
    with open(episodes_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n[Stage 6] Saved {len(all_records)} test episode records to {episodes_csv_path}")

    # Build and save aggregated test_results.csv
    agg_fieldnames = [
        "condition", "driver", "mode", "model_seed",
        "laps_completed", "laps_ratio", "mean_progress",
        "mean_steps_to_finish", "mean_offroad_events",
        "mean_centre_distance", "mean_max_centre_distance",
        "mean_steer_jitter", "mean_speed", "autonomy"
    ]
    agg_records = []
    for cond_name, _, _ in conditions_to_run:
        for driver_name, mode, seed_num in DRIVERS:
            drecs = [r for r in all_records if r["condition"] == cond_name and r["driver"] == driver_name]
            laps = sum(1 for r in drecs if r["lap_completed"])
            finished_steps = [r["steps"] for r in drecs if r["lap_completed"]]
            mean_finish_steps = np.mean(finished_steps) if finished_steps else 0.0

            agg_rec = {
                "condition": cond_name,
                "driver": driver_name,
                "mode": mode,
                "model_seed": seed_num if seed_num is not None else "",
                "laps_completed": laps,
                "laps_ratio": f"{laps}/20",
                "mean_progress": round(np.mean([r["track_progress"] for r in drecs]), 2),
                "mean_steps_to_finish": round(mean_finish_steps, 1),
                "mean_offroad_events": round(np.mean([r["offroad_events"] for r in drecs]), 2),
                "mean_centre_distance": round(np.mean([r["mean_centre_distance"] for r in drecs]), 2),
                "mean_max_centre_distance": round(np.mean([r["max_centre_distance"] for r in drecs]), 2),
                "mean_steer_jitter": round(np.mean([r["steer_jitter"] for r in drecs]), 4),
                "mean_speed": round(np.mean([r["mean_speed"] for r in drecs]), 2),
                "autonomy": round(np.mean([r["autonomy"] for r in drecs]), 2),
            }
            agg_records.append(agg_rec)

    with open(results_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fieldnames)
        writer.writeheader()
        writer.writerows(agg_records)
    print(f"[Stage 6] Saved {len(agg_records)} aggregated test records to {results_csv_path}")

    # Print summary per condition
    print("\n" + "=" * 95)
    print("STAGE 6: FINAL TEST-TRACK EVALUATION SUMMARY")
    print(f"{'Condition':<12} | {'Mode':<10} | {'Laps/20':<15} | {'Progress %':<15} | {'Offroad':<12} | {'Centre Dist':<12}")
    print("-" * 95)
    for cond_name, _, _ in conditions_to_run:
        print(f"\n--- Condition: {cond_name} ---")
        for ref_name in ["expert", "zero"]:
            r_list = [r for r in agg_records if r["condition"] == cond_name and r["driver"] == ref_name]
            if r_list:
                r = r_list[0]
                print(f"{cond_name:<12} | {ref_name:<10} | {r['laps_ratio']:<15} | {r['mean_progress']:<15.1f} | {r['mean_offroad_events']:<12.2f} | {r['mean_centre_distance']:<12.2f}")

        for mode in ["plain", "aug", "dagger"]:
            m_list = [r for r in agg_records if r["condition"] == cond_name and r["mode"] == mode]
            laps = [r["laps_completed"] for r in m_list]
            prog = [r["mean_progress"] for r in m_list]
            off = [r["mean_offroad_events"] for r in m_list]
            cd = [r["mean_centre_distance"] for r in m_list]

            lap_str = f"{np.mean(laps):.1f} +/- {np.std(laps):.1f}"
            prog_str = f"{np.mean(prog):.1f}% +/- {np.std(prog):.1f}%"
            off_str = f"{np.mean(off):.2f} +/- {np.std(off):.2f}"
            cd_str = f"{np.mean(cd):.2f} +/- {np.std(cd):.2f}"
            print(f"{'':<12} | {mode:<10} | {lap_str:<15} | {prog_str:<15} | {off_str:<12} | {cd_str:<12}")

    print("=" * 95)
    print(f"Total Stage 6 time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
