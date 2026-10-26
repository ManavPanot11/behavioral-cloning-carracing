"""
merge_stage6_results.py
Merges per-condition test outputs:
  results/test_episodes_clean.csv
  results/test_episodes_disturbed.csv
  results/test_episodes_speed_1.25.csv
  results/test_episodes_speed_1.50.csv
into:
  results/test_episodes.csv
  results/test_results.csv
"""
import os, sys, csv
import numpy as np

CONDITIONS = ["clean", "disturbed", "speed_1.25", "speed_1.50"]
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
    all_episodes = []
    for cond in CONDITIONS:
        path = f"results/test_episodes_{cond}.csv"
        if not os.path.exists(path):
            print(f"[Merge] Warning: {path} not found.")
            continue
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            recs = list(reader)
            print(f"[Merge] Loaded {len(recs)} records from {path}")
            all_episodes.extend(recs)

    if not all_episodes:
        print("[Merge] Error: No test episodes found to merge!")
        return

    episodes_csv_path = "results/test_episodes.csv"
    fieldnames = list(all_episodes[0].keys())
    with open(episodes_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_episodes)
    print(f"[Merge] Wrote {len(all_episodes)} total test episodes to {episodes_csv_path}")

    # Build test_results.csv
    agg_fieldnames = [
        "condition", "driver", "mode", "model_seed",
        "laps_completed", "laps_ratio", "mean_progress",
        "mean_steps_to_finish", "mean_offroad_events",
        "mean_centre_distance", "mean_max_centre_distance",
        "mean_steer_jitter", "mean_speed", "autonomy"
    ]
    agg_records = []
    for cond in CONDITIONS:
        for driver_name, mode, seed_num in DRIVERS:
            drecs = [r for r in all_episodes if r["condition"] == cond and r["driver"] == driver_name]
            if not drecs:
                continue
            laps = sum(1 for r in drecs if r["lap_completed"] in [True, "True", 1, "1"])
            finished_steps = [float(r["steps"]) for r in drecs if r["lap_completed"] in [True, "True", 1, "1"]]
            mean_finish_steps = np.mean(finished_steps) if finished_steps else 0.0

            agg_rec = {
                "condition": cond,
                "driver": driver_name,
                "mode": mode,
                "model_seed": seed_num if seed_num is not None else "",
                "laps_completed": laps,
                "laps_ratio": f"{laps}/{len(drecs)}",
                "mean_progress": round(np.mean([float(r["track_progress"]) for r in drecs]), 2),
                "mean_steps_to_finish": round(mean_finish_steps, 1),
                "mean_offroad_events": round(np.mean([float(r["offroad_events"]) for r in drecs]), 2),
                "mean_centre_distance": round(np.mean([float(r["mean_centre_distance"]) for r in drecs]), 2),
                "mean_max_centre_distance": round(np.mean([float(r["max_centre_distance"]) for r in drecs]), 2),
                "mean_steer_jitter": round(np.mean([float(r["steer_jitter"]) for r in drecs]), 4),
                "mean_speed": round(np.mean([float(r["mean_speed"]) for r in drecs]), 2),
                "autonomy": round(np.mean([float(r["autonomy"]) for r in drecs]), 2),
            }
            agg_records.append(agg_rec)

    results_csv_path = "results/test_results.csv"
    with open(results_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fieldnames)
        writer.writeheader()
        writer.writerows(agg_records)
    print(f"[Merge] Wrote {len(agg_records)} aggregated records to {results_csv_path}")

    # Summary table
    print("\n" + "=" * 95)
    print("STAGE 6: MERGED FINAL TEST-TRACK EVALUATION SUMMARY")
    print(f"{'Condition':<12} | {'Mode':<10} | {'Laps/20':<15} | {'Progress %':<15} | {'Offroad':<12} | {'Centre Dist':<12}")
    print("-" * 95)
    for cond in CONDITIONS:
        for ref_name in ["expert", "zero"]:
            r_list = [r for r in agg_records if r["condition"] == cond and r["driver"] == ref_name]
            if r_list:
                r = r_list[0]
                print(f"{cond:<12} | {ref_name:<10} | {r['laps_ratio']:<15} | {r['mean_progress']:<15.1f} | {r['mean_offroad_events']:<12.2f} | {r['mean_centre_distance']:<12.2f}")

        for mode in ["plain", "aug", "dagger"]:
            m_list = [r for r in agg_records if r["condition"] == cond and r["mode"] == mode]
            if not m_list:
                continue
            laps = [r["laps_completed"] for r in m_list]
            prog = [r["mean_progress"] for r in m_list]
            off = [r["mean_offroad_events"] for r in m_list]
            cd = [r["mean_centre_distance"] for r in m_list]

            lap_str = f"{np.mean(laps):.1f} +/- {np.std(laps):.1f}"
            prog_str = f"{np.mean(prog):.1f}% +/- {np.std(prog):.1f}%"
            off_str = f"{np.mean(off):.2f} +/- {np.std(off):.2f}"
            cd_str = f"{np.mean(cd):.2f} +/- {np.std(cd):.2f}"
            print(f"{'':<12} | {mode:<10} | {lap_str:<15} | {prog_str:<15} | {off_str:<12} | {cd_str:<12}")
        print("-" * 95)

if __name__ == "__main__":
    main()
