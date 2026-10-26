"""
f1sim/__main__.py - CLI dispatcher for f1sim commands.
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="f1sim - Autonomous Race Simulator & Telemetry")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # doctor command
    subparsers.add_parser("doctor", help="Run system diagnostics and verify simulator health")

    # race command
    parser_race = subparsers.add_parser("race", help="Launch live F1 dashboard race")
    parser_race.add_argument("--seed", type=int, default=3003, help="Showcase track seed (3000-3019)")
    parser_race.add_argument("--laps", type=int, default=3, help="Number of laps to race")
    parser_race.add_argument("--video", type=str, default="videos/f1_race_3003.mp4", help="Path to save hero MP4")
    parser_race.add_argument("--gif", type=str, default="videos/f1_race_3003.gif", help="Path to save showcase GIF")
    parser_race.add_argument("--telemetry", type=str, default="data/telemetry_latest.json.gz", help="Path to save replay telemetry")

    # weekend command
    parser_wk = subparsers.add_parser("weekend", help="Run full Grand Prix race weekend")
    parser_wk.add_argument("--seed", type=int, default=3003, help="Showcase track seed (3000-3019)")
    parser_wk.add_argument("--laps", type=int, default=3, help="Number of laps to race")
    parser_wk.add_argument("--video", type=str, default="videos/f1_weekend_3003.mp4", help="Path to save hero MP4")
    parser_wk.add_argument("--gif", type=str, default="videos/f1_weekend_3003.gif", help="Path to save showcase GIF")
    parser_wk.add_argument("--telemetry", type=str, default="data/telemetry_latest.json.gz", help="Path to save replay telemetry")

    # export3d command
    parser_exp = subparsers.add_parser("export3d", help="Run race and export 3D replay JSON for WebGL viewer")
    parser_exp.add_argument("--seed", type=int, default=3003, help="Showcase track seed (3000-3019)")
    parser_exp.add_argument("--laps", type=int, default=2, help="Number of laps to race")
    parser_exp.add_argument("--out", type=str, default="docs/viewer/data/race_3003.json", help="Path to save 3D JSON")

    args = parser.parse_args()

    if args.command == "doctor":
        from f1sim.doctor import run_doctor
        sys.exit(run_doctor())
    elif args.command in ("race", "weekend"):
        from f1sim.session import RaceSession
        print("=" * 60)
        print(f"Starting Autonomous Grand Prix — Seed {args.seed} ({args.laps} Laps)")
        print("=" * 60)
        session = RaceSession(track_seed=args.seed, total_laps=args.laps)
        session.setup_drivers()

        def on_progress(step, max_steps, lb):
            p1 = lb[0] if lb else {}
            print(f"Step {step:4d} | Leader: {p1.get('code', '---')} | Laps: {p1.get('laps', 0)}/{args.laps} | Gap: {p1.get('gap', '-')}", flush=True)

        video_path = getattr(args, "video", None)
        gif_path = getattr(args, "gif", None)
        telemetry_path = getattr(args, "telemetry", "data/telemetry_latest.json.gz")

        results = session.run_race(
            record_video=bool(video_path),
            video_path=video_path,
            record_gif=bool(gif_path),
            gif_path=gif_path,
            progress_callback=on_progress,
        )

        # Save telemetry
        if telemetry_path:
            os.makedirs(os.path.dirname(telemetry_path) if os.path.dirname(telemetry_path) else ".", exist_ok=True)
            session.recorder.save_json_gz(telemetry_path)
            print(f"Saved telemetry trace to {telemetry_path}")

        print("\n" + "=" * 60)
        print("GRAND PRIX OFFICIAL CLASSIFICATION")
        print("=" * 60)
        for row in results["leaderboard"]:
            fast_lap_str = " (FASTEST LAP)" if row.get("is_fastest_lap") else ""
            print(f"P{row['pos']} | {row['code']} - {row['name']} ({row['team_code']}) | Laps: {row['laps']} | Gap: {row['gap']} | Best: {row['best_lap']}s{fast_lap_str}")

        print(f"\nHero Video: {video_path}")
        print(f"Showcase GIF: {gif_path}")
        print("=" * 60)
    elif args.command == "export3d":
        from f1sim.export3d import run_and_export_3d
        print("=" * 60)
        print(f"Exporting 3D Grand Prix Replay — Seed {args.seed} ({args.laps} Laps)")
        print("=" * 60)
        def on_progress(step, max_steps, p1):
            print(f"Step {step:4d} | Leader: {p1.get('code', '---')} | Laps: {p1.get('laps', 0)}/{args.laps} | Gap: {p1.get('gap', '-')}", flush=True)
        run_and_export_3d(track_seed=args.seed, total_laps=args.laps, out_path=args.out, progress_callback=on_progress)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
