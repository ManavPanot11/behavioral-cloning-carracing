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
    parser_race.add_argument("--laps", type=int, default=10, help="Number of laps to race")
    parser_race.add_argument("--video", type=str, default="videos/f1_race_3003.mp4", help="Path to save hero MP4")
    parser_race.add_argument("--gif", type=str, default="videos/f1_race_3003.gif", help="Path to save showcase GIF")
    parser_race.add_argument("--telemetry", type=str, default="data/telemetry_latest.json.gz", help="Path to save replay telemetry")

    # weekend command
    parser_wk = subparsers.add_parser("weekend", help="Run full Grand Prix race weekend")
    parser_wk.add_argument("--seed", type=int, default=3003, help="Showcase track seed (3000-3019)")
    parser_wk.add_argument("--laps", type=int, default=10, help="Number of laps to race")
    parser_wk.add_argument("--video", type=str, default="videos/f1_weekend_3003.mp4", help="Path to save hero MP4")
    parser_wk.add_argument("--gif", type=str, default="videos/f1_weekend_3003.gif", help="Path to save showcase GIF")
    parser_wk.add_argument("--telemetry", type=str, default="data/telemetry_latest.json.gz", help="Path to save replay telemetry")

    # export3d command
    parser_exp = subparsers.add_parser("export3d", help="Run race and export 3D replay JSON for WebGL viewer")
    parser_exp.add_argument("--seed", type=int, default=3003, help="Showcase track seed (3000-3019)")
    parser_exp.add_argument("--laps", type=int, default=10, help="Number of laps to race")
    parser_exp.add_argument("--out", type=str, default="docs/viewer/data/race_3003.json", help="Path to save 3D JSON")

    # video3d command
    parser_vid = subparsers.add_parser("video3d", help="Generate 3D WebGL replay video, GIF, or screenshots")
    parser_vid.add_argument("--replay", type=str, default="docs/viewer/data/race_3003.json", help="Path to replay JSON")
    parser_vid.add_argument("--out", type=str, default="docs/media/race3d.mp4", help="Output video or GIF path")
    parser_vid.add_argument("--camera", type=str, default="mixed", choices=["mixed", "chase", "onboard", "tv", "topdown", "trackside", "orbit"], help="Camera mode")
    parser_vid.add_argument("--size", type=str, default="1280x720", help="Resolution WxH")
    parser_vid.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser_vid.add_argument("--duration", type=float, default=35.0, help="Duration in seconds")
    parser_vid.add_argument("--start-step", type=int, default=0, help="Starting step")
    parser_vid.add_argument("--screenshots", action="store_true", help="Generate 6 showcase screenshots")
    parser_vid.add_argument("--benchmark", action="store_true", help="Measure WebGL performance benchmark")
    parser_vid.add_argument("--port", type=int, default=8080, help="Local HTTP server port")

    # live command
    parser_live = subparsers.add_parser("live", help="Launch live interactive race streaming to 3D viewer")
    parser_live.add_argument("--seed", type=str, default="random", help="Track seed (integer >= 6000 or 'random')")
    parser_live.add_argument("--laps", type=int, default=10, help="Number of laps to race")
    parser_live.add_argument("--traffic", type=str, default="contact", choices=["contact", "follow"], help="Traffic mode")
    parser_live.add_argument("--pace-variance", type=float, default=0.03, help="Per-driver pace variance (0.03 for +/-3%%, 0 to disable)")
    parser_live.add_argument("--port", type=int, default=8080, help="Local HTTP static server port")
    parser_live.add_argument("--ws-port", type=int, default=8765, help="WebSocket streaming port")
    parser_live.add_argument("--out", type=str, default=None, help="Path to save replay at conclusion")
    parser_live.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")

    # make_replays command
    parser_mr = subparsers.add_parser("make_replays", help="Generate N diverse replays and index.json catalog")
    parser_mr.add_argument("--n", type=int, default=8, help="Number of replays to generate (default: 8)")
    parser_mr.add_argument("--laps", type=int, default=10, help="Number of laps per replay")
    parser_mr.add_argument("--out-dir", type=str, default="docs/viewer/data", help="Output directory for replays")

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
    elif args.command == "video3d":
        from f1sim.video3d import measure_performance, capture_screenshots, generate_video
        if getattr(args, "benchmark", False):
            measure_performance(replay_path=args.replay, port=args.port)
        elif args.screenshots:
            capture_screenshots(replay_path=args.replay, out_dir="docs/media", port=args.port)
        else:
            parts = args.size.lower().split("x")
            w, h = int(parts[0]), int(parts[1])
            generate_video(
                replay_path=args.replay,
                out_path=args.out,
                camera_mode=args.camera,
                target_width=w,
                target_height=h,
                fps=args.fps,
                duration=args.duration,
                start_step=args.start_step,
                port=args.port,
            )
    elif args.command == "live":
        from f1sim.live import run_live_race
        run_live_race(
            seed=args.seed,
            laps=args.laps,
            traffic=args.traffic,
            pace_variance=args.pace_variance,
            http_port=args.port,
            ws_port=args.ws_port,
            out_path=args.out,
            open_browser=not args.no_browser,
        )
    elif args.command == "make_replays":
        from f1sim.live import make_replays
        make_replays(n=args.n, laps=args.laps, out_dir=args.out_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
