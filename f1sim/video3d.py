"""
f1sim/video3d.py - Deterministic 3D WebGL Replay Video & Media Generator.

Captures frame-by-frame 3D WebGL race replay using headless Chrome / Selenium,
and encodes broadcast-quality MP4 videos, lightweight hero GIFs, and high-res screenshots.
"""

import os
import sys
import time
import io
import json
import argparse
import http.server
import threading
from PIL import Image
import imageio
import numpy as np


def ensure_server(port=8080):
    """Ensure a local HTTP server is running serving the repo root."""
    import urllib.request
    try:
        req = urllib.request.urlopen(f"http://localhost:{port}/docs/viewer/index.html", timeout=1.0)
        if req.status == 200:
            return None  # Server already running
    except Exception:
        pass

    # Start lightweight server in daemon thread
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Quiet

    httpd = http.server.ThreadingHTTPServer(("localhost", port), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.5)
    return httpd


def init_browser(width=1920, height=1080):
    """Initialize headless Chrome with WebGL support enabled."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opts = Options()
    opts.add_argument("--headless=new")
    # Add small margin for browser window framing
    opts.add_argument(f"--window-size={width + 16},{height + 110}")
    opts.add_argument("--enable-webgl")
    opts.add_argument("--ignore-gpu-blocklist")
    opts.add_argument("--disable-gpu-vsync")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=opts)
    return driver


def capture_screenshots(replay_path, out_dir="docs/media", port=8080):
    """Capture 6 distinct showcase screenshots from replay."""
    os.makedirs(out_dir, exist_ok=True)
    server = ensure_server(port)

    # Relative replay URL from docs/viewer/
    replay_file = os.path.basename(replay_path)
    viewer_url = f"http://localhost:{port}/docs/viewer/index.html?capture=1&replay=data/{replay_file}"

    shots = [
        {"name": "screen_01_grid_lights.png", "frame": 0, "cam": 1, "desc": "Starting Grid & 5-Light Gantry"},
        {"name": "screen_02_chase_action.png", "frame": 350, "cam": 1, "desc": "Chase Cam Multi-Car Duel"},
        {"name": "screen_03_onboard_cockpit.png", "frame": 520, "cam": 2, "desc": "Onboard Nose Camera"},
        {"name": "screen_04_tv_helicopter.png", "frame": 800, "cam": 3, "desc": "TV Helicopter Orbit"},
        {"name": "screen_05_topdown_seam.png", "frame": 120, "cam": 4, "desc": "Top-Down Single Start/Finish Line"},
        {"name": "screen_06_trackside_handover.png", "frame": 650, "cam": 5, "desc": "Trackside Dynamic Handover"},
    ]

    print(f"Capturing {len(shots)} showcase screenshots from {replay_path}...")
    driver = init_browser(1920, 1080)
    try:
        driver.get(viewer_url)
        time.sleep(2.0)

        for s in shots:
            driver.execute_script(f"window.seekFrame({s['frame']}, {s['cam']});")
            time.sleep(0.15)
            png_bytes = driver.get_screenshot_as_png()
            img = Image.open(io.BytesIO(png_bytes))
            # Resize if necessary to exact 1920x1080
            if img.size != (1920, 1080):
                img = img.resize((1920, 1080), Image.Resampling.LANCZOS)
            out_file = os.path.join(out_dir, s["name"])
            img.save(out_file, "PNG", optimize=True)
            print(f"  Saved: {out_file} ({s['desc']}) - {os.path.getsize(out_file)/1024:.1f} KB")

    finally:
        driver.quit()


def generate_video(
    replay_path,
    out_path,
    camera_mode="mixed",
    target_width=1280,
    target_height=720,
    fps=30,
    duration=35.0,
    start_step=0,
    port=8080,
):
    """
    Generate deterministic video or GIF from 3D replay.
    camera_mode: 'chase' (1), 'onboard' (2), 'tv' (3), 'topdown' (4), 'trackside' (5), 'orbit' (6), or 'mixed'
    """
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    server = ensure_server(port)

    # Load replay metadata for step count
    with open(replay_path, "r", encoding="utf-8") as f:
        meta = json.load(f).get("meta", {})
    total_sim_steps = meta.get("total_steps", 2000)
    sim_fps = meta.get("fps", 50)

    # Frame timing
    total_frames = int(duration * fps)
    step_increment = (50.0 / fps)  # 50 Hz sim mapped to target fps

    replay_file = os.path.basename(replay_path)
    viewer_url = f"http://localhost:{port}/docs/viewer/index.html?capture=1&replay=data/{replay_file}"

    cam_mode_map = {
        "chase": 1,
        "onboard": 2,
        "tv": 3,
        "topdown": 4,
        "trackside": 5,
        "orbit": 6,
    }

    is_gif = out_path.lower().endswith(".gif")
    target_mp4 = out_path if not is_gif else out_path + ".tmp.mp4"

    print("=" * 60)
    print(f"Generating {'GIF' if is_gif else 'MP4'} Media: {out_path}")
    print(f"Resolution: {target_width}x{target_height} @ {fps} fps | Duration: {duration:.1f}s ({total_frames} frames)")
    print(f"Camera Mode: {camera_mode}")
    print("=" * 60)

    driver = init_browser(target_width, target_height)
    try:
        driver.get(viewer_url)
        time.sleep(2.0)

        # Video writer setup
        writer = imageio.get_writer(
            target_mp4,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            ffmpeg_params=["-preset", "fast", "-crf", "22"],
        )

        t_start = time.time()
        for frame_idx in range(total_frames):
            sim_step = min(total_sim_steps - 1, start_step + frame_idx * step_increment)

            # Determine camera for mixed mode
            if camera_mode == "mixed":
                t_sec = frame_idx / fps
                if t_sec < 4.0:
                    current_cam = 1  # Chase cam at start
                elif t_sec < 11.0:
                    current_cam = 2  # Onboard nose cam into turn 1
                elif t_sec < 20.0:
                    current_cam = 3  # TV Helicopter
                elif t_sec < 28.0:
                    current_cam = 5  # Trackside pass
                else:
                    current_cam = 1  # Chase cam to finish
            else:
                current_cam = cam_mode_map.get(camera_mode.lower(), 1)

            driver.execute_script(f"window.seekFrame({sim_step:.2f}, {current_cam});")
            png_bytes = driver.get_screenshot_as_png()
            img = Image.open(io.BytesIO(png_bytes))

            if img.size != (target_width, target_height):
                img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

            writer.append_data(np.array(img.convert("RGB")))

            if (frame_idx + 1) % 50 == 0 or frame_idx == total_frames - 1:
                elapsed = time.time() - t_start
                rate = (frame_idx + 1) / elapsed
                print(f"  Rendered frame {frame_idx + 1:4d}/{total_frames} ({rate:.1f} fps) | Sim Step: {sim_step:.1f}")

        writer.close()

        if not is_gif:
            print(f"\nMP4 Generated: {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB)")
        else:
            print("Converting MP4 to palette-optimized GIF via ffmpeg...")
            import subprocess
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            # 2-pass palettegen filter to produce crisp sub-10MB GIF
            cmd = [
                ffmpeg_exe, "-y", "-i", target_mp4,
                "-vf", f"fps={fps},scale={target_width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
                out_path
            ]
            subprocess.run(cmd, check=True)
            if os.path.exists(target_mp4):
                os.remove(target_mp4)
            print(f"\nGIF Generated: {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB)")

    finally:
        driver.quit()


def measure_performance(replay_path="docs/viewer/data/race_3003.json", port=8080):
    """Benchmark Three.js draw calls, triangles, and render frame time."""
    server = ensure_server(port)
    replay_file = os.path.basename(replay_path)
    viewer_url = f"http://localhost:{port}/docs/viewer/index.html?capture=1&replay=data/{replay_file}"

    driver = init_browser(1920, 1080)
    try:
        driver.get(viewer_url)
        time.sleep(2.0)

        perf = driver.execute_script("""
            const app = window.app;
            const renderer = app.scene.renderer;
            const scene = app.scene.scene;
            const cam = app.cameraDirector.activeCamera;
            
            // Warm up
            renderer.render(scene, cam);
            
            const N = 120;
            const t0 = performance.now();
            for (let i = 0; i < N; i++) {
                app.currentStep = (200 + i) % app.maxSteps;
                app.updateCars(app.currentStep);
                renderer.render(scene, cam);
            }
            const t1 = performance.now();
            const avgMs = (t1 - t0) / N;
            
            return {
                drawCalls: renderer.info.render.calls,
                triangles: renderer.info.render.triangles,
                geometries: renderer.info.memory.geometries,
                textures: renderer.info.memory.textures,
                avgFrameTimeMs: avgMs,
                effectiveFps: 1000 / avgMs
            };
        """)

        print("=" * 60)
        print("3D WEBGL ENGINE PERFORMANCE BENCHMARK")
        print("=" * 60)
        print(f"Draw Calls:      {perf['drawCalls']}")
        print(f"Triangles:       {perf['triangles']:,}")
        print(f"Geometries:      {perf['geometries']}")
        print(f"Textures:        {perf['textures']}")
        print(f"Avg Frame Time:  {perf['avgFrameTimeMs']:.3f} ms")
        print(f"Effective Rate:  {perf['effectiveFps']:.1f} FPS (Target: 60 FPS)")
        print("=" * 60)
        return perf
    finally:
        driver.quit()


def main():
    parser = argparse.ArgumentParser(description="f1sim video3d - 3D Replay Video & Media Generator")
    parser.add_argument("--replay", type=str, default="docs/viewer/data/race_3003.json", help="Path to replay JSON")
    parser.add_argument("--out", type=str, default="docs/media/race3d.mp4", help="Output file path (.mp4 or .gif)")
    parser.add_argument("--camera", type=str, default="mixed", choices=["mixed", "chase", "onboard", "tv", "topdown", "trackside", "orbit"], help="Camera mode")
    parser.add_argument("--size", type=str, default="1280x720", help="Resolution WxH (e.g. 1920x1080, 1280x720, 960x540)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--duration", type=float, default=35.0, help="Video duration in seconds")
    parser.add_argument("--start-step", type=int, default=0, help="Starting sim step")
    parser.add_argument("--screenshots", action="store_true", help="Generate 6 showcase screenshots")
    parser.add_argument("--benchmark", action="store_true", help="Run WebGL performance benchmark")
    parser.add_argument("--port", type=int, default=8080, help="Local HTTP server port")

    args = parser.parse_args()

    parts = args.size.lower().split("x")
    width, height = int(parts[0]), int(parts[1])

    if args.benchmark:
        measure_performance(replay_path=args.replay, port=args.port)
        return

    if args.screenshots:
        capture_screenshots(replay_path=args.replay, out_dir="docs/media", port=args.port)
        return

    generate_video(
        replay_path=args.replay,
        out_path=args.out,
        camera_mode=args.camera,
        target_width=width,
        target_height=height,
        fps=args.fps,
        duration=args.duration,
        start_step=args.start_step,
        port=args.port,
    )


if __name__ == "__main__":
    main()
