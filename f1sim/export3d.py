"""
f1sim/export3d.py - Replay exporter for the 3D WebGL Replay Viewer.

Exports complete race simulation sessions into a compact, self-contained JSON
format matching docs/viewer/COORDS.md:
- Track geometry (centreline, full road polygons, left/right edges, kerbs, single start/finish line, grid boxes)
- Physical constants extracted from Gym CarRacing source
- Multi-car trajectories (positions, hull angles, speed, steer, gas, brake, 4-wheel track-limit flags, lap progress)
- Stewards notices and F1 timing events
"""

import os
import json
import math
import argparse
import numpy as np

from gymnasium.envs.box2d.car_racing import TRACK_WIDTH, FPS
from f1sim.drivers import compute_staggered_grid_positions


# Default 6-Car Grid Specification
SIX_CAR_SPECS = [
    {
        "id": "car_apx",
        "team": "APX",
        "code": "VAN",
        "name": "Alex Vance",
        "type": "model",
        "model_file": "models/dagger_s0.keras",
        "number": 4,
        "grid_pos": 1,
    },
    {
        "id": "car_vel",
        "team": "VEL",
        "code": "SIL",
        "name": "Marco Silva",
        "type": "model",
        "model_file": "models/plain_s0.keras",
        "number": 16,
        "grid_pos": 2,
    },
    {
        "id": "car_ndx",
        "team": "NDX",
        "code": "LIN",
        "name": "Lukas Lind",
        "type": "model",
        "model_file": "models/aug_s0.keras",
        "number": 23,
        "grid_pos": 3,
    },
    {
        "id": "car_zen",
        "team": "ZEN",
        "code": "TAN",
        "name": "Kai Tanaka",
        "type": "expert",
        "number": 88,
        "grid_pos": 4,
    },
    {
        "id": "car_rdr",
        "team": "RDR",
        "code": "OCO",
        "name": "Liam O'Connor",
        "type": "model",
        "model_file": "models/dagger_s1.keras",
        "number": 11,
        "grid_pos": 5,
    },
    {
        "id": "car_orb",
        "team": "ORB",
        "code": "ROS",
        "name": "Elena Rostov",
        "type": "model",
        "model_file": "models/aug_s1.keras",
        "number": 77,
        "grid_pos": 6,
    },
]


def extract_track_geometry(ref_env):
    """
    Extract exact track centerline, ribbon polygons, edges, kerbs, and finish line.
    """
    uw = ref_env.unwrapped
    track = uw.track
    road_poly = uw.road_poly

    centerline = []
    for alpha, beta, tx, ty in track:
        centerline.append([round(float(tx), 3), round(float(ty), 3), round(float(beta), 4)])

    polygons = []
    left_edge = []
    right_edge = []

    for poly, color in road_poly:
        poly_coords = [[round(float(pt[0]), 3), round(float(pt[1]), 3)] for pt in poly]
        polygons.append(poly_coords)
        left_edge.append(poly_coords[0])
        right_edge.append(poly_coords[1])

    # Start / Finish line (perpendicular to track tangent at waypoint 0)
    p0 = track[0]
    beta0 = p0[1]
    w0_x, w0_y = p0[2], p0[3]
    dx = -math.sin(beta0) * TRACK_WIDTH
    dy = -math.cos(beta0) * TRACK_WIDTH
    start_finish = {
        "p1": [round(float(w0_x - dx), 3), round(float(w0_y - dy), 3)],
        "p2": [round(float(w0_x + dx), 3), round(float(w0_y + dy), 3)],
        "tangent": [round(float(math.cos(beta0)), 4), round(float(-math.sin(beta0)), 4)],
    }

    # Staggered grid boxes (6 slots behind waypoint 0)
    track_pts = np.array([[p[2], p[3]] for p in track])
    grid_data = compute_staggered_grid_positions(track_pts, grid_slots=6, spacing_meters=6.5, lateral_offset=1.75)
    grid_boxes = []
    for g in grid_data:
        gx, gy = g["pos"]
        gh = g["heading"]
        # Box dimensions: length 5.5m, width 2.6m
        cos_h = math.cos(gh)
        sin_h = math.sin(gh)
        hl = 2.75
        hw = 1.3
        corners = [
            [-hw, -hl],
            [+hw, -hl],
            [+hw, +hl],
            [-hw, +hl]
        ]
        world_corners = []
        for cx, cy in corners:
            wx = gx + cx * (-sin_h) + cy * cos_h
            wy = gy + cx * cos_h + cy * sin_h
            world_corners.append([round(float(wx), 3), round(float(wy), 3)])
        grid_boxes.append({
            "slot": g["slot"],
            "center": [round(float(gx), 3), round(float(gy), 3)],
            "heading": round(float(gh), 4),
            "corners": world_corners
        })

    # High-curvature kerbs (segments where curvature exceeds threshold)
    kerbs = []
    for i in range(len(track)):
        prev_i = (i - 1) % len(track)
        next_i = (i + 1) % len(track)
        b_prev = track[prev_i][1]
        b_next = track[next_i][1]
        db = abs(b_next - b_prev)
        if db > math.pi:
            db = 2 * math.pi - db
        if db > 0.08:  # Sharp corner
            kerbs.append({
                "index": i,
                "center": [round(float(track[i][2]), 3), round(float(track[i][3]), 3)],
                "curvature": round(float(db), 3),
            })

    return {
        "centerline": centerline,
        "half_width": round(float(TRACK_WIDTH), 4),
        "total_width": round(float(TRACK_WIDTH * 2), 4),
        "polygons": polygons,
        "left_edge": left_edge,
        "right_edge": right_edge,
        "start_finish": start_finish,
        "grid_boxes": grid_boxes,
        "kerbs": kerbs,
    }


def run_and_export_3d(
    track_seed=3003,
    total_laps=10,
    car_specs=None,
    out_path="docs/viewer/data/race_3003.json",
    progress_callback=None,
    traffic_mode="contact",
):
    """
    Run multi-car race simulation and export 3D replay JSON using shared-world Box2D engine.
    """
    from f1sim.world import MultiCarWorld, validate_seed
    validate_seed(track_seed)

    mcw = MultiCarWorld(
        track_seed=track_seed,
        total_laps=total_laps,
        car_specs=car_specs or SIX_CAR_SPECS,
        traffic_mode=traffic_mode,
        fps=FPS,
    )

    def prog(step, p1):
        if progress_callback:
            progress_callback(step, mcw.session_end_step, p1)

    payload = mcw.run_full_race(progress_callback=prog if progress_callback else None)
    mcw.close()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Exported 3D replay to {out_path} ({file_size_mb:.2f} MB, {payload['meta']['total_steps']} steps, {payload['meta']['num_cars']} cars)")
    return payload, out_path


def main():
    parser = argparse.ArgumentParser(description="Export race simulation to 3D replay JSON")
    parser.add_argument("--seed", type=int, default=3003, help="Showcase track seed (3000-3019 or >=6000)")
    parser.add_argument("--laps", type=int, default=10, help="Number of laps to simulate")
    parser.add_argument("--traffic", type=str, default="contact", choices=["contact", "follow"], help="Traffic mode")
    parser.add_argument("--out", type=str, default="docs/viewer/data/race_3003.json", help="Output path for 3D JSON")
    args = parser.parse_args()

    print(f"Starting 3D replay export for Seed {args.seed} ({args.laps} Laps, Traffic: {args.traffic})...")
    def on_prog(step, max_s, p1):
        print(f"Step {step:4d} | P1: {p1.get('code', '---')} | Laps: {p1.get('laps', 0)}/{args.laps} | Gap: {p1.get('gap', '-')}", flush=True)

    run_and_export_3d(track_seed=args.seed, total_laps=args.laps, out_path=args.out, progress_callback=on_prog, traffic_mode=args.traffic)


if __name__ == "__main__":
    main()
