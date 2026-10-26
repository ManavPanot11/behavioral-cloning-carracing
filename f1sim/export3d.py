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
import sys
import json
import math
import argparse
import numpy as np
import gymnasium as gym

from gymnasium.envs.box2d.car_racing import TRACK_WIDTH, SCALE, PLAYFIELD, FPS
from gymnasium.envs.box2d.car_dynamics import SIZE, WHEELPOS, WHEEL_R, WHEEL_W

from f1sim.env import make_env
from f1sim.drivers import (
    TEAMS, ExpertDriver, ModelDriver,
    compute_staggered_grid_positions
)
from f1sim.timing import TimingSystem
from f1sim.stewards import Stewards, RulesConfig
from expert import wheels_on_track, get_car_state


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
        # poly is 4 vertices: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        # (x1, y1) and (x4, y4) are left edge; (x2, y2) and (x3, y3) are right edge
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
        # 4 corners
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
            # Outer kerb
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
):
    """
    Run multi-car race simulation and export 3D replay JSON.
    """
    assert 3000 <= track_seed <= 3019, f"Showcase seed must be 3000-3019, got {track_seed}"
    specs = car_specs or SIX_CAR_SPECS
    num_cars = len(specs)

    # 1. Initialize environments and drivers
    envs = {}
    drivers = {}
    driver_meta = {}
    timing = TimingSystem(fps=FPS, num_checkpoints=100)
    stewards = Stewards(fps=FPS)

    for spec in specs:
        cid = spec["id"]
        code = spec["code"]
        name = spec["name"]
        team = spec["team"]
        driver_meta[cid] = spec

        timing.register_driver(cid, code, name, team)
        stewards.register_car(cid, code)

        env = make_env(seed=track_seed, max_steps=max(15000, total_laps * 1600 + 1000), render_mode="rgb_array", race_mode=True)
        envs[cid] = env

        if spec["type"] == "expert":
            driver = ExpertDriver(name=name, code=code, team=team)
        else:
            driver = ModelDriver(spec["model_file"], name=name, code=code, team=team)
        drivers[cid] = driver

    # 2. Extract track geometry from reference env
    ref_env = list(envs.values())[0]
    obs0, _ = ref_env.reset(seed=track_seed)
    track_geom = extract_track_geometry(ref_env)
    track_pts = np.array([[p[0], p[1]] for p in track_geom["centerline"]])
    grid_boxes = track_geom["grid_boxes"]

    # 3. Initialize environments on track
    obs_dict = {}
    for i, spec in enumerate(specs):
        cid = spec["id"]
        env = envs[cid]
        obs, _ = env.reset(seed=track_seed)
        obs_dict[cid] = obs

    # 4. Simulation Trajectory Storage
    car_trajectories = {cid: {
        "x": [],
        "y": [],
        "heading": [],
        "hull_angle": [],
        "speed": [],
        "steer": [],
        "gas": [],
        "brake": [],
        "wheels": [],
        "progress": [],
        "lap": [],
        "offroad": [],
        "centre_dist": [],
        "lateral_g": []
    } for cid in drivers}

    race_events = [
        {"step": 0, "time": 0.0, "type": "lights_on", "text": "RED LIGHTS ILLUMINATING"},
        {"step": 20, "time": 0.4, "type": "lights_seq", "text": "FORMATION GRID SET"},
        {"step": 50, "time": 1.0, "type": "lights_out", "text": "LIGHTS OUT AND AWAY WE GO!"},
    ]

    winner_id = None
    winner_finish_step = -1
    max_steps = max(3500, total_laps * 1500 + 1000)

    # 5. Main Race Simulation Loop
    for step in range(max_steps):
        # Step each car
        for i, (cid, env) in enumerate(envs.items()):
            slot_info = grid_boxes[i]
            gx, gy = slot_info["center"]
            gh = slot_info["heading"]

            launch_delay = i * 2  # 0 to 10 frames reaction time
            launch_step = 50 + launch_delay

            if step < launch_step:
                # Pre-start: positioned in assigned starting grid slot holding brakes
                car_x, car_y = gx, gy
                heading = gh
                hull_angle = gh
                speed = 0.0
                act = [0.0, 0.0, 1.0]
                wheel_flags = [True, True, True, True]
                progress_pct = 0.0
                lap_num = 0
                is_offroad = False
                cdist = 0.0
                lat_g = 0.0
            elif timing.drivers[cid].laps_completed >= total_laps or stewards.states[cid].is_dnf:
                # Finished race or retired: hold terminal state
                traj = car_trajectories[cid]
                car_x = traj["x"][-1]
                car_y = traj["y"][-1]
                heading = traj["heading"][-1]
                hull_angle = traj["hull_angle"][-1]
                speed = 0.0
                act = [0.0, 0.0, 1.0]
                wheel_flags = traj["wheels"][-1]
                progress_pct = traj["progress"][-1]
                lap_num = traj["lap"][-1]
                is_offroad = traj["offroad"][-1]
                cdist = traj["centre_dist"][-1]
                lat_g = 0.0
            else:
                # Normal simulation step
                act = drivers[cid].act(obs_dict[cid], env, step=step)
                obs, reward, terminated, truncated, info = env.step(act)
                obs_dict[cid] = obs

                car_x, car_y, heading, speed = get_car_state(env)
                hull_angle = float(env.unwrapped.car.hull.angle)
                uw_car = env.unwrapped.car
                wheel_flags = [bool(len(w.tiles) > 0) for w in uw_car.wheels]  # FL, FR, RL, RR
                on_track_wheels = sum(wheel_flags)
                is_offroad = (on_track_wheels == 0)

                # Stewards rules check
                stewards.check_step(step, cid, is_offroad, speed)

                # Timing system
                lap_finished = info.get("lap_finished", False)
                progress_pct = info.get("progress_pct", 0.0)
                timing.update_driver_progress(cid, current_step=step, progress_pct=progress_pct, lap_finished=lap_finished)

                # Centre-line distance
                diff = track_pts - np.array([car_x, car_y])
                cdist = float(np.sqrt(np.min(np.sum(diff ** 2, axis=1))))
                lat_g = round(float(abs(act[0]) * speed * 0.015), 2)

                # Winner check
                if timing.drivers[cid].laps_completed >= total_laps and winner_id is None:
                    winner_id = cid
                    winner_finish_step = step
                    race_events.append({
                        "step": step,
                        "time": round(step / FPS, 2),
                        "type": "chequered_flag",
                        "text": f"CHEQUERED FLAG: {timing.drivers[cid].code} WINS THE GRAND PRIX!"
                    })

            # Record state
            traj = car_trajectories[cid]
            traj["x"].append(round(float(car_x), 2))
            traj["y"].append(round(float(car_y), 2))
            traj["heading"].append(round(float(heading), 3))
            traj["hull_angle"].append(round(float(hull_angle), 3))
            traj["speed"].append(round(float(speed), 1))
            traj["steer"].append(round(float(act[0]), 2))
            traj["gas"].append(round(float(act[1]), 2))
            traj["brake"].append(round(float(act[2]), 2))
            traj["wheels"].append(wheel_flags)
            traj["progress"].append(round(float(progress_pct), 1))
            traj["lap"].append(int(timing.drivers[cid].current_lap))
            traj["offroad"].append(is_offroad)
            traj["centre_dist"].append(round(float(cdist), 1))
            traj["lateral_g"].append(round(float(lat_g), 1))

        # Append new steward events
        recent_notices = stewards.get_recent_notices(step, window_steps=1)
        for notice in recent_notices:
            if notice.step == step:
                race_events.append({
                    "step": step,
                    "time": round(step / FPS, 2),
                    "type": notice.level,
                    "text": notice.message,
                    "car_id": notice.car_id,
                })

        # Chequered flag cooldown
        all_finished = all(
            (timing.drivers[cid].laps_completed >= total_laps or stewards.states[cid].is_dnf)
            for cid in drivers
        )
        if all_finished or (winner_id is not None and (step - winner_finish_step) >= 350):
            break

        if progress_callback and step % 250 == 0:
            lb = timing.compute_leaderboard(current_step=step)
            p1 = lb[0] if lb else {}
            progress_callback(step, max_steps, p1)

    for env in envs.values():
        env.close()

    # 6. Final Leaderboard and Classification
    final_lb = timing.compute_leaderboard(current_step=step)
    serializable_lb = []
    for row in final_lb:
        r = dict(row)
        for s_key in ("s1", "s2", "s3"):
            sec = r.get(s_key)
            if sec is not None and hasattr(sec, "time_s"):
                r[s_key] = {"time_s": round(float(sec.time_s), 3), "status": sec.status}
            else:
                r[s_key] = None
        serializable_lb.append(r)

    # 7. Assemble Complete 3D Replay JSON
    drivers_list = []
    for spec in specs:
        cid = spec["id"]
        team_info = TEAMS.get(spec["team"], {})
        drivers_list.append({
            "id": cid,
            "code": spec["code"],
            "name": spec["name"],
            "team": spec["team"],
            "team_name": team_info.get("name", spec["team"]),
            "number": spec.get("number", 0),
            "color": team_info.get("hex", "#FFFFFF"),
            "model": spec.get("model_file", "expert"),
            "grid_pos": spec.get("grid_pos", 1),
        })

    payload = {
        "meta": {
            "title": f"Autonomous Grand Prix — 3D Replay Seed {track_seed}",
            "track_seed": track_seed,
            "total_laps": total_laps,
            "fps": FPS,
            "num_cars": num_cars,
            "total_steps": step + 1,
            "duration_s": round((step + 1) / FPS, 2),
            "lights_out_step": 50,
            "sim_constants": {
                "SCALE": SCALE,
                "TRACK_WIDTH": TRACK_WIDTH,
                "ROAD_HALF_WIDTH": TRACK_WIDTH,
                "PLAYFIELD": PLAYFIELD,
                "CAR_LENGTH": 5.0,
                "CAR_WIDTH": 2.4,
                "WHEEL_RADIUS": WHEEL_R * SIZE,
                "WHEEL_WIDTH": WHEEL_W * 2 * SIZE,
                "WHEELBASE": (80 + 82) * SIZE,
            },
            "disclaimer": "3D VISUALISATION - SIM PHYSICS IS 2D. Unofficial F1-inspired telemetry.",
            "speed_conversion": "KM/H = world_speed * 3.6 (exact m/s to km/h conversion)",
        },
        "drivers": drivers_list,
        "track": track_geom,
        "events": race_events,
        "leaderboard": serializable_lb,
        "cars": car_trajectories,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Exported 3D replay to {out_path} ({file_size_mb:.2f} MB, {step + 1} steps, {num_cars} cars)")
    return payload, out_path


def main():
    parser = argparse.ArgumentParser(description="Export race simulation to 3D replay JSON")
    parser.add_argument("--seed", type=int, default=3003, help="Showcase track seed (3000-3019)")
    parser.add_argument("--laps", type=int, default=2, help="Number of laps to simulate")
    parser.add_argument("--out", type=str, default="docs/viewer/data/race_3003.json", help="Output path for 3D JSON")
    args = parser.parse_args()

    print(f"Starting 3D replay export for Seed {args.seed} ({args.laps} Laps)...")
    def on_prog(step, max_s, p1):
        print(f"Step {step:4d} | P1: {p1.get('code', '---')} | Laps: {p1.get('laps', 0)}/{args.laps} | Gap: {p1.get('gap', '-')}", flush=True)

    run_and_export_3d(track_seed=args.seed, total_laps=args.laps, out_path=args.out, progress_callback=on_prog)


if __name__ == "__main__":
    main()
