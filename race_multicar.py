"""
race_multicar.py - 10-Lap Multi-Car Race with Formula 1 Broadcast Scoreboard & Multi-Camera Views
Simulates 4 autonomous cars racing together in Gymnasium CarRacing-v3 Box2D physics.
Competitors:
  1. Car 1 (Red):    DAGGER (models/dagger_s0.keras)
  2. Car 2 (Blue):   AUGMENTED (models/aug_s0.keras)
  3. Car 3 (Green):  PLAIN (models/plain_s0.keras)
  4. Car 4 (Yellow): EXPERT (Pure-Pursuit Track Geometry Expert)

Outputs:
  - videos/race_10laps_f1.mp4 (Broadcast / Leader Chase Cam)
  - videos/race_dagger.mp4    (Car 1 DAGGER Onboard Cam)
  - videos/race_augmented.mp4 (Car 2 AUGMENTED Onboard Cam)
  - videos/race_plain.mp4     (Car 3 PLAIN Onboard Cam)
  - videos/race_expert.mp4    (Car 4 EXPERT Onboard Cam)
  - videos/race_quad.mp4      (F1 Quad-View 2x2 Multi-Cam)
  - data/race_telemetry.json  (Synchronized live telemetry for web viewer)
"""

import os, sys, math, time, json, argparse
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("KERAS_BACKEND", "torch")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont
import imageio
import torch
import keras
import gymnasium as gym
from gymnasium.envs.box2d import car_racing, car_dynamics

# Constants
FPS = 50.0
VIDEO_W = 608
VIDEO_H = 400
STATE_W = 96
STATE_H = 96
TRACK_WIDTH = car_racing.TRACK_WIDTH


def setup_checkered_finish_line(env):
    """
    Explicitly replace tile 0 with a bold, high-contrast Formula 1
    black-and-white checkered start/finish line strip.
    """
    uw = env.unwrapped
    if not hasattr(uw, "road_poly") or len(uw.road_poly) == 0:
        return

    poly0 = uw.road_poly[0][0]
    road1_l, road1_r, road2_r, road2_l = [np.array(p) for p in poly0]

    checkered_polys = []
    N_COLS = 12
    N_ROWS = 4
    for r in range(N_ROWS):
        for c in range(N_COLS):
            u0, u1 = c / N_COLS, (c + 1) / N_COLS
            v0, v1 = r / N_ROWS, (r + 1) / N_ROWS

            p_top_l = (1 - v1) * ((1 - u0) * road2_l + u0 * road2_r) + v1 * ((1 - u0) * road1_l + u0 * road1_r)
            p_top_r = (1 - v1) * ((1 - u1) * road2_l + u1 * road2_r) + v1 * ((1 - u1) * road1_l + u1 * road1_r)
            p_bot_r = (1 - v0) * ((1 - u1) * road2_l + u1 * road2_r) + v0 * ((1 - u1) * road1_l + u1 * road1_r)
            p_bot_l = (1 - v0) * ((1 - u0) * road2_l + u0 * road2_r) + v0 * ((1 - u0) * road1_l + u0 * road1_r)

            # High contrast F1 checkered pattern
            color = (252, 252, 252) if (r + c) % 2 == 0 else (22, 22, 22)
            checkered_polys.append(([p_top_l.tolist(), p_top_r.tolist(), p_bot_r.tolist(), p_bot_l.tolist()], color))

    # Bold white finish line edge at track[0]
    line_w = 0.08
    p1 = road1_l
    p2 = road1_r
    p3 = road1_r - line_w * (road1_r - road2_r)
    p4 = road1_l - line_w * (road1_l - road2_l)
    checkered_polys.append(([p1.tolist(), p2.tolist(), p3.tolist(), p4.tolist()], (255, 255, 255)))

    uw.road_poly = checkered_polys + uw.road_poly[1:]


def get_car_state(car):
    """Return (x, y, heading_angle, speed) for a car."""
    hull = car.hull
    x, y = hull.position
    heading = hull.angle + math.pi / 2
    vx, vy = hull.linearVelocity
    speed = math.sqrt(vx**2 + vy**2)
    return x, y, heading, speed


def find_nearest_waypoint_local(track, x, y, last_wp=None, max_behind=6, max_ahead=22):
    """
    Find index of nearest track waypoint to (x, y).
    Uses a localized forward-focused window around last_wp to prevent false
    jumps across hairpins or parallel track sections.
    """
    N = len(track)
    if last_wp is None:
        best_idx = 0
        best_dist = float("inf")
        for i, (alpha, beta, tx, ty) in enumerate(track):
            d = (tx - x) ** 2 + (ty - y) ** 2
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    best_idx = last_wp
    best_dist = float("inf")
    for offset in range(-max_behind, max_ahead + 1):
        idx = (last_wp + offset) % N
        tx, ty = track[idx][2:4]
        d = (tx - x) ** 2 + (ty - y) ** 2
        if d < best_dist:
            best_dist = d
            best_idx = idx

    # If car is extremely far off track (> 22m), fallback to global search
    if best_dist > 500.0:
        for i, (alpha, beta, tx, ty) in enumerate(track):
            d = (tx - x) ** 2 + (ty - y) ** 2
            if d < best_dist:
                best_dist = d
                best_idx = i

    return best_idx


def speed_controller(speed, steer, target_speed_straight=70.0, target_speed_turn=38.0, min_target_speed=26.0):
    """Calculate gas and brake based on speed and steering angle."""
    steer_mag = abs(steer)
    if steer_mag < 0.1:
        target_speed = target_speed_straight
    elif steer_mag > 0.4:
        target_speed = target_speed_turn
    else:
        blend = (steer_mag - 0.1) / 0.3
        target_speed = target_speed_straight * (1 - blend) + target_speed_turn * blend

    target_speed = max(min_target_speed, target_speed)

    if speed < target_speed:
        deficit = target_speed - speed
        gas = min(0.9, 0.4 + 0.05 * deficit)
        brake = 0.0
    else:
        excess = speed - target_speed
        gas = 0.0
        brake = min(0.7, 0.15 + 0.04 * excess) if excess > 2.0 else 0.0

    return float(gas), float(brake)


def compute_overtake_bias(cars, my_idx, track):
    """
    Look for cars ahead within overtaking range and apply a lateral
    steering bias to slipstream and overtake cleanly without colliding.
    """
    my_car = cars[my_idx]
    x, y, heading, speed = get_car_state(my_car)
    fwd = np.array([-math.sin(heading - math.pi / 2), math.cos(heading - math.pi / 2)])
    right = np.array([math.cos(heading - math.pi / 2), math.sin(heading - math.pi / 2)])

    steer_bias = 0.0
    throttle_mult = 1.0

    for j, other_car in enumerate(cars):
        if j == my_idx:
            continue
        ox, oy, ohead, ospeed = get_car_state(other_car)
        rel = np.array([ox - x, oy - y])
        d_fwd = np.dot(rel, fwd)
        d_lat = np.dot(rel, right)

        if 0.5 < d_fwd < 13.0 and abs(d_lat) < 3.2:
            # Pass on the open side (steer < 0 turns LEFT, steer > 0 turns RIGHT)
            if d_lat >= 0:
                steer_bias -= 0.26  # other car on right -> steer LEFT
            else:
                steer_bias += 0.26  # other car on left -> steer RIGHT

            if d_fwd < 4.0 and speed > ospeed:
                throttle_mult = min(throttle_mult, 0.72)

    return float(np.clip(steer_bias, -0.4, 0.4)), throttle_mult


class F1ScoreboardDrawer:
    """Renders authentic Formula 1 timing tower and HUD telemetry overlay."""

    def __init__(self):
        font_paths = [
            r"C:\Windows\Fonts\segoeuib.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
        ]
        reg_font_paths = [
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arial.ttf",
        ]

        bold_path = next((p for p in font_paths if os.path.exists(p)), None)
        reg_path = next((p for p in reg_font_paths if os.path.exists(p)), None)

        if bold_path and reg_path:
            self.font_h1 = ImageFont.truetype(bold_path, 13)
            self.font_bold = ImageFont.truetype(bold_path, 11)
            self.font_num = ImageFont.truetype(bold_path, 12)
            self.font_reg = ImageFont.truetype(reg_path, 11)
            self.font_sub = ImageFont.truetype(reg_path, 9)
        else:
            self.font_h1 = ImageFont.load_default()
            self.font_bold = ImageFont.load_default()
            self.font_num = ImageFont.load_default()
            self.font_reg = ImageFont.load_default()
            self.font_sub = ImageFont.load_default()

    def draw_overlay(self, base_image, race_state, compact=False):
        """Draw F1 timing tower on the top-left of base_image."""
        overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        x0, y0 = 10, 10
        w = 152
        h_head = 24
        row_h = 24
        gap = 2

        current_lap = race_state["display_lap"]
        total_laps = race_state["total_laps"]
        status_text = race_state["status_text"]
        status_color = race_state["status_color"]

        # ── Header Bar ───────────────────────────────────────────────
        draw.rounded_rectangle([x0, y0, x0 + w, y0 + h_head], radius=4,
                               fill=(16, 18, 22, 235), outline=(50, 55, 65, 200), width=1)
        draw.rounded_rectangle([x0 + 3, y0 + 3, x0 + 26, y0 + h_head - 3], radius=2,
                               fill=(225, 6, 0, 255))
        draw.text((x0 + 6, y0 + 4), "F1", font=self.font_h1, fill=(255, 255, 255))
        draw.text((x0 + 33, y0 + 4), f"LAP {current_lap}/{total_laps}", font=self.font_h1, fill=(255, 255, 255))
        pill_w = 40
        draw.rounded_rectangle([x0 + w - pill_w - 4, y0 + 4, x0 + w - 4, y0 + h_head - 4], radius=2,
                               fill=status_color)
        draw.text((x0 + w - pill_w, y0 + 5), status_text, font=self.font_sub, fill=(255, 255, 255))

        # ── Driver Rows ──────────────────────────────────────────────
        cur_y = y0 + h_head + gap
        drivers = race_state["standings"]
        fastest_car_idx = race_state["fastest_lap_car"]

        for rank, d in enumerate(drivers, start=1):
            draw.rounded_rectangle([x0, cur_y, x0 + w, cur_y + row_h], radius=3,
                                   fill=(16, 18, 22, 225), outline=(45, 50, 60, 180), width=1)
            draw.text((x0 + 6, cur_y + 4), str(rank), font=self.font_num, fill=(250, 250, 250))
            draw.rectangle([x0 + 19, cur_y + 3, x0 + 23, cur_y + row_h - 3], fill=d["color"] + (255,))
            draw.text((x0 + 28, cur_y + 4), d["tag"], font=self.font_bold, fill=(255, 255, 255))
            gap_txt = d["gap"]
            gap_col = (255, 215, 0) if gap_txt == "LEADER" else (220, 225, 235)
            draw.text((x0 + 67, cur_y + 5), gap_txt, font=self.font_reg, fill=gap_col)

            if d["idx"] == fastest_car_idx:
                draw.rounded_rectangle([x0 + w - 24, cur_y + 5, x0 + w - 5, cur_y + row_h - 5], radius=2,
                                       fill=(160, 32, 240, 255))
                draw.text((x0 + w - 20, cur_y + 5), "FL", font=self.font_sub, fill=(255, 255, 255))

            cur_y += row_h + gap

        # ── Telemetry & Focus Box ─────────────────────────────────────
        if not compact:
            cur_y += 3
            draw.rounded_rectangle([x0, cur_y, x0 + w, cur_y + 44], radius=4,
                                   fill=(16, 18, 22, 230), outline=(45, 50, 60, 180), width=1)
            focus_name = race_state["focus_name"]
            speed_kmh = race_state["focus_speed_kmh"]
            gas_pct = race_state["focus_gas"]
            brake_pct = race_state["focus_brake"]

            draw.text((x0 + 8, cur_y + 4), f"FOCUS: {focus_name}", font=self.font_sub, fill=(180, 185, 195))
            draw.text((x0 + 8, cur_y + 19), f"{speed_kmh:3.0f} KM/H", font=self.font_bold, fill=(255, 255, 255))

            draw.text((x0 + 92, cur_y + 4), "THR", font=self.font_sub, fill=(0, 220, 80))
            draw.rectangle([x0 + 114, cur_y + 6, x0 + 144, cur_y + 11], fill=(40, 40, 50, 255))
            thr_w = int(30 * max(0.0, min(1.0, gas_pct)))
            if thr_w > 0:
                draw.rectangle([x0 + 114, cur_y + 6, x0 + 114 + thr_w, cur_y + 11], fill=(0, 220, 80, 255))

            draw.text((x0 + 92, cur_y + 18), "BRK", font=self.font_sub, fill=(240, 50, 50))
            draw.rectangle([x0 + 114, cur_y + 20, x0 + 144, cur_y + 25], fill=(40, 40, 50, 255))
            brk_w = int(30 * max(0.0, min(1.0, brake_pct)))
            if brk_w > 0:
                draw.rectangle([x0 + 114, cur_y + 20, x0 + 114 + brk_w, cur_y + 25], fill=(240, 50, 50, 255))

        # Winner Banner when race is completed
        if race_state["winner_name"]:
            winner_y = y0 + 10
            win_w = 260
            win_x = (base_image.size[0] - win_w) // 2
            draw.rounded_rectangle([win_x, winner_y, win_x + win_w, winner_y + 36], radius=5,
                                   fill=(15, 15, 20, 240), outline=(255, 215, 0, 255), width=2)
            draw.text((win_x + 14, winner_y + 8), f"WINNER: {race_state['winner_name']}!",
                      font=self.font_h1, fill=(255, 215, 0))

        return Image.alpha_composite(base_image.convert("RGBA"), overlay).convert("RGB")


def render_car_view(uw, target_car, cars, zoom_factor, out_w, out_h, surf):
    """Render the camera frame focused on target_car."""
    angle_cam = -target_car.hull.angle
    zoom_cam = zoom_factor * car_racing.SCALE
    sc_x = -(target_car.hull.position[0]) * zoom_cam
    sc_y = -(target_car.hull.position[1]) * zoom_cam
    trans_cam = pygame.math.Vector2((sc_x, sc_y)).rotate_rad(angle_cam)
    trans_cam = (out_w / 2 + trans_cam[0], out_h / 2 + trans_cam[1])

    uw.surf = surf
    uw._render_road(zoom_cam, trans_cam, angle_cam)
    for c in cars:
        c.draw(uw.surf, zoom_cam, trans_cam, angle_cam, True)

    return uw._create_image_array(uw.surf, (out_w, out_h))


def run_race(laps=10, track_seed=1000, video_dir="videos", zoom_factor=2.05, generate_multiview=True):
    """
    Execute a full 10-lap multi-car race with 4 autonomous cars, Box2D collisions,
    overtaking AI, F1 scoreboard overlay, multi-camera streams, and telemetry export.
    """
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("=" * 70)
    print(f"[RACE] STARTING FORMULA 1 AUTONOMOUS GRAND PRIX - {laps} LAPS")
    print(f"   Track Seed: {track_seed}")
    print(f"   Video Directory: {video_dir}")
    print("=" * 70, flush=True)

    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95)
    env.reset(seed=track_seed)
    uw = env.unwrapped
    track = uw.track
    N_TRACK = len(track)
    print(f"Track generated with {N_TRACK} waypoints.", flush=True)

    setup_checkered_finish_line(env)

    driver_configs = [
        {"name": "DAGGER", "tag": "DAG", "color": (235, 25, 25), "type": "model",
         "model_path": "models/dagger_s0.keras", "v_straight": 71.5, "v_turn": 39.5, "file": "race_dagger.mp4"},
        {"name": "AUGMENTED", "tag": "AUG", "color": (25, 140, 245), "type": "model",
         "model_path": "models/aug_s0.keras", "v_straight": 70.0, "v_turn": 38.5, "file": "race_augmented.mp4"},
        {"name": "PLAIN", "tag": "PLN", "color": (35, 200, 60), "type": "model",
         "model_path": "models/plain_s0.keras", "v_straight": 67.5, "v_turn": 37.0, "file": "race_plain.mp4"},
        {"name": "EXPERT", "tag": "EXP", "color": (245, 195, 20), "type": "expert",
         "model_path": None, "v_straight": 71.0, "v_turn": 39.0, "file": "race_expert.mp4"},
    ]

    models = []
    for cfg in driver_configs:
        if cfg["type"] == "model":
            m = keras.models.load_model(cfg["model_path"])
            models.append(m)
            print(f"Loaded {cfg['name']} model from {cfg['model_path']}", flush=True)
        else:
            models.append(None)
            print(f"Loaded {cfg['name']} pure-pursuit geometry driver", flush=True)

    # Starting grid slots
    grid_slots = [
        (-2, -2.2),  # P1: pole position (Red / DAGGER)
        (-4, +2.2),  # P2: outside grid slot (Blue / AUG)
        (-6, -2.2),  # P3: inside row 2 (Green / PLN)
        (-8, +2.2),  # P4: outside row 2 (Yellow / EXP)
    ]

    cars = []
    for i, (k, lat) in enumerate(grid_slots):
        beta, x, y = track[k % N_TRACK][1:4]
        gx = x + lat * math.cos(beta)
        gy = y + lat * math.sin(beta)
        if i == 0:
            c = uw.car
            c.hull.position = (gx, gy)
            c.hull.angle = beta
        else:
            c = car_dynamics.Car(uw.world, beta, gx, gy)
        c.hull.color = [channel / 255.0 for channel in driver_configs[i]["color"]]
        cars.append(c)

    # Video Writers setup
    main_video_path = os.path.join(video_dir, "race_10laps_f1.mp4")
    writers = {
        "broadcast": imageio.get_writer(main_video_path, fps=int(FPS), codec="libx264", quality=8),
    }

    if generate_multiview:
        for i, cfg in enumerate(driver_configs):
            p = os.path.join(video_dir, cfg["file"])
            writers[cfg["name"]] = imageio.get_writer(p, fps=int(FPS), codec="libx264", quality=7)
        quad_path = os.path.join(video_dir, "race_quad.mp4")
        writers["quad"] = imageio.get_writer(quad_path, fps=int(FPS), codec="libx264", quality=7)

    print(f"[RECORDING] Initialized {len(writers)} video streams...", flush=True)

    # State tracking
    laps_completed = [0] * 4
    lap_start_steps = [0] * 4
    has_started = [False] * 4
    s1_passed = [False] * 4
    s2_passed = [False] * 4
    s3_passed = [False] * 4
    s1_idx = int(N_TRACK * 0.28)  # Sector 1 Checkpoint (~82)
    s2_idx = int(N_TRACK * 0.58)  # Sector 2 Checkpoint (~170)
    s3_idx = int(N_TRACK * 0.85)  # Sector 3 Checkpoint (~249)
    last_lap_times = [None] * 4
    best_lap_times = [float("inf")] * 4
    lap_history = [[] for _ in range(4)]
    waypoint_progress = [0] * 4
    prev_waypoint = [0] * 4

    for i, c in enumerate(cars):
        gx, gy = c.hull.position
        initial_wp = find_nearest_waypoint_local(track, gx, gy, None)
        waypoint_progress[i] = initial_wp
        prev_waypoint[i] = initial_wp

    fastest_lap_time = float("inf")
    fastest_lap_car = None
    race_winner = None
    cooldown_steps = 0
    MAX_COOLDOWN = 350

    scoreboard = F1ScoreboardDrawer()
    surf_video = pygame.Surface((VIDEO_W, VIDEO_H))
    surf_state = pygame.Surface((STATE_W, STATE_H))
    surf_quad_tile = pygame.Surface((304, 200))

    # Telemetry export data structure
    telemetry_data = {
        "total_laps": laps,
        "track_seed": track_seed,
        "num_cars": 4,
        "track_waypoints": [[float(tx), float(ty)] for (_, _, tx, ty) in track],
        "finish_line": [float(track[0][2]), float(track[0][3])],
        "driver_info": driver_configs,
        "frames": [],
        "final_classification": [],
    }

    step_count = 0
    t_start = time.time()
    GRID_STAND_STEPS = 50

    while True:
        step_count += 1
        is_grid_start = step_count < GRID_STAND_STEPS

        # ── Steer and Accelerate Each Car ─────────────────────────────
        actions = []
        for i, c in enumerate(cars):
            cfg = driver_configs[i]
            x, y, heading, speed = get_car_state(c)
            nearest_wp = find_nearest_waypoint_local(track, x, y, last_wp=prev_waypoint[i])
            waypoint_progress[i] = nearest_wp

            if is_grid_start:
                steer, gas, brake = 0.0, 0.0, 1.0
                actions.append((steer, gas, brake))
                continue

            target_wp = (nearest_wp + 5) % N_TRACK
            tx, ty = track[target_wp][2:4]
            desired_angle = math.atan2(ty - y, tx - x)
            angle_err = (desired_angle - heading + math.pi) % (2 * math.pi) - math.pi

            n_wheels_on = sum(1 for w in c.wheels if len(w.tiles) > 0)
            is_offroad = n_wheels_on == 0
            ang_vel = c.hull.angularVelocity

            if is_offroad:
                # Anti-spin damping if spinning on slippery grass
                if abs(ang_vel) > 1.4:
                    steer = float(np.clip(1.2 * ang_vel, -0.6, 0.6))
                    gas = 0.1
                    brake = 0.25
                else:
                    # Target 5 waypoints ahead along track flow to merge smoothly
                    rejoin_wp = (nearest_wp + 5) % N_TRACK
                    rx, ry = track[rejoin_wp][2:4]
                    rejoin_angle = math.atan2(ry - y, rx - x)
                    rejoin_err = (rejoin_angle - heading + math.pi) % (2 * math.pi) - math.pi
                    steer = float(np.clip(-1.8 * rejoin_err, -0.65, 0.65))
                    gas = 0.55 if abs(rejoin_err) < 1.0 else 0.30
                    brake = 0.0
            elif abs(angle_err) > 1.25 or cfg["type"] == "expert":
                steer = float(np.clip(-2.8 * angle_err, -1.0, 1.0))
                ot_steer, throttle_mult = compute_overtake_bias(cars, i, track)
                steer = float(np.clip(steer + ot_steer, -1.0, 1.0))
                gas, brake = speed_controller(speed, steer, cfg["v_straight"], cfg["v_turn"])
                gas *= throttle_mult
            else:
                # Neural model onboard camera inference
                angle_cam = -c.hull.angle
                zoom_cam = car_racing.ZOOM * car_racing.SCALE
                sc_x = -(c.hull.position[0]) * zoom_cam
                sc_y = -(c.hull.position[1]) * zoom_cam
                trans_cam = pygame.math.Vector2((sc_x, sc_y)).rotate_rad(angle_cam)
                trans_cam = (STATE_W / 2 + trans_cam[0], STATE_H / 4 + trans_cam[1])

                uw.surf = surf_state
                uw._render_road(zoom_cam, trans_cam, angle_cam)
                for other_c in cars:
                    other_c.draw(uw.surf, zoom_cam, trans_cam, angle_cam, False)
                obs = uw._create_image_array(uw.surf, (STATE_W, STATE_H))
                cropped = obs[:84, :, :].astype(np.float32)[np.newaxis]

                with torch.no_grad():
                    pred = float(models[i](cropped, training=False).cpu().numpy()[0, 0])
                    model_steer = float(np.clip(pred, -1.0, 1.0))

                geom_steer = float(np.clip(-2.8 * angle_err, -1.0, 1.0))
                if abs(angle_err) > 0.42:
                    steer = geom_steer
                else:
                    steer = 0.82 * model_steer + 0.18 * geom_steer

                ot_steer, throttle_mult = compute_overtake_bias(cars, i, track)
                steer = float(np.clip(steer + ot_steer, -1.0, 1.0))
                gas, brake = speed_controller(speed, steer, cfg["v_straight"], cfg["v_turn"])
                gas *= throttle_mult

            actions.append((steer, gas, brake))

        # Advance Box2D physics for all cars
        for i, c in enumerate(cars):
            s, g, b = actions[i]
            c.steer(-s)
            c.gas(g)
            c.brake(b)
            c.step(1.0 / FPS)

        uw.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        uw.t += 1.0 / FPS

        # ── Deterministic Lap Counting & Sector Logic ─────────────────
        for i in range(4):
            wp = waypoint_progress[i]
            prev_wp = prev_waypoint[i]

            if not has_started[i]:
                # Start Lap 1 when leaving grid stand across start/finish
                if step_count > GRID_STAND_STEPS and (wp < 25 or prev_wp > N_TRACK - 10):
                    has_started[i] = True
                    lap_start_steps[i] = step_count
                    s1_passed[i] = False
                    s2_passed[i] = False
                    s3_passed[i] = False
                    print(f"[RACE] Car {i} ({driver_configs[i]['name']}) launched - Lap 1 started!", flush=True)
            else:
                # Progressively mark Sector 1, 2, 3
                if abs(wp - s1_idx) < 22:
                    s1_passed[i] = True
                if s1_passed[i] and abs(wp - s2_idx) < 22:
                    s2_passed[i] = True
                if s1_passed[i] and s2_passed[i] and abs(wp - s3_idx) < 25:
                    s3_passed[i] = True

                # Finish line crossing:
                # Must have passed all 3 sectors sequentially AND crossed waypoint 0 wrap-around
                crossed_line = (prev_wp >= N_TRACK - 12 and wp <= 10) or (wp <= 5 and prev_wp > s3_idx)
                lap_elapsed = (step_count - lap_start_steps[i]) / FPS

                if s1_passed[i] and s2_passed[i] and s3_passed[i] and crossed_line and lap_elapsed >= 12.0:
                    laps_completed[i] += 1
                    lap_time = lap_elapsed
                    lap_start_steps[i] = step_count
                    s1_passed[i] = False
                    s2_passed[i] = False
                    s3_passed[i] = False
                    last_lap_times[i] = lap_time
                    best_lap_times[i] = min(best_lap_times[i], lap_time)
                    lap_history[i].append(lap_time)

                    if lap_time < fastest_lap_time:
                        fastest_lap_time = lap_time
                        fastest_lap_car = i

                    print(f"[LAP] Lap {laps_completed[i]}/{laps} | Car {i} ({driver_configs[i]['name']}): "
                          f"{lap_time:.2f}s (Best: {best_lap_times[i]:.2f}s)", flush=True)

                    if laps_completed[i] >= laps and race_winner is None:
                        race_winner = driver_configs[i]["name"]
                        print(f"[WINNER] CHEQUERED FLAG! {race_winner} WINS THE {laps}-LAP GRAND PRIX!", flush=True)

            prev_waypoint[i] = wp


        # ── Dynamic Standings & Timing Calculations ───────────────────
        scores = []
        for i in range(4):
            tot = (laps_completed[i] * N_TRACK + waypoint_progress[i]) if has_started[i] else -(i * 10)
            scores.append((tot, i))
        scores.sort(reverse=True)

        leader_tot = scores[0][0]
        standings = []
        for rank, (tot, car_i) in enumerate(scores):
            cfg = driver_configs[car_i]
            if rank == 0:
                gap_str = "LEADER"
            else:
                dist_gap = leader_tot - tot
                time_gap = dist_gap / max(25.0, driver_configs[car_i]["v_straight"] * 0.35)
                gap_str = f"+{time_gap:.2f}s"

            standings.append({
                "idx": car_i,
                "name": cfg["name"],
                "tag": cfg["tag"],
                "color": cfg["color"],
                "gap": gap_str,
            })

        max_lap = max(laps_completed) + 1
        display_lap = min(laps, max_lap)
        if race_winner:
            status_text = "FINISH"
            status_color = (255, 180, 0, 240)
        elif display_lap == laps:
            status_text = "FINAL"
            status_color = (230, 20, 20, 240)
        elif is_grid_start:
            status_text = "GRID"
            status_color = (220, 20, 20, 240)
        else:
            status_text = "LIVE"
            status_color = (0, 180, 50, 230)

        # Telemetry sample (sample every 5 frames = 10Hz)
        if step_count % 5 == 0:
            cars_telemetry = []
            for i, c in enumerate(cars):
                x, y, head, spd = get_car_state(c)
                cars_telemetry.append({
                    "idx": i,
                    "name": driver_configs[i]["name"],
                    "x": round(float(x), 2),
                    "y": round(float(y), 2),
                    "speed_kmh": round(float(spd * 3.6), 1),
                    "gas": round(float(actions[i][1]), 2),
                    "brake": round(float(actions[i][2]), 2),
                    "lap": laps_completed[i],
                    "waypoint": waypoint_progress[i],
                })

            telemetry_data["frames"].append({
                "step": step_count,
                "time_sec": round(step_count / FPS, 2),
                "lap": display_lap,
                "status": status_text,
                "winner": race_winner,
                "standings": [{"idx": s["idx"], "gap": s["gap"]} for s in standings],
                "cars": cars_telemetry,
            })

        # ── Multi-Camera Rendering Pipeline ───────────────────────────
        # 1. Leader / Broadcast Cam (follows race leader or Car 0 in battle)
        leader_idx = scores[0][1]
        car_frames = {}

        # Render all 4 individual car views
        for i, c in enumerate(cars):
            arr = render_car_view(uw, c, cars, zoom_factor, VIDEO_W, VIDEO_H, surf_video)
            car_frames[i] = arr

        # Prepare race state dict for HUD overlay
        def make_race_state(focused_car_idx):
            fc = cars[focused_car_idx]
            _, _, _, spd = get_car_state(fc)
            return {
                "display_lap": display_lap,
                "total_laps": laps,
                "status_text": status_text,
                "status_color": status_color,
                "standings": standings,
                "fastest_lap_car": fastest_lap_car,
                "focus_name": driver_configs[focused_car_idx]["name"],
                "focus_speed_kmh": spd * 3.6,
                "focus_gas": actions[focused_car_idx][1],
                "focus_brake": actions[focused_car_idx][2],
                "winner_name": race_winner,
            }

        # Main Broadcast Frame (Follows leader)
        state_broad = make_race_state(leader_idx)
        pil_broad = Image.fromarray(car_frames[leader_idx])
        out_broad = scoreboard.draw_overlay(pil_broad, state_broad, compact=False)
        writers["broadcast"].append_data(np.array(out_broad))

        # Individual Onboard Views
        if generate_multiview:
            for i, cfg in enumerate(driver_configs):
                state_i = make_race_state(i)
                pil_i = Image.fromarray(car_frames[i])
                out_i = scoreboard.draw_overlay(pil_i, state_i, compact=False)
                writers[cfg["name"]].append_data(np.array(out_i))

            # Quad-View (2x2 Multi-Cam Grid)
            quad_img = Image.new("RGB", (VIDEO_W, VIDEO_H), (10, 10, 14))
            draw_quad = ImageDraw.Draw(quad_img)
            positions = [(0, 0), (304, 0), (0, 200), (304, 200)]
            for i, (qx, qy) in enumerate(positions):
                sub_img = Image.fromarray(car_frames[i]).resize((304, 200), Image.Resampling.BILINEAR)
                quad_img.paste(sub_img, (qx, qy))
                cfg = driver_configs[i]
                draw_quad.rectangle([qx + 4, qy + 4, qx + 60, qy + 20], fill=(15, 15, 20, 220))
                draw_quad.rectangle([qx + 4, qy + 4, qx + 8, qy + 20], fill=cfg["color"])
                draw_quad.text((qx + 12, qy + 6), cfg["name"], font=scoreboard.font_sub, fill=(255, 255, 255))

            # Overlay compact timing tower on top of Quad-View
            quad_overlaid = scoreboard.draw_overlay(quad_img, state_broad, compact=True)
            writers["quad"].append_data(np.array(quad_overlaid))

        # Check race finish condition
        if race_winner is not None:
            cooldown_steps += 1
            if cooldown_steps >= MAX_COOLDOWN or all(l >= laps for l in laps_completed):
                print("[FINISH] Race concluded! Completing victory cool-down roll.", flush=True)
                break

        # Progress reporting
        if step_count % 500 == 0:
            elapsed = time.time() - t_start
            p1_name = standings[0]["name"]
            print(f"Step {step_count:5d} | Lap {display_lap}/{laps} | P1: {p1_name} | "
                  f"Leader Laps: {laps_completed[leader_idx]}/{laps} | {step_count / elapsed:5.1f} FPS", flush=True)

    for name, w in writers.items():
        w.close()

    elapsed_total = time.time() - t_start
    print("=" * 70, flush=True)
    print(f"[COMPLETED] 10-LAP GRAND PRIX COMPLETED IN {elapsed_total:.1f}s ({step_count} frames)", flush=True)
    print(f"[OUTPUT] Primary broadcast video: {main_video_path}", flush=True)
    print("=" * 70, flush=True)

    # Compile Final Classification
    classification = []
    print("\nFINAL CLASSIFICATION:")
    print(f"{'POS':<4} {'DRIVER':<12} {'LAPS':<6} {'BEST LAP':<10} {'STATUS'}")
    for rank, d in enumerate(standings, start=1):
        idx = d["idx"]
        best_str = f"{best_lap_times[idx]:.2f}s" if best_lap_times[idx] < float("inf") else "--"
        fl_flag = " (FL)" if idx == fastest_lap_car else ""
        print(f"P{rank:<3} {d['name']:<12} {laps_completed[idx]:<6} {best_str + fl_flag:<10} FINISHED", flush=True)
        classification.append({
            "pos": rank,
            "name": d["name"],
            "tag": d["tag"],
            "laps": laps_completed[idx],
            "best_lap": round(best_lap_times[idx], 2) if best_lap_times[idx] < float("inf") else None,
            "is_fastest": idx == fastest_lap_car,
            "lap_history": [round(lt, 2) for lt in lap_history[idx]],
        })

    telemetry_data["final_classification"] = classification
    telemetry_path = "data/race_telemetry.json"
    with open(telemetry_path, "w", encoding="utf-8") as f:
        json.dump(telemetry_data, f, indent=2)
    print(f"[OUTPUT] Synchronized race telemetry saved to: {telemetry_path}", flush=True)

    return main_video_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-car 10-lap race with F1 Scoreboard & Multi-Cam Views")
    parser.add_argument("--laps", type=int, default=10, help="Number of race laps (default: 10)")
    parser.add_argument("--seed", type=int, default=1000, help="Track seed (default: 1000)")
    parser.add_argument("--video_dir", type=str, default="videos", help="Output videos directory")
    parser.add_argument("--zoom", type=float, default=2.05, help="Camera zoom factor (default: 2.05)")
    parser.add_argument("--no_multiview", action="store_true", help="Only render broadcast video")
    args = parser.parse_args()

    run_race(laps=args.laps, track_seed=args.seed, video_dir=args.video_dir,
             zoom_factor=args.zoom, generate_multiview=not args.no_multiview)
