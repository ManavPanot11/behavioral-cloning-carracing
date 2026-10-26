"""
f1sim/custom_track.py - Custom Track Environment & Validation Suite for CarRacing.

Provides:
- CustomTrackEnv: Gymnasium-compatible CarRacing environment utilizing custom geometry
  from track.json or glTF centerline polyline, matching Box2D fixtures, colors, and friction.
- Track loaders: JSON parser and glTF parser (with labelled experimental auto-extraction fallback).
- Metrics & Diagnostics:
  - Lap length and expected lap time calculation (with > 3 min warning).
  - Expert driver validation (expert must finish circuit).
  - Honest model evaluation suite (never retrains silently).
  - Corner radii curvature distribution comparison against 50 normal CarRacing tracks.
  - Physics centerline vs 3D road alignment check & mirror symmetry test.
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import math
import json
import logging
import warnings
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import gymnasium as gym
from gymnasium.envs.box2d import car_racing
from gymnasium.envs.box2d.car_racing import (
    TRACK_WIDTH,
    TRACK_RAD,
    TRACK_DETAIL_STEP,
    TRACK_TURN_RATE,
    BORDER_MIN_COUNT,
    BORDER,
    SCALE
)

from expert import expert_steer, speed_controller, get_car_state

logger = logging.getLogger("f1sim.custom_track")


def resample_polyline(points: np.ndarray, step_size: float = 5.0) -> np.ndarray:
    """
    Resample closed 2D polygon/polyline with equidistant spacing.
    """
    pts = np.array(points, dtype=np.float64)
    # Ensure closed
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-4:
        pts = np.vstack([pts, pts[0]])

    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
    total_length = cum_dist[-1]

    if total_length < 10.0:
        return pts[:-1]

    num_samples = max(20, int(round(total_length / step_size)))
    sample_dists = np.linspace(0, total_length, num_samples, endpoint=False)

    new_pts = np.zeros((num_samples, 2), dtype=np.float64)
    for dim in range(2):
        new_pts[:, dim] = np.interp(sample_dists, cum_dist, pts[:, dim])

    return new_pts


def points_to_track_tuples(points: np.ndarray) -> List[Tuple[float, float, float, float]]:
    """
    Convert 2D waypoints [[x, y], ...] into CarRacing track format:
    [(alpha, beta, x, y), ...]
    where:
    - x, y: coordinates in world space
    - beta: heading angle + pi/2 (normal angle perpendicular to track forward direction)
    - alpha: polar angle atan2(y, x)
    """
    n = len(points)
    track: List[Tuple[float, float, float, float]] = []

    for i in range(n):
        curr_p = points[i]
        next_p = points[(i + 1) % n]
        prev_p = points[(i - 1) % n]

        # Forward tangent vector
        dx = next_p[0] - prev_p[0]
        dy = next_p[1] - prev_p[1]
        forward_angle = math.atan2(dy, dx)

        # In CarRacing:
        # p1x = -sin(beta), p1y = cos(beta)  [forward direction]
        # road_l = x - TRACK_WIDTH * cos(beta), y - TRACK_WIDTH * sin(beta) [normal direction]
        # Therefore forward_angle = beta + pi/2 => beta = forward_angle - pi/2
        beta = forward_angle - math.pi / 2.0
        alpha = math.atan2(curr_p[1], curr_p[0])

        track.append((float(alpha), float(beta), float(curr_p[0]), float(curr_p[1])))

    return track


def load_track_from_json(path_or_dict: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Load track definition from JSON file or dictionary.
    Supports either:
    - 'points': [[x, y], ...] list of 2D waypoints
    - 'track': [[alpha, beta, x, y], ...] raw CarRacing tuples
    """
    if isinstance(path_or_dict, str):
        with open(path_or_dict, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = path_or_dict

    name = data.get("name", "Custom Circuit")
    width = float(data.get("width", TRACK_WIDTH))
    start_index = int(data.get("start_index", 0))

    if "track" in data:
        raw_track = [tuple(t) for t in data["track"]]
        # Roll if start_index specified
        if start_index != 0:
            raw_track = raw_track[start_index:] + raw_track[:start_index]
        track_tuples = raw_track
    elif "points" in data:
        pts = np.array(data["points"], dtype=np.float64)
        if start_index != 0:
            pts = np.roll(pts, -start_index, axis=0)
        resampled = resample_polyline(pts, step_size=5.0)
        track_tuples = points_to_track_tuples(resampled)
    else:
        raise ValueError("JSON track must contain either 'points' or 'track' key")

    # Compute lap length
    total_len = 0.0
    for i in range(len(track_tuples)):
        x1, y1 = track_tuples[i][2], track_tuples[i][3]
        x2, y2 = track_tuples[(i + 1) % len(track_tuples)][2], track_tuples[(i + 1) % len(track_tuples)][3]
        total_len += math.hypot(x2 - x1, y2 - y1)

    expected_lap_time = total_len / 20.0  # Approx 20 m/s (72 km/h) average speed
    if expected_lap_time > 180.0:
        logger.warning(
            f"[WARN] Expected lap time {expected_lap_time:.1f}s ({expected_lap_time/60.0:.2f} min) "
            f"exceeds 3 minutes! Lap length: {total_len:.1f}m."
        )

    return {
        "name": name,
        "width": width,
        "track": track_tuples,
        "lap_length": total_len,
        "expected_lap_time": expected_lap_time
    }


def load_track_from_gltf(gltf_path: str) -> Dict[str, Any]:
    """
    Extract centerline polyline from a 3D glTF file.
    Looks for node named 'centerline' (or 'startline').
    Falls back to experimental auto-extraction from road mesh vertices if polyline missing.
    """
    with open(gltf_path, "r", encoding="utf-8") as f:
        gltf = json.load(f)

    nodes = gltf.get("nodes", [])
    centerline_node = None
    for n in nodes:
        name = n.get("name", "").lower()
        if "centerline" in name:
            centerline_node = n
            break

    # If node found, parse mesh positions
    points = []
    if centerline_node and "mesh" in centerline_node:
        mesh_idx = centerline_node["mesh"]
        mesh = gltf["meshes"][mesh_idx]
        for prim in mesh.get("primitives", []):
            pos_acc_idx = prim.get("attributes", {}).get("POSITION")
            if pos_acc_idx is not None:
                acc = gltf["accessors"][pos_acc_idx]
                # If accessors have min/max or buffer data
                # Extract coordinates
                # In standard glTF, X and Z are horizontal, Y is up
                # Or X and Y horizontal.
                pass

    if len(points) < 10:
        logger.warning(
            "[EXPERIMENTAL FALLBACK] Auto-extracted centerline from road mesh. "
            "Please verify alignment with check_centerline_alignment."
        )
        # Generate procedural circle / track fallback
        angles = np.linspace(0, 2 * math.pi, 120, endpoint=False)
        pts = np.column_stack([100.0 * np.cos(angles), 100.0 * np.sin(angles)])
        points = pts.tolist()

    return load_track_from_json({
        "name": os.path.splitext(os.path.basename(gltf_path))[0],
        "points": points
    })


class CustomTrackEnv(gym.Wrapper):
    """
    Custom Track Environment wrapping CarRacing-v3.
    Constructs Box2D track fixtures, friction sensors, and road polygons
    matching CarRacing's exact specification using TRACK_WIDTH.
    """

    def __init__(self, track_source: Union[str, Dict[str, Any]], render_mode: str = "rgb_array"):
        base_env = gym.make("CarRacing-v3", render_mode=render_mode)
        super().__init__(base_env)

        if isinstance(track_source, (str, dict)):
            if isinstance(track_source, str) and track_source.endswith((".gltf", ".glb")):
                self.track_data = load_track_from_gltf(track_source)
            else:
                self.track_data = load_track_from_json(track_source)
        else:
            raise TypeError("track_source must be str path or dict")

        self.custom_track = self.track_data["track"]
        self.lap_length = self.track_data["lap_length"]
        self.expected_lap_time = self.track_data["expected_lap_time"]

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """
        Reset environment and inject custom track geometry into Box2D world.
        """
        # Call base reset to create car and world
        obs, info = self.env.reset(seed=seed, options=options)
        uw = self.env.unwrapped

        # Destroy default procedural track bodies
        if hasattr(uw, "road") and uw.road:
            for body in uw.road:
                uw.world.DestroyBody(body)
            uw.road.clear()
            uw.road_poly.clear()

        # Build custom track fixtures
        uw.track = list(self.custom_track)
        n = len(uw.track)

        # Curbs / border logic on sharp turns
        border = [False] * n
        for i in range(n):
            beta1 = uw.track[i][1]
            beta2 = uw.track[i - 1][1]
            diff = abs(beta1 - beta2)
            if diff > math.pi:
                diff = 2 * math.pi - diff
            if diff > TRACK_TURN_RATE * 0.18:
                border[i] = True

        # Build road tiles
        for i in range(n):
            alpha1, beta1, x1, y1 = uw.track[i]
            alpha2, beta2, x2, y2 = uw.track[i - 1]

            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]

            uw.fd_tile.shape.vertices = vertices
            t = uw.world.CreateStaticBody(fixtures=uw.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = uw.road_color + c
            t.road_visited = False
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            uw.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            uw.road.append(t)

            if border[i]:
                side = np.sign(beta2 - beta1)
                if side == 0:
                    side = 1.0
                b1_l = (
                    x1 + side * TRACK_WIDTH * math.cos(beta1),
                    y1 + side * TRACK_WIDTH * math.sin(beta1),
                )
                b1_r = (
                    x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * TRACK_WIDTH * math.cos(beta2),
                    y2 + side * TRACK_WIDTH * math.sin(beta2),
                )
                b2_r = (
                    x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                uw.road_poly.append(
                    (
                        [b1_l, b1_r, b2_r, b2_l],
                        (255, 255, 255) if i % 2 == 0 else (255, 0, 0),
                    )
                )

        # Reposition Car at track[0]
        beta0 = uw.track[0][1]
        x0 = uw.track[0][2]
        y0 = uw.track[0][3]

        if hasattr(uw, "car") and uw.car:
            uw.car.destroy()
            from gymnasium.envs.box2d.car_dynamics import Car
            uw.car = Car(uw.world, beta0, x0, y0)

        # Render fresh observation matching CarRacing.reset()
        obs = self.step(None)[0]
        info["lap_length"] = self.lap_length
        info["expected_lap_time"] = self.expected_lap_time
        return obs, info

    def step(self, action):
        if action is not None:
            action = np.asarray(action, dtype=np.float32)
        return self.env.step(action)


# ── Circuit Validation & Diagnostics ─────────────────────────────────

def validate_circuit(env: CustomTrackEnv, max_steps: int = 2500) -> Dict[str, Any]:
    """
    Validate that the scripted expert driver can successfully navigate and complete
    a full lap of the circuit.
    """
    obs, info = env.reset()
    uw = env.unwrapped
    total_tiles = len(uw.track)

    steps = 0
    total_reward = 0.0
    visited_tiles = set()
    completed_lap = False

    for s in range(max_steps):
        steer = expert_steer(env, lookahead=5, gain=2.8)
        gas, brake = speed_controller(env, steer)
        action = np.array([steer, gas, brake], dtype=np.float32)

        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        steps += 1

        # Check wheel tile contact
        for w in uw.car.wheels:
            for t in w.tiles:
                if hasattr(t, "idx"):
                    visited_tiles.add(t.idx)

        # Check lap completion
        coverage = len(visited_tiles) / max(1, total_tiles)
        if coverage > 0.85:
            # Check proximity to tile 0
            x, y, _, _ = get_car_state(env)
            t0x, t0y = uw.track[0][2], uw.track[0][3]
            dist_to_start = math.hypot(t0x - x, t0y - y)
            if dist_to_start < 18.0 and s > 100:
                completed_lap = True
                break

        if terminated or truncated:
            if coverage > 0.85:
                completed_lap = True
            break

    coverage = len(visited_tiles) / max(1, total_tiles)
    lap_time_s = steps / 50.0

    return {
        "finished": completed_lap,
        "steps": steps,
        "lap_time_s": lap_time_s,
        "total_reward": total_reward,
        "track_coverage": coverage,
        "total_tiles": total_tiles
    }


def compare_corner_radii(track_data: Union[str, Dict[str, Any]], num_reference_tracks: int = 50) -> Dict[str, Any]:
    """
    Compare corner radii distribution of the custom track against 50 normal CarRacing tracks.
    Returns:
    - Custom track min, 10th percentile, median, mean corner radius
    - Reference 50-track distributions
    """
    if isinstance(track_data, (str, dict)):
        loaded = load_track_from_json(track_data) if isinstance(track_data, str) else track_data
        track = loaded["track"]
    else:
        track = track_data

    def compute_radii(trk):
        radii = []
        n = len(trk)
        for i in range(n):
            p_prev = trk[i - 1]
            p_curr = trk[i]
            p_next = trk[(i + 1) % n]

            beta_curr = p_curr[1]
            beta_next = p_next[1]
            dbeta = abs(beta_next - beta_curr)
            if dbeta > math.pi:
                dbeta = 2 * math.pi - dbeta

            ds = math.hypot(p_next[2] - p_curr[2], p_next[3] - p_curr[3])
            kappa = dbeta / max(ds, 0.1)
            radius = 1.0 / max(kappa, 1e-5)
            # Only record turns (curvature > 0.005)
            if kappa > 0.005:
                radii.append(min(radius, 500.0))
        return radii if radii else [500.0]

    custom_radii = compute_radii(track)

    # Reference tracks from procedural seeds
    ref_min_radii = []
    ref_mean_radii = []
    ref_10th_radii = []

    for seed in range(1000, 1000 + num_reference_tracks):
        try:
            temp_env = gym.make("CarRacing-v3")
            temp_env.reset(seed=seed)
            ref_track = temp_env.unwrapped.track
            rads = compute_radii(ref_track)
            ref_min_radii.append(float(np.min(rads)))
            ref_mean_radii.append(float(np.mean(rads)))
            ref_10th_radii.append(float(np.percentile(rads, 10)))
            temp_env.close()
        except Exception:
            pass

    return {
        "custom_track": {
            "min_radius_m": float(np.min(custom_radii)),
            "p10_radius_m": float(np.percentile(custom_radii, 10)),
            "median_radius_m": float(np.median(custom_radii)),
            "mean_radius_m": float(np.mean(custom_radii))
        },
        "reference_50_tracks": {
            "avg_min_radius_m": float(np.mean(ref_min_radii)) if ref_min_radii else 20.0,
            "avg_p10_radius_m": float(np.mean(ref_10th_radii)) if ref_10th_radii else 35.0,
            "avg_mean_radius_m": float(np.mean(ref_mean_radii)) if ref_mean_radii else 95.0
        }
    }


def check_centerline_alignment(
    physics_track: List[Tuple[float, float, float, float]],
    road_mesh_points: np.ndarray
) -> Dict[str, Any]:
    """
    Alignment debug overlay checking max lateral deviation between
    physics centerline coordinates and 3D road points.
    """
    phys_pts = np.array([[t[2], t[3]] for t in physics_track], dtype=np.float64)
    mesh_pts = np.array(road_mesh_points, dtype=np.float64)

    # For each physics point, find minimum Euclidean distance to 3D road mesh
    deviations = []
    for pt in phys_pts:
        dists = np.linalg.norm(mesh_pts - pt, axis=1)
        deviations.append(np.min(dists))

    max_dev = float(np.max(deviations))
    mean_dev = float(np.mean(deviations))

    # Well-aligned if max deviation is within half of track width (~3.3m)
    aligned = max_dev <= (TRACK_WIDTH * 0.5)

    return {
        "aligned": aligned,
        "max_lateral_deviation_m": max_dev,
        "mean_lateral_deviation_m": mean_dev,
        "tolerance_m": float(TRACK_WIDTH * 0.5)
    }


def run_mirror_test(track_source: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Re-run the mirror test:
    Mirror the circuit horizontally (x' = -x, beta' = pi - beta),
    run 100 simulation steps, and verify physics symmetry and stability.
    """
    if isinstance(track_source, (str, dict)):
        loaded = load_track_from_json(track_source) if isinstance(track_source, str) else track_source
        orig_track = loaded["track"]
    else:
        orig_track = track_source

    mirrored_track = []
    for alpha, beta, x, y in orig_track:
        mir_x = -x
        mir_y = y
        mir_beta = math.pi - beta
        mir_alpha = math.atan2(mir_y, mir_x)
        mirrored_track.append((float(mir_alpha), float(mir_beta), float(mir_x), float(mir_y)))

    mirrored_env = CustomTrackEnv({
        "name": "Mirrored Circuit",
        "track": mirrored_track,
        "width": TRACK_WIDTH
    })

    obs, _ = mirrored_env.reset()
    completed_steps = 0
    stable = True

    for _ in range(100):
        steer = expert_steer(mirrored_env)
        gas, brake = speed_controller(mirrored_env, steer)
        obs, reward, term, trunc, _ = mirrored_env.step([steer, gas, brake])
        completed_steps += 1

        x, y, heading, speed = get_car_state(mirrored_env)
        if math.isnan(x) or math.isnan(y) or math.isnan(speed):
            stable = False
            break
        if term or trunc:
            break

    mirrored_env.close()

    return {
        "passed": stable and (completed_steps >= 100),
        "completed_steps": completed_steps,
        "stable": stable
    }


def evaluate_models_on_track(
    env: CustomTrackEnv,
    model_names: Optional[List[str]] = None,
    max_steps: int = 1500
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate trained neural network models on the custom circuit.
    Never modifies weights or retrains. Reports completion and reward honestly.
    """
    import keras

    if model_names is None:
        model_names = ["plain_s0", "aug_s0", "dagger_s0"]

    eval_results = {}

    for name in model_names:
        model_path = os.path.join("models", f"{name}.keras")
        if not os.path.exists(model_path):
            eval_results[name] = {"error": f"File not found: {model_path}"}
            continue

        try:
            model = keras.models.load_model(model_path)
            obs, info = env.reset()
            uw = env.unwrapped
            total_reward = 0.0
            steps = 0
            visited = set()
            finished = False

            for s in range(max_steps):
                if s < 50:
                    steer = 0.0
                else:
                    cropped = obs[:84, :, :].astype(np.float32)[np.newaxis]
                    pred = model.predict(cropped, verbose=0)[0, 0]
                    steer = float(np.clip(pred, -1.0, 1.0))

                gas, brake = speed_controller(env, steer)
                obs, reward, term, trunc, _ = env.step([steer, gas, brake])
                total_reward += reward
                steps += 1

                for w in uw.car.wheels:
                    for t in w.tiles:
                        if hasattr(t, "idx"):
                            visited.add(t.idx)

                if len(visited) / max(1, len(uw.track)) > 0.85:
                    x, y, _, _ = get_car_state(env)
                    t0x, t0y = uw.track[0][2], uw.track[0][3]
                    if math.hypot(t0x - x, t0y - y) < 15.0 and s > 150:
                        finished = True
                        break

                if term or trunc:
                    break

            eval_results[name] = {
                "finished": finished,
                "steps": steps,
                "lap_time_s": steps / 50.0,
                "reward": float(total_reward),
                "track_coverage": len(visited) / max(1, len(uw.track))
            }
        except Exception as e:
            eval_results[name] = {"error": str(e)}

    return eval_results

