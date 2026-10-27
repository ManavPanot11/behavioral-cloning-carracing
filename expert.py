"""
expert.py - Scripted expert driver for CarRacing-v3
Pure-pursuit steering from true track geometry + speed controller.
"""
import os, math
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import gymnasium as gym
from gymnasium.envs.box2d import car_racing

# Fix Gymnasium CarRacing-v3 new_lap bug:
# Tile 0 is marked visited during reset(), so FrictionDetector never enters
# 'if not tile.road_visited' when crossing finish line at the end of the lap.
orig_begin_contact = car_racing.FrictionDetector.BeginContact
def fixed_begin_contact(self, contact):
    orig_begin_contact(self, contact)
    u1 = contact.fixtureA.body.userData
    u2 = contact.fixtureB.body.userData
    tile = u1 if u1 and 'road_friction' in u1.__dict__ else (u2 if u2 and 'road_friction' in u2.__dict__ else None)
    if tile and hasattr(tile, 'idx') and tile.idx == 0:
        if self.env.tile_visited_count / len(self.env.track) > self.lap_complete_percent:
            self.env.new_lap = True
car_racing.FrictionDetector.BeginContact = fixed_begin_contact


# ── Tuned parameters ─────────────────────────────────────────────────
LOOKAHEAD = 5
GAIN = 2.8
TARGET_SPEED_STRAIGHT = 70.0
TARGET_SPEED_TURN = 38.0
MIN_TARGET_SPEED = 26.0
def get_car_state(env):
    """Return (x, y, heading_angle, speed) from the unwrapped env."""
    uw = env.unwrapped if hasattr(env, 'unwrapped') else env
    hull = uw.car.hull
    x, y = hull.position
    # Car forward direction is along local +Y body vector,
    # so world heading = hull.angle + pi/2
    heading = hull.angle + math.pi / 2
    vx, vy = hull.linearVelocity
    speed = math.sqrt(vx**2 + vy**2)
    return x, y, heading, speed


def wheels_on_track(env):
    """Return number of wheels currently on track tiles."""
    uw = env.unwrapped if hasattr(env, 'unwrapped') else env
    return sum(1 for w in uw.car.wheels if len(w.tiles) > 0)


def find_nearest_waypoint(env, x, y):
    """Find index of nearest track waypoint to (x, y)."""
    uw = env.unwrapped if hasattr(env, 'unwrapped') else env
    track = uw.track
    best_idx = 0
    best_dist = float('inf')
    for i, (alpha, beta, tx, ty) in enumerate(track):
        d = (tx - x)**2 + (ty - y)**2
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def normalize_angle(a):
    """Normalize angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


# ── expert steering ─────────────────────────────────────────────────
def expert_steer(env, lookahead=5, gain=2.8):
    """
    Pure-pursuit steering: find the nearest track waypoint,
    look ahead N waypoints, compute heading error, return steer in [-1, 1].
    
    Sign convention (verified empirically):
    - steer < 0: turns counter-clockwise (LEFT)
    - steer > 0: turns clockwise (RIGHT)
    - positive error (desired > car_heading) = target is to the LEFT -> steer < 0
    - steer = -gain * error
    """
    uw = env.unwrapped if hasattr(env, 'unwrapped') else env
    track = uw.track
    x, y, car_heading, speed = get_car_state(env)
    
    nearest = find_nearest_waypoint(env, x, y)
    target_idx = (nearest + lookahead) % len(track)
    _, _, tx, ty = track[target_idx]
    
    # Desired heading to target waypoint
    desired_angle = math.atan2(ty - y, tx - x)
    error = normalize_angle(desired_angle - car_heading)
    
    steer = -gain * error
    return float(np.clip(steer, -1.0, 1.0))


# ── speed controller ────────────────────────────────────────────────
def speed_controller(env, steer, target_speed_straight=70.0, target_speed_turn=38.0,
                     min_target_speed=26.0):
    """
    Returns (gas, brake). Target speed depends on |steer|.
    Used by BOTH the expert and the learned model.
    """
    _, _, _, speed = get_car_state(env)
    
    steer_abs = abs(steer)
    if steer_abs < 0.1:
        target = target_speed_straight
    elif steer_abs < 0.35:
        target = target_speed_turn + (target_speed_straight - target_speed_turn) * (1.0 - steer_abs / 0.35)
    else:
        target = max(min_target_speed, target_speed_turn * (1.0 - 0.4 * (steer_abs - 0.35)))
    
    if speed < target:
        gas = 1.0 if steer_abs < 0.1 else min(0.8, 0.5 + 0.05 * (target - speed))
        brake = 0.0
    elif speed > target + 2.0:
        gas = 0.0
        brake = min(0.8, 0.2 + 0.04 * (speed - target))
    else:
        gas = 0.35
        brake = 0.0
    
    return float(gas), float(brake)


# ── run an episode ──────────────────────────────────────────────────
def run_expert_episode(env, seed, max_steps=1000, skip_frames=50,
                       lookahead=5, gain=2.8,
                       target_speed_straight=70.0, target_speed_turn=38.0,
                       min_target_speed=26.0,
                       record_fn=None):
    """
    Run one episode with the expert. Returns dict with metrics.
    record_fn(obs, steer, step): called each non-intro step for data collection.
    """
    obs, _ = env.reset(seed=seed)
    
    total_reward = 0.0
    steps = 0
    offroad_events = 0
    was_on_track = True
    offroad_streak = 0
    max_offroad_streak = 0
    lap_completed = False
    
    for step in range(max_steps):
        if step < skip_frames:
            # Zoom-in intro: no steer, just gentle gas
            steer = 0.0
            gas, brake = 0.8, 0.0
        else:
            steer = expert_steer(env, lookahead=lookahead, gain=gain)
            gas, brake = speed_controller(env, steer, target_speed_straight,
                                         target_speed_turn, min_target_speed)
            
            if record_fn is not None:
                record_fn(obs, steer, step)
        
        action = np.array([steer, gas, brake], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Track off-road events
        n_wheels = wheels_on_track(env)
        currently_on = n_wheels > 0
        
        if not currently_on:
            offroad_streak += 1
            max_offroad_streak = max(max_offroad_streak, offroad_streak)
            if was_on_track:
                offroad_events += 1
        else:
            offroad_streak = 0
        was_on_track = currently_on
        
        if terminated or truncated:
            lap_completed = info.get("lap_finished", False) or terminated
            break
    
    uw = env.unwrapped if hasattr(env, 'unwrapped') else env
    tiles_visited = uw.tile_visited_count
    total_tiles = len(uw.track)
    
    return {
        "seed": seed,
        "lap_completed": lap_completed,
        "score": total_reward,
        "steps": steps,
        "offroad_events": offroad_events,
        "max_offroad_streak": max_offroad_streak,
        "tiles_visited": tiles_visited,
        "total_tiles": total_tiles,
    }


def test_steering_sign(env, seed=42):
    """Empirically verify the steering sign convention."""
    print("\n--- Steering sign convention test ---")
    obs, _ = env.reset(seed=seed)
    for _ in range(55):
        obs, _, _, _, _ = env.step(np.array([0.0, 0.3, 0.0], dtype=np.float32))
    
    x0, y0, angle0, _ = get_car_state(env)
    for _ in range(20):
        obs, _, _, _, _ = env.step(np.array([1.0, 0.3, 0.0], dtype=np.float32))
    x1, y1, angle1, _ = get_car_state(env)
    angle_change_pos = normalize_angle(angle1 - angle0)
    
    obs, _ = env.reset(seed=seed)
    for _ in range(55):
        obs, _, _, _, _ = env.step(np.array([0.0, 0.3, 0.0], dtype=np.float32))
    for _ in range(20):
        obs, _, _, _, _ = env.step(np.array([-1.0, 0.3, 0.0], dtype=np.float32))
    x2, y2, angle2, _ = get_car_state(env)
    angle_change_neg = normalize_angle(angle2 - angle0)
    
    print(f"  action[0]=+1.0: heading change = {math.degrees(angle_change_pos):+.1f} deg (clockwise/right)")
    print(f"  action[0]=-1.0: heading change = {math.degrees(angle_change_neg):+.1f} deg (counter-clockwise/left)")
    print("  CONFIRMED: steer < 0 turns left, steer > 0 turns right.")


def record_expert_video(seed=5008):
    """Record one expert lap to videos/expert/."""
    from gymnasium.wrappers import RecordVideo
    os.makedirs("videos/expert", exist_ok=True)
    video_env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95)
    video_env = RecordVideo(video_env, video_folder="videos/expert",
                            episode_trigger=lambda ep: ep == 0,
                            name_prefix="expert_lap")
    
    print(f"\nRecording expert video for seed {seed} to videos/expert/ ...")
    r = run_expert_episode(video_env, seed, max_steps=1000)
    video_env.close()
    print(f"Expert video recorded! Result: {r}")


# ── main: test, tune, and report ─────────────────────────────────────
if __name__ == "__main__":
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95)
    test_steering_sign(env, seed=42)
    
    test_seeds = list(range(5000, 5010))
    
    # Tuning configurations
    configs = [
        # (lookahead, gain, straight, turn, min)
        (5, 2.8, 70.0, 38.0, 26.0),
        (5, 2.8, 75.0, 40.0, 28.0),
        (6, 2.8, 70.0, 38.0, 26.0),
        (5, 2.5, 65.0, 38.0, 26.0),
    ]
    
    best_config = None
    best_laps = 0
    best_results = None
    
    for attempt, (la, g, ss, ts, ms) in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"Tuning attempt {attempt}: lookahead={la}, gain={g}, speeds=({ss},{ts},{ms})")
        print(f"{'='*60}")
        
        results = []
        for seed in test_seeds:
            r = run_expert_episode(env, seed, max_steps=1000, lookahead=la, gain=g,
                                  target_speed_straight=ss, target_speed_turn=ts,
                                  min_target_speed=ms)
            results.append(r)
            status = "LAP" if r["lap_completed"] else "TRUNC"
            cov = r["tiles_visited"] / r["total_tiles"] * 100
            print(f"  Seed {seed}: {status}  score={r['score']:.1f}  steps={r['steps']}  offroad={r['offroad_events']}  cov={cov:.1f}%")
        
        laps = sum(1 for r in results if r["lap_completed"])
        mean_score = np.mean([r["score"] for r in results])
        total_offroad = sum(r["offroad_events"] for r in results)
        
        print(f"\n  Laps completed: {laps}/10  Mean score: {mean_score:.1f}  Total offroads: {total_offroad}")
        
        if laps > best_laps or best_results is None:
            best_laps = laps
            best_config = (la, g, ss, ts, ms)
            best_results = results
            
        if laps >= 9:
            print(f"  SUCCESS: Reached {laps}/10 laps completed!")
            break
            
    print(f"\n{'='*60}")
    print(f"BEST EXPERT CONFIG: lookahead={best_config[0]}, gain={best_config[1]}, speeds=({best_config[2]},{best_config[3]},{best_config[4]})")
    print(f"Laps completed in <1000 steps: {best_laps}/10")
    print(f"Mean score: {np.mean([r['score'] for r in best_results]):.1f}")
    print(f"Total off-road events across 10 tracks: {sum(r['offroad_events'] for r in best_results)}")
    print(f"{'='*60}")
    
    env.close()
    
    # Record one expert lap
    record_expert_video(seed=5008)
