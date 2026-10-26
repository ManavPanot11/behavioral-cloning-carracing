import os, math
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d import car_racing

orig_contact = car_racing.FrictionDetector.BeginContact
def fixed_BeginContact(self, contact):
    orig_contact(self, contact)
    u1 = contact.fixtureA.body.userData
    u2 = contact.fixtureB.body.userData
    tile = u1 if u1 and 'road_friction' in u1.__dict__ else (u2 if u2 and 'road_friction' in u2.__dict__ else None)
    if tile and hasattr(tile, 'idx') and tile.idx == 0:
        if self.env.tile_visited_count / len(self.env.track) > self.lap_complete_percent:
            self.env.new_lap = True
car_racing.FrictionDetector.BeginContact = fixed_BeginContact

def norm(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

laps = 0
scores = []
offroads = []

for seed in range(5000, 5010):
    env = gym.make('CarRacing-v3', render_mode='rgb_array', lap_complete_percent=0.95)
    obs, _ = env.reset(seed=seed)
    track = env.unwrapped.track
    offroad_count = 0
    was_on = True
    
    for step in range(1000):
        hull = env.unwrapped.car.hull
        x, y = hull.position
        car_heading = hull.angle + math.pi / 2
        dists = [(tx-x)**2 + (ty-y)**2 for _, _, tx, ty in track]
        nearest = np.argmin(dists)
        target_idx = (nearest + 5) % len(track)
        _, _, tx, ty = track[target_idx]
        desired_angle = math.atan2(ty - y, tx - x)
        error = norm(desired_angle - car_heading)
        steer = float(np.clip(-2.8 * error, -1.0, 1.0))
        vx, vy = hull.linearVelocity
        speed = math.sqrt(vx**2 + vy**2)
        
        steer_abs = abs(steer)
        if steer_abs < 0.1:
            target = 75.0
        elif steer_abs < 0.4:
            target = 44.0 + (75.0 - 44.0) * (1.0 - steer_abs / 0.4)
        else:
            target = max(30.0, 44.0 * (1.0 - 0.4 * (steer_abs - 0.4)))
            
        if step < 50:
            steer = 0.0
            gas, brake = 0.8, 0.0
        else:
            if speed < target - 1.0:
                gas = min(1.0, 0.6 + 0.05 * (target - speed))
                brake = 0.0
            elif speed > target + 2.0:
                gas = 0.0
                brake = min(0.8, 0.2 + 0.03 * (speed - target))
            else:
                gas = 0.35
                brake = 0.0
                
        obs, r, term, trunc, info = env.step(np.array([steer, gas, brake], dtype=np.float32))
        
        on_track = sum(1 for w in env.unwrapped.car.wheels if len(w.tiles) > 0) > 0
        if not on_track and was_on:
            offroad_count += 1
        was_on = on_track
        
        if term:
            laps += 1
            scores.append(env.unwrapped.reward)
            offroads.append(offroad_count)
            print(f"Seed {seed}: LAP COMPLETED at step {step}! score={env.unwrapped.reward:.1f}, offroads={offroad_count}, visited={env.unwrapped.tile_visited_count}/{len(track)} ({env.unwrapped.tile_visited_count/len(track)*100:.1f}%)", flush=True)
            break
        if trunc:
            scores.append(env.unwrapped.reward)
            offroads.append(offroad_count)
            print(f"Seed {seed}: TRUNCATED at step {step}! score={env.unwrapped.reward:.1f}, offroads={offroad_count}, visited={env.unwrapped.tile_visited_count}/{len(track)} ({env.unwrapped.tile_visited_count/len(track)*100:.1f}%)", flush=True)
            break
    env.close()

print(f"\n==========================================")
print(f"Total laps completed: {laps}/10")
print(f"Mean score: {np.mean(scores):.1f}")
print(f"Total offroad events: {sum(offroads)}")
print(f"==========================================")
