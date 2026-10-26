import gymnasium as gym, numpy as np
from expert import (expert_steer, speed_controller, get_car_state,
                    LOOKAHEAD, GAIN)

def inspect_track(seed=1000):
    env = gym.make("CarRacing-v3", render_mode="rgb_array", max_episode_steps=2000)
    env.reset(seed=seed)
    uw = env.unwrapped
    waypoints = np.array([[p[2], p[3]] for p in uw.track], dtype=np.float32)
    total_tiles = len(uw.track)

    triggers = []
    for frac in [0.25, 0.50, 0.75]:
        twp = int(frac * total_tiles)
        rng = np.random.RandomState(seed * 1000 + twp)
        sign = float(rng.choice([-1.0, 1.0]))
        triggers.append({"target_wp": twp, "offset": sign * 0.6, "triggered": False, "step": -1})

    for step in range(2000):
        car_x, car_y, heading, speed = get_car_state(env)
        dists = np.linalg.norm(waypoints - np.array([car_x, car_y]), axis=1)
        nearest = int(np.argmin(dists))
        cdist = dists[nearest]

        offset = 0.0
        if step >= 50:
            for tr in triggers:
                if not tr["triggered"] and nearest >= tr["target_wp"]:
                    tr["triggered"] = True
                    tr["step"] = step
                    print(f"Triggered target_wp={tr['target_wp']} at step {step}, offset={tr['offset']:+.1f}, dist={cdist:.2f}")
                    break

        for tr in triggers:
            if tr["triggered"] and tr["step"] <= step < tr["step"] + 20:
                offset = tr["offset"]
                frame_in_dist = step - tr["step"]
                if frame_in_dist in [0, 5, 10, 15, 19]:
                    print(f"   dist frame {frame_in_dist:2d} (step {step}): raw_s={raw_s:+.2f}, exec_s={np.clip(raw_s+offset, -1, 1):+.2f}, dist={cdist:.2f}")

        raw_s = expert_steer(env, lookahead=LOOKAHEAD, gain=GAIN) if step >= 50 else 0.0
        exec_s = float(np.clip(raw_s + offset, -1.0, 1.0))
        gas, brake = speed_controller(env, exec_s)
        obs, r, term, trunc, info = env.step(np.array([exec_s, gas, brake], dtype=np.float32))
        if term or trunc:
            break
    env.close()

if __name__ == "__main__":
    inspect_track(1000)
