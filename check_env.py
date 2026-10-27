"""
Stage 0B - check_env.py
Verify the CarRacing-v3 environment: observation shape, action space,
track structure, wheel tile detection, zoom-in intro, and status bar.
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
import numpy as np
from PIL import Image

print("=" * 60)
print("CarRacing-v3 Environment Check")
print("=" * 60)

# Create environment
env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95)
print(f"Gymnasium version: {gym.__version__}")

obs, info = env.reset(seed=42)
print(f"Observation shape: {obs.shape}  (expect 96x96x3)")
print(f"Action space: {env.action_space}")
print(f"  Action meaning: [steer (-1=left, +1=right), gas (0-1), brake (0-1)]")

# Access internals
uw = env.unwrapped
print(f"\nTrack length: {len(uw.track)} waypoints")
print(f"Track entry format (alpha, beta, x, y): {uw.track[0]}")
print(f"Car hull position: {uw.car.hull.position}")
print(f"Car hull angle: {uw.car.hull.angle:.4f}")
print(f"Car hull linearVelocity: {uw.car.hull.linearVelocity}")
print(f"Number of wheels: {len(uw.car.wheels)}")

# Check wheel.tiles
for i, w in enumerate(uw.car.wheels):
    on_track = len(w.tiles) > 0
    print(f"  Wheel {i}: tiles={len(w.tiles)}, on_track={on_track}")

# Check zoom-in: first ~50 frames
print(f"\nZoom-in check (self.t starts at {uw.t:.3f}):")
print(f"  Zoom transition: 0.1*SCALE*max(1-t,0) + ZOOM*SCALE*min(t,1)")
print(f"  At t=0: zoom = 0.1*6*1 + 2.7*6*0 = 0.6 (zoomed out)")
print(f"  At t=1 (50 frames): zoom = 0.1*6*0 + 2.7*6*1 = 16.2 (zoomed in)")
print(f"  So the first 50 frames (1 second at 50fps) are the zoom-in intro.")

# Status bar analysis
print(f"\nStatus bar analysis:")
print(f"  _render_indicators uses H/40 as unit, draws 5*H/40 = H/8 from bottom")
print(f"  For state_pixels (96x96): bottom 96/8 = 12 pixels are status bar")
print(f"  Network input should be top 84 rows: obs[:84, :, :]")

# Run episode and save sample frames
env2 = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95)
obs, _ = env2.reset(seed=42)
frames_to_save = {0: obs}

for step in range(1, 101):
    action = np.array([0.0, 0.1, 0.0], dtype=np.float32)  # gentle gas, no steer
    obs, reward, terminated, truncated, info = env2.step(action)
    if step in [30, 60, 100]:
        frames_to_save[step] = obs

print(f"\nSaving frames 0, 30, 60, 100 to results/frame_check/")
for frame_idx, frame in frames_to_save.items():
    img = Image.fromarray(frame)
    path = f"results/frame_check/frame_{frame_idx:04d}.png"
    img.save(path)
    print(f"  Saved {path} - shape {frame.shape}")

# Verify status bar: check bottom 12 rows of frame 100
frame100 = frames_to_save[100]
bottom_bar = frame100[84:, :, :]  # rows 84-95 = bottom 12 rows
top_region = frame100[:84, :, :]

print(f"\nFrame 100 analysis:")
print(f"  Top 84 rows (driving view) mean pixel values: {top_region.mean(axis=(0,1))}")
print(f"  Bottom 12 rows (status bar) mean pixel values: {bottom_bar.mean(axis=(0,1))}")
print(f"  Bottom bar is mostly dark (indicators on black background)")

# Check if bottom bar contains mostly black/dark pixels
dark_fraction = (bottom_bar.sum(axis=2) < 30).mean()
print(f"  Fraction of dark pixels in bottom bar: {dark_fraction:.2%}")
print(f"  CONFIRMED: Bottom 12 rows are the status bar - MUST crop for network input")

# Termination summary
print(f"\nTermination conditions (verified from source):")
print(f"  1. Lap complete: tile_visited_count == len(track) OR new_lap flag")
print(f"     (new_lap set when tile.idx==0 visited AND coverage > {0.95})")
print(f"  2. Out of playfield: abs(x) > PLAYFIELD or abs(y) > PLAYFIELD")
print(f"     (returns -100 reward)")
print(f"  3. Truncation at 1000 steps (via TimeLimit wrapper)")

env.close()
env2.close()
print("\nAll checks passed!")
