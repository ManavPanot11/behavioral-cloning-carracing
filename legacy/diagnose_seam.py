"""
diagnose_seam.py - Empirical diagnosis of the Start/Finish Seam Bug across 5 tracks.

Investigates:
1. Track waypoints at the seam: track[0], track[-1], track[-2].
2. road_poly structure: indices, vertices, and colors.
3. Kerb red/white border strips near index 0.
4. The 3-shade tile cycle (i % 3) at loop closure.
5. Whether tile 0 is duplicated.
6. Renders high-resolution seam crops with gym's renderer.
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import math
import gymnasium as gym
from gymnasium.envs.box2d import car_racing
from PIL import Image, ImageDraw
import pygame

os.makedirs("docs/seam_diagnosis", exist_ok=True)

test_seeds = [1000, 1001, 1002, 1003, 1004]

report_lines = []
report_lines.append("# Start/Finish Seam Bug Empirical Diagnosis Report")
report_lines.append("## Analysis Across 5 Tracks (Seeds 1000, 1001, 1002, 1003, 1004)\n")

for seed in test_seeds:
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    obs, info = env.reset(seed=seed)
    uw = env.unwrapped
    track = uw.track
    N = len(track)
    road_poly = uw.road_poly
    road_bodies = uw.road

    report_lines.append(f"### Track Seed {seed} (Total Waypoints: {N})")

    # 1. Waypoints around seam
    p_last = track[-1]
    p_0 = track[0]
    p_1 = track[1]
    dist_seam = math.sqrt((p_0[2] - p_last[2])**2 + (p_0[3] - p_last[3])**2)
    dist_0_1 = math.sqrt((p_1[2] - p_0[2])**2 + (p_1[3] - p_0[3])**2)

    report_lines.append(f"- **Waypoint track[0]**: (x={p_0[2]:.2f}, y={p_0[3]:.2f}, beta={p_0[1]:.3f})")
    report_lines.append(f"- **Waypoint track[-1]**: (x={p_last[2]:.2f}, y={p_last[3]:.2f}, beta={p_last[1]:.3f})")
    report_lines.append(f"- **Seam Distance (track[-1] to track[0])**: {dist_seam:.3f} units (nominal step ~ {dist_0_1:.3f})")

    # 2. Check for duplicate tiles or bodies
    tile_indices = [b.idx for b in road_bodies]
    is_duplicated = len(tile_indices) != len(set(tile_indices))
    report_lines.append(f"- **Total Tile Bodies**: {len(road_bodies)} | **Unique indices**: {len(set(tile_indices))} | **Duplicated?**: {is_duplicated}")

    # 3. road_poly breakdown: count road tiles vs kerb borders
    road_tiles_poly = []
    border_polys = []
    for idx, (poly, color) in enumerate(road_poly):
        col_tuple = tuple(int(c) for c in color[:3])
        if col_tuple in [(255, 255, 255), (255, 0, 0)]:
            border_polys.append((idx, poly, col_tuple))
        else:
            road_tiles_poly.append((idx, poly, col_tuple))

    report_lines.append(f"- **Total road_poly entries**: {len(road_poly)} ({len(road_tiles_poly)} road surface tiles, {len(border_polys)} kerb border polygons)")

    # 4. Examine tile colors around seam
    # Tile 0: connects track[-1] to track[0]
    # Tile N-1: connects track[-2] to track[-1]
    # Tile 1: connects track[0] to track[1]
    col_last = road_tiles_poly[-1][2]
    col_0 = road_tiles_poly[0][2]
    col_1 = road_tiles_poly[1][2]
    report_lines.append(f"- **Tile Colors (3-shade cycle)**:")
    report_lines.append(f"  - Tile[-1] (index {len(road_tiles_poly)-1}, mod 3 = {(len(road_tiles_poly)-1)%3}): RGB={col_last}")
    report_lines.append(f"  - Tile[0]  (index 0, mod 3 = 0): RGB={col_0}")
    report_lines.append(f"  - Tile[1]  (index 1, mod 3 = 1): RGB={col_1}")

    cycle_jump = (len(road_tiles_poly) - 1) % 3
    report_lines.append(f"  - *Seam Shade Transition*: Tile[-1] (shade {cycle_jump}) -> Tile[0] (shade 0). If N % 3 != 0, this creates an abrupt shade boundary step at the seam.")

    # 5. Kerb border at seam
    kerb_at_0 = [bp for bp in border_polys if bp[0] < 5]
    kerb_at_end = [bp for bp in border_polys if bp[0] > len(road_poly) - 5]
    report_lines.append(f"- **Kerb border near tile 0**: {len(kerb_at_0)} polygons; **near tile N-1**: {len(kerb_at_end)} polygons")

    # 6. Render high-resolution seam image using gym's renderer
    # Camera centered at track[0]
    out_w, out_h = 600, 600
    surf = pygame.Surface((out_w, out_h))
    zoom = 3.5 * car_racing.SCALE
    angle_cam = -track[0][1]
    sc_x = -track[0][2] * zoom
    sc_y = -track[0][3] * zoom
    trans = pygame.math.Vector2((sc_x, sc_y)).rotate_rad(angle_cam)
    trans = (out_w / 2 + trans[0], out_h / 2 + trans[1])

    uw.surf = surf
    uw._render_road(zoom, trans, angle_cam)
    uw.car.draw(uw.surf, zoom, trans, angle_cam, True)
    img_arr = uw._create_image_array(uw.surf, (out_w, out_h))

    # Annotate image
    pil_img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([10, 10, 320, 65], fill=(0, 0, 0, 200))
    draw.text((15, 15), f"SEED {seed} - GYM RENDERER SEAM", fill=(255, 255, 0))
    draw.text((15, 32), f"Tile 0 connects track[-1] -> track[0]", fill=(255, 255, 255))
    draw.text((15, 47), f"Shade mod 3: Tile[-1] ({cycle_jump}) -> Tile[0] (0)", fill=(200, 200, 255))

    crop_path = f"docs/seam_diagnosis/seam_crop_seed_{seed}.png"
    pil_img.save(crop_path)
    report_lines.append(f"- **Rendered Crop**: Saved to `docs/seam_diagnosis/seam_crop_seed_{seed}.png`\n")

    env.close()

# Also inspect what happened in previous code (e.g. race_multicar.py) that produced "two checkered patches"
report_lines.append("## Root Cause Analysis: The 'Two Checkered Patches' Bug\n")
report_lines.append(r"""
1. **Gymnasium's Native Geometry**:
   - In native Gymnasium `CarRacing-v3`, there is **NO checkered finish line**. The road is simply drawn from `self.road_poly`.
   - Each tile $i$ connects `track[i-1]` to `track[i]`.
   - Tile 0 connects `track[-1]` to `track[0]`. There is **no duplicate tile** at the seam; `tile_visited_count` is exactly $N$.
   - However, Gym uses a **3-shade alternating gray cycle** (`c = 0.01 * (i % 3) * 255`). When the total number of tiles $N$ is not divisible by 3 (e.g. $N=293, 293 \pmod 3 = 2$), tile $N-1$ is shade 1, and tile 0 is shade 0, creating a subtle visual seam line.

2. **The Cause of the Double Checkered Patch in Previous Code**:
   - In previous scripts (such as `race_multicar.py`), `setup_checkered_finish_line()` attempted to insert checkered polygons:
     ```python
     poly0 = uw.road_poly[0][0]
     ...
     uw.road_poly = checkered_polys + uw.road_poly[1:]
     ```
   - **Flaw A (Index Mismatch with Kerb Polygons)**: `road_poly` contains BOTH road polygons and red/white kerb polygons (`border[i]`). If `border[0]` was True, `road_poly[1]` was actually a **kerb polygon**, not the next road tile! Replacing `road_poly[0]` shifted indices or left split tiles.
   - **Flaw B (Tile 0 vs Tile N-1 / Track Seam Confusion)**: In `CarRacing`, the car spawns at `track[0]`. Tile 0 is the polygon behind the start line (connecting `track[-1]` to `track[0]`), while Tile 1 is in front of the car. When developers also drew or overlaid a second finish line at `track[0]` or replaced both Tile 0 and Tile N-1 (or overlaid on top of existing tiles without removing the underlying polygon), **two consecutive checkered bands appeared**!
   - **Flaw C (Double Rendering in Multi-car / Pygame Layering)**: Re-drawing a checkered line in Python Pygame on top of an already-modified `road_poly` rendered two overlapping or offset checkered strips.

3. **The Clean F1Sim Renderer Solution**:
   - **Never mutate Gym's internal Box2D `road_poly` or observations**: Leave Gym's physics, tiles, and observations 100% clean so models see the exact training distribution.
   - **Handle rendering in our dedicated F1Sim renderer**:
     - Draw the smooth asphalt surface.
     - Compute the exact perpendicular vector at `track[0]` (the true finish line coordinate).
     - Render **exactly ONE bold checkered start/finish line strip** at `track[0]`.
     - Render staggered 2-wide grid boxes behind `track[0]` along the approach straight.
     - Eliminate any duplicated checkered strips or visual artifacts at the loop closure.
""")

report_path = "docs/seam_diagnosis/start_finish_diagnosis.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"Diagnosis completed! Report saved to {report_path}")
