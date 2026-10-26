# Start/Finish Seam Bug Empirical Diagnosis Report
## Analysis Across 5 Tracks (Seeds 1000, 1001, 1002, 1003, 1004)

### Track Seed 1000 (Total Waypoints: 293)
- **Waypoint track[0]**: (x=70.48, y=-16.84, beta=-1.461)
- **Waypoint track[-1]**: (x=63.54, y=-17.73, beta=-1.269)
- **Seam Distance (track[-1] to track[0])**: 6.999 units (nominal step ~ 3.500)
- **Total Tile Bodies**: 293 | **Unique indices**: 293 | **Duplicated?**: False
- **Total road_poly entries**: 357 (293 road surface tiles, 64 kerb border polygons)
- **Tile Colors (3-shade cycle)**:
  - Tile[-1] (index 292, mod 3 = 1): RGB=(104, 104, 104)
  - Tile[0]  (index 0, mod 3 = 0): RGB=(102, 102, 102)
  - Tile[1]  (index 1, mod 3 = 1): RGB=(102, 102, 102)
  - *Seam Shade Transition*: Tile[-1] (shade 1) -> Tile[0] (shade 0). If N % 3 != 0, this creates an abrupt shade boundary step at the seam.
- **Kerb border near tile 0**: 1 polygons; **near tile N-1**: 2 polygons
- **Rendered Crop**: Saved to `docs/seam_diagnosis/seam_crop_seed_1000.png`

### Track Seed 1001 (Total Waypoints: 312)
- **Waypoint track[0]**: (x=215.91, y=-54.08, beta=-0.167)
- **Waypoint track[-1]**: (x=214.74, y=-60.98, beta=-0.167)
- **Seam Distance (track[-1] to track[0])**: 7.000 units (nominal step ~ 3.500)
- **Total Tile Bodies**: 312 | **Unique indices**: 312 | **Duplicated?**: False
- **Total road_poly entries**: 362 (312 road surface tiles, 50 kerb border polygons)
- **Tile Colors (3-shade cycle)**:
  - Tile[-1] (index 311, mod 3 = 2): RGB=(107, 107, 107)
  - Tile[0]  (index 0, mod 3 = 0): RGB=(102, 102, 102)
  - Tile[1]  (index 1, mod 3 = 1): RGB=(102, 102, 102)
  - *Seam Shade Transition*: Tile[-1] (shade 2) -> Tile[0] (shade 0). If N % 3 != 0, this creates an abrupt shade boundary step at the seam.
- **Kerb border near tile 0**: 0 polygons; **near tile N-1**: 0 polygons
- **Rendered Crop**: Saved to `docs/seam_diagnosis/seam_crop_seed_1001.png`

### Track Seed 1002 (Total Waypoints: 275)
- **Waypoint track[0]**: (x=218.59, y=-52.82, beta=-0.122)
- **Waypoint track[-1]**: (x=217.74, y=-59.77, beta=-0.122)
- **Seam Distance (track[-1] to track[0])**: 7.000 units (nominal step ~ 3.500)
- **Total Tile Bodies**: 275 | **Unique indices**: 275 | **Duplicated?**: False
- **Total road_poly entries**: 360 (275 road surface tiles, 85 kerb border polygons)
- **Tile Colors (3-shade cycle)**:
  - Tile[-1] (index 274, mod 3 = 1): RGB=(104, 104, 104)
  - Tile[0]  (index 0, mod 3 = 0): RGB=(102, 102, 102)
  - Tile[1]  (index 1, mod 3 = 1): RGB=(102, 102, 102)
  - *Seam Shade Transition*: Tile[-1] (shade 1) -> Tile[0] (shade 0). If N % 3 != 0, this creates an abrupt shade boundary step at the seam.
- **Kerb border near tile 0**: 0 polygons; **near tile N-1**: 0 polygons
- **Rendered Crop**: Saved to `docs/seam_diagnosis/seam_crop_seed_1002.png`

### Track Seed 1003 (Total Waypoints: 300)
- **Waypoint track[0]**: (x=207.28, y=-49.19, beta=-0.347)
- **Waypoint track[-1]**: (x=204.90, y=-55.77, beta=-0.347)
- **Seam Distance (track[-1] to track[0])**: 7.000 units (nominal step ~ 3.500)
- **Total Tile Bodies**: 300 | **Unique indices**: 300 | **Duplicated?**: False
- **Total road_poly entries**: 362 (300 road surface tiles, 62 kerb border polygons)
- **Tile Colors (3-shade cycle)**:
  - Tile[-1] (index 299, mod 3 = 2): RGB=(107, 107, 107)
  - Tile[0]  (index 0, mod 3 = 0): RGB=(102, 102, 102)
  - Tile[1]  (index 1, mod 3 = 1): RGB=(102, 102, 102)
  - *Seam Shade Transition*: Tile[-1] (shade 2) -> Tile[0] (shade 0). If N % 3 != 0, this creates an abrupt shade boundary step at the seam.
- **Kerb border near tile 0**: 0 polygons; **near tile N-1**: 0 polygons
- **Rendered Crop**: Saved to `docs/seam_diagnosis/seam_crop_seed_1003.png`

### Track Seed 1004 (Total Waypoints: 298)
- **Waypoint track[0]**: (x=135.22, y=-32.78, beta=-1.220)
- **Waypoint track[-1]**: (x=128.65, y=-35.19, beta=-1.220)
- **Seam Distance (track[-1] to track[0])**: 7.000 units (nominal step ~ 3.500)
- **Total Tile Bodies**: 298 | **Unique indices**: 298 | **Duplicated?**: False
- **Total road_poly entries**: 363 (298 road surface tiles, 65 kerb border polygons)
- **Tile Colors (3-shade cycle)**:
  - Tile[-1] (index 297, mod 3 = 0): RGB=(102, 102, 102)
  - Tile[0]  (index 0, mod 3 = 0): RGB=(102, 102, 102)
  - Tile[1]  (index 1, mod 3 = 1): RGB=(102, 102, 102)
  - *Seam Shade Transition*: Tile[-1] (shade 0) -> Tile[0] (shade 0). If N % 3 != 0, this creates an abrupt shade boundary step at the seam.
- **Kerb border near tile 0**: 0 polygons; **near tile N-1**: 0 polygons
- **Rendered Crop**: Saved to `docs/seam_diagnosis/seam_crop_seed_1004.png`

## Root Cause Analysis: The 'Two Checkered Patches' Bug


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
