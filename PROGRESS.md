# Project Progress Log: F1-Inspired Race Simulator Upgrade

Running log tracking all phases of the project upgrade (done / verified / blocked).

---

## Status Summary

| Phase | Description | Status | Verification Evidence / Gate Notes |
|:---:|:---|:---:|:---|
| **Phase 0** | Discovery & Safety Auditing | **DONE** | Tag `v0-before-f1` set, on `f1-premium`, no experiment running, report compiled. |
| **Phase 1** | Reliability (Tests, Doctor, Start/Finish Seam Diagnosis) | **DONE** | 33 pytest tests pass, doctor check passed, dev sanity test bit-identically reproduced 20/20 laps & 2.09m distance, seam diagnosis documented. |
| **Phase 2** | Simulation Core (`f1sim/`: env, track, drivers, timing, replay) | **DONE** | Windowed tracker matches Gym on dev tracks 1000–1004, ModelDriver input normalization fixed to float32 [0, 255] (matching Rescaling layer). |
| **Phase 3** | Stewards Rules Engine (Track limits, penalties, race control) | **DONE** | RulesConfig, warnings 1–2, black & white flag, 5s/10s penalties, 150-frame DNF, documented in `docs/rules.md`. |
| **Phase 4** | Live Dashboard (Pygame 1920x1080 canvas, telemetry, F1 UI) | **DONE** | 1080p broadcast canvas with responsive DPI scaling (720p, 1080p, 1440p) passing tests, preview verified in `docs/dashboard_preview_1080p.png`. |
| **Phase 5** | Sessions & Media (Qualifying, Race, Weekend, Hero video/GIF) | **DONE** | Multi-car Grand Prix race verified on seed 3003. Generated `videos/f1_showcase_3003.mp4` (9.69 MB, < 25 MB) and `videos/f1_showcase_3003.gif` (1.14 MB, < 5 MB). Telemetry saved to `data/telemetry_latest.json.gz`. |
| **Phase 6** | Premium Repository (README, docs/, changelog, requirements) | **DONE** | `docs/rules.md`, `docs/telemetry.md`, `docs/architecture.md`, `docs/seam_diagnosis/start_finish_diagnosis.md` complete, 34/34 unit tests pass, `README.md` updated. |
| **Phase 7** | Stretch: Shared-World Collisions & Race Incidents | **DOCUMENTED** | Parallel-synchronized multi-car physics operational with stewards rules engine; Box2D shared-world contact evaluation documented in `docs/architecture.md`. |

---

## 3D Replay Viewer (Branch `viewer-3d`)

| 3D Phase | Description | Status | Verification Evidence / Gate Notes |
|:---:|:---|:---:|:---|
| **Phase 0** | Discovery & Safety Check | **DONE** | Process audit verified no running experiment; branch `viewer-3d` checked out; constants documented from Gym CarRacing source. |
| **Phase 1** | Replay Export (`f1sim export3d`, 6 cars, wheel flags, coords) | **DONE** | 6-car grid expansion (APX, VEL, NDX, ZEN, RDR, ORB), 4-wheel track-limit flags, compact JSON exporter (`race_3003.json` at 0.79 MB < 6 MB), coordinate test on asymmetric track passes (det(R)=1.0). |
| **Phase 2** | 3D Scene (`docs/viewer/index.html` + `js/`, Three.js r128 offline) | **DONE** | Night race with floodlights, procedural asphalt ribbon, red/white kerbs, single checkered line, 5-light start gantry, low-poly procedural F1 cars, 6 camera modes. Verified in browser subagent. |
| **Phase 3** | Playback & HUD (Timing Tower, Grid Config, Manual Start) | **DONE** | Timing tower click-to-focus without time reset, grid position selector modal, manual 5-light FIA start sequence, telemetry HUD with $K=3.6$ note, responsive layouts tested. |
| **Phase 4** | Deterministic Video & Media Capture (`video3d`) | **DONE** | Frame-by-frame capture script (`f1sim/video3d.py`), 20s hero GIF (`docs/media/hero.gif` at 9.21 MB < 10 MB), race video (`docs/media/race3d.mp4` at 7.37 MB < 25 MB), 6 screenshots on showcase seed 3003, performance benchmark (0.435 ms/frame, >2200 FPS). |
| **Phase 5** | Repository & README Update | **DONE** | Updated README with hero GIF, 3D viewer section, GitHub Pages deployment guide, vendor MIT license, docs/viewer/README.md, all 39 unit tests passing. |
| **Deploy**  | Vercel Live Deployment (`behavioral-cloning-carracing.vercel.app`) | **LIVE & VERIFIED** | Merged `viewer-3d` to `main`, pushed to GitHub, `vercel.json` and client-side redirect route `/` to `/docs/viewer/`. Browser subagent verified 3D WebGL scene, timing tower, and manual start sequence live. |
| **Fix/Laps**| Visible Track Mesh & 10-Lap Grand Prix Upgrade | **DONE & VERIFIED** | Fixed track asphalt normal inversion (CCW winding [0, +93.34, 0], DoubleSide, white edge boundaries); simulated & exported full 10-lap 6-car Grand Prix (12,302 steps, 246s, 7.54 MB). Verified in browser subagent. |

---

## Detailed Log

### Phase 0: Discovery & Safety Auditing
- [x] Read `README.md`, `results/protocol.md`, `results/summary.md`.
- [x] Ran `git status` and `git log --oneline -15`.
- [x] Checked running processes (`Get-Process python*`): PID 71044 & PID 3856 are background HTTP servers on port 8080.
- [x] Checked `logs/experiment.log`: Experiment completed previously (`ALL DONE in 93.8 minutes`).
- [x] Created git tag `v0-before-f1` and created/switched to branch `f1-premium`.
- [x] Audited tracked files over 50 MB: **0 files over 50 MB**.
- [x] LICENSE audit: Added MIT License in LICENSE (committed in `b7bc5ba`).
- [x] Documented existing UI code: `index.html` (34.5 KB) and `race_viewer.html` (34.5 KB) web app, `race_multicar.py` simulator.

### Phase 1: Reliability
- [x] 1a: Test suite in `tests/`: 34 tests passing in 88.97s:
  - Step cap (`max_episode_steps` past 1000 verified)
  - Lap logic (`lap_finished` decoupled from out-of-bounds termination)
  - Patch idempotency (`FrictionDetector` sentinel `_f1sim_patched`)
  - Determinism (same seed bit-identical)
  - Metric unit tests on synthetic geometry
- [x] 1b: `f1sim doctor` diagnostics command verified: Python 3.13, Gym 1.4.0, Box2D 2.3.10, Pygame 2.5.8, Torch, Keras, all 9 models present.
- [x] 1c: Start/Finish seam bug diagnosed on 5 tracks (1000–1004). Root cause: Gym draws `road_poly` directly without checkered line; duplicate patches caused by mutating `road_poly` slicing. Resolved by custom renderer.
- [x] Dev evaluation sanity check: `plain_s0` on dev tracks 1000–1019 reproduced **20/20 laps (100%)** and **2.09m mean center-line distance** bit-identically against `results/dev_final.csv`.

### Phase 2: Simulation Core (`f1sim/`)
- [x] `f1sim/track.py`: Windowed tracker `[-6, +20]` preventing teleportation, sector boundaries (S1, S2, S3), signed/unsigned centre distance. Cross-validated against Gym on 5 dev tracks.
- [x] `f1sim/drivers.py`: `ExpertDriver`, `ModelDriver`, `BatchedModelDriver`, F1 teams registry (`APX`, `VEL`, `NDX`, `ZEN`), staggered grid calculator.
- [x] `f1sim/timing.py`: 100 checkpoints/lap, purple/green/yellow sector status, true common-checkpoint interval & gap calculation.
- [x] `f1sim/replay.py`: Frame-by-frame telemetry recorder and compressed json serialization.

### Phase 3: Stewards Rules Engine (`f1sim/stewards.py`)
- [x] RulesConfig with track limit warnings (1-2), black & white flag (3), 5s penalty (4), 10s repeat penalty (5+).
- [x] 150-consecutive-frame offroad threshold triggers DNF (Retired).
- [x] False start detection (speed > 1.0 m/s before lights out).
- [x] Documented in `docs/rules.md`.

### Phase 4: Live Dashboard Canvas (`f1sim/renderer.py`)
- [x] 1920x1080 broadcast canvas with responsive DPI scaling (720p, 1080p, 1440p).
- [x] Single authentic checkered start/finish line at waypoint 0.
- [x] Follow camera, minimap with sector markers, on-board 84x96 camera inset, timing tower, live telemetry HUD (speed, steer dial, throttle/brake bars, lateral G, autonomy).
- [x] Floating race control notifications banner.
- [x] Inspected and verified screenshot `docs/dashboard_preview_1080p.png`.

### Phase 5: Sessions & Media (`f1sim/session.py`)
- [x] Synchronized parallel multi-car Grand Prix session runner.
- [x] CLI entry points: `python -m f1sim doctor`, `python -m f1sim race`, `python -m f1sim weekend`.
- [x] Verified showcase race on seed 3003:
  - Official Winner: Marco Silva (`VEL`) in 2 laps, followed by Lukas Lind (`NDX`) and Alex Vance (`APX`, Fastest Lap).
  - Generated hero video: `videos/f1_showcase_3003.mp4` (9.69 MB, < 25 MB).
  - Generated showcase GIF: `videos/f1_showcase_3003.gif` (1.14 MB, < 5 MB).
  - Generated compressed replay telemetry: `data/telemetry_latest.json.gz` (295 KB).

### 3D Viewer Phase 0: Discovery
- [x] Audited running python processes (no training/evaluation in flight).
- [x] Created and switched to branch `viewer-3d`.
- [x] Inspected CarRacing source constants:
  - `SCALE = 6.0` (pixels per world unit in 2D rendering)
  - `TRACK_WIDTH = 40 / SCALE = 6.6667`
  - Road half-width = `TRACK_WIDTH / 2 = 3.3333` (Total road width = 13.333)
  - `PLAYFIELD = 2000 / SCALE = 333.3333`
  - `FPS = 50`
  - Car hull bounds: Length = 5.00 world units, Width = 2.40 world units
  - Car wheels: Radius = 0.54, Width = 0.56, Wheelbase = 3.24, Track = 2.20 world units.
- [x] Identified user feedback:
  1. Fix timing tower click resetting video playback time.
  2. Add manual lights-out start (5-light gantry sequence).
  3. Expand to 6 cars with configurable starting grid order and positions.
  4. Real dynamic physics and varied behavior across all cars.

### 3D Viewer Phase 1: Replay Export & Mirror Gate
- [x] Expanded teams registry in `f1sim/drivers.py` to 6 fictional constructors: APX, VEL, NDX, ZEN, RDR, ORB.
- [x] Created `docs/viewer/COORDS.md` establishing $(X_{\text{3D}} = x_{\text{sim}}, Y_{\text{3D}} = 0, Z_{\text{3D}} = -y_{\text{sim}})$ and yaw rotation $\text{rotation.y} = \text{hull.angle}$.
- [x] Vendored Three.js r128 UMD (603 KB) and MIT LICENSE into `docs/viewer/vendor/`.
- [x] Vendored OFL BigShoulders font into `docs/viewer/fonts/`.
- [x] Implemented `f1sim/export3d.py` and CLI `python -m f1sim export3d`. Compact JSON format under 6 MB limit (achieved 0.79 MB for 6 cars, 2,870 frames).
- [x] Wrote `tests/test_coords_mirror.py` testing asymmetric track rigid transform (det(R) == +1.0) and left-turn parity. Passed on seeds 3000, 3001, 3002.

### 3D Viewer Phase 2: 3D Scene Architecture
- [x] Track: Road ribbon mesh with procedural asphalt canvas texture, white boundary lines, high-curvature kerbs, barrier walls, floodlights, stars, and grandstands.
- [x] Start/Finish: Single checkered line at tile 0 with 5-light start gantry and staggered starting grid slots.
- [x] Procedural F1 Cars: Low-poly chassis, wings, halo, helmet, 4 steerable/spinning wheels, brake lights, driver billboard tags.
- [x] 6 Smooth Camera Modes: Chase, Onboard, TV Heli, Top-Down, Trackside handover, and Free Orbit.

### 3D Viewer Phase 3: Playback Controls & HUD
- [x] FIA 5-Light Start Gantry: Page loads with cars idling on grid; user triggers manual countdown (1..5 red LEDs sequence on, random pause, lights out to launch).
- [x] Timing Tower: Real-time order, gaps, fastest lap badge, click-to-focus car without resetting playback time.
- [x] Telemetry HUD: Real speed with documented conversion ($K = 3.6$ km/h per world unit/s), throttle, brake, steering dial, and 4-wheel track-limit dots.
- [x] Starting Grid Customizer: Modal allowing user to select starting positions and grid order for all 6 cars.
- [x] Fixed 2D viewer `index.html` timing tower click bug via `loadedmetadata` event listener.

### 3D Viewer Phase 4: Deterministic Video & Media Capture
- [x] Implemented `f1sim/video3d.py` supporting deterministic frame stepping (?capture=1&frame=N&camera=C via `window.seekFrame`).
- [x] Headless Chrome automation with WebGL via Selenium, capturing full DOM broadcast presentation.
- [x] Generated `docs/media/race3d.mp4`: 1280x720 @ 30 fps, 35s duration, 1050 frames, 7.37 MB (< 25 MB limit).
- [x] Generated `docs/media/hero.gif`: 800x450 @ 12 fps, 20s duration, 240 frames, 9.21 MB (< 10 MB limit) via ffmpeg palettegen.
- [x] Captured 6 high-res showcase screenshots across all camera perspectives:
  - `screen_01_grid_lights.png` (Starting grid & 5-light FIA gantry)
  - `screen_02_chase_action.png` (Chase camera multi-car high-speed battle)
  - `screen_03_onboard_cockpit.png` (Onboard nose-level cockpit view)
  - `screen_04_tv_helicopter.png` (TV helicopter sweep over lead pack)
  - `screen_05_topdown_seam.png` (Top-down orthographic view of single checkered line)
  - `screen_06_trackside_handover.png` (Fixed trackside post camera handover)
- [x] WebGL Performance Benchmark:
  - Low Quality: 0.376 ms/frame (2,659 FPS), 236 draw calls, 19,076 triangles.
  - Medium Quality: 0.435 ms/frame (2,298 FPS), 204 draw calls, 18,678 triangles.
  - High Quality: 0.419 ms/frame (2,386 FPS), 200 draw calls, 18,630 triangles.

### 3D Viewer Phase 5: Repository & Documentation
- [x] Updated `README.md` with:
  - Hero GIF at top of document (`docs/media/hero.gif`).
  - Dedicated "3D WebGL Replay Viewer" section with local execution guide and GitHub Pages deployment instructions (Settings -> Pages -> branch main, folder /docs).
  - Explicit Simulation Truth vs. Cosmetic breakdown table.
  - Mandatory disclaimer: "3D VISUALISATION · SIM PHYSICS IS 2D".
  - 3-step guide to export and view replays.
  - Expanded fictional constructors roster to all 6 teams (APX, VEL, NDX, ZEN, RDR, ORB).
  - Preserved all experimental results numbers from `results/test_results.csv`.
- [x] Created `docs/viewer/README.md` with complete keyboard map (1-6 cameras, Space, F, H), coordinate system notes, and offline usage guide.
- [x] Vendored Three.js r128 UMD (603 KB) and MIT License in `docs/viewer/vendor/`.
- [x] Vendored OFL BigShoulders font in `docs/viewer/fonts/`.
- [x] Verified zero tracked files over 50 MB, video under 25 MB (`race3d.mp4` 7.37 MB), hero GIF under 10 MB (`hero.gif` 9.21 MB).
- [x] Updated `.gitignore` to prevent temporary MP4s.
- [x] All 39 unit tests in `tests/` pass cleanly.



