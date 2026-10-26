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

## Detailed Log

### Phase 0: Discovery & Safety Auditing
- [x] Read `README.md`, `results/protocol.md`, `results/summary.md`.
- [x] Ran `git status` and `git log --oneline -15`.
- [x] Checked running processes (`Get-Process python*`): PID 71044 & PID 3856 are background HTTP servers on port 8080.
- [x] Checked `logs/experiment.log`: Experiment completed previously (`ALL DONE in 93.8 minutes`).
- [x] Created git tag `v0-before-f1` and created/switched to branch `f1-premium`.
- [x] Audited tracked files over 50 MB: **0 files over 50 MB**.
- [x] LICENSE audit: **No LICENSE file found**. Proposing MIT License to user.
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
