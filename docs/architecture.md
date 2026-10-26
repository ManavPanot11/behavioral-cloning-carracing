# System Architecture & Engineering Design

This document details the modular software architecture of the `f1sim` simulation engine and broadcast suite.

```mermaid
graph TD
    subgraph Core Simulation
        A[CarRacing-v3 Environment] --> B[f1sim.env.F1EnvWrapper]
        B --> C[Idempotent Finish Line Patch]
        B --> D[f1sim.track.Track]
    end

    subgraph Policies & Control
        D --> E[f1sim.drivers.ExpertDriver]
        D --> F[f1sim.drivers.ModelDriver]
        F --> G[84x96 Cropped Neural Model]
        E --> H[Speed Controller]
        F --> H
    end

    subgraph Adjudication & Timing
        B --> I[f1sim.timing.TimingSystem]
        B --> J[f1sim.stewards.Stewards]
        I --> K[100 Micro-Checkpoints]
        J --> L[Track Limits & Penalties]
    end

    subgraph Broadcast & UI
        I --> M[f1sim.renderer.DashboardRenderer]
        J --> M
        B --> M
        M --> N[1920x1080 Broadcast Canvas]
        N --> O[Hero MP4 / GIF Exporter]
        N --> P[Web Race Viewer]
    end
```

---

## 1. Modular Subsystems

### `f1sim.env` (Environment & Patching)
- **Problem Solved**: Gymnasium's native `CarRacing-v3` has two known bugs:
  1. A silent 1000-step `TimeLimit` wrapper that truncates legitimate laps.
  2. A race condition where driving outside the playfield sets `terminated=True`, which naive scripts misclassified as a completed lap.
- **Solution**:
  - `apply_finish_line_patch()`: Idempotently wraps `FrictionDetector.BeginContact` with a sentinel `_f1sim_patched` attribute to detect lap crossing when visited tiles $> 95\%$.
  - `F1EnvWrapper`: Explicitly decouples `lap_finished` from `out_of_bounds`, normalizes actions, and provides a multi-lap `race_mode`.

### `f1sim.track` (Geometry & Windowed Tracking)
- **Problem Solved**: Global nearest-neighbor search jumps or teleports when tracks loop or self-intersect.
- **Solution**:
  - Constrained local window search `[-6, +20]` along track nodes.
  - Continuous fractional progress interpolation along track tangent vectors.
  - Perpendicular signed and unsigned centre-line distance calculation.

### `f1sim.drivers` (Policies & Fictional Teams)
- **Integrity Guarantee**:
  - Models receive the exact 84x96 cropped RGB gym observation with float32 pixel values.
  - Fictional teams adhere to unofficial branding rules: `APX` (Apex Racing GP), `VEL` (Veloce Corse), `NDX` (Nordic Dynamics), `ZEN` (Zenith Motorsport).

### `f1sim.timing` & `f1sim.stewards` (Timing & Adjudication)
- **Precision**: 100 micro-sector checkpoints calculate true clock deltas $(\Delta t = t_B - t_A)$ at common physical coordinates.
- **Regulation**:
  - Excursions 1-2: Warning.
  - Excursion 3: Black & White Flag.
  - Excursion 4: 5s Time Penalty.
  - Excursions 5+: 10s Time Penalties.
  - Consecutive off-road $> 150$ frames: Immediate DNF / Retirement.

### `f1sim.renderer` (Broadcast Graphics)
- 1920x1080 Pygame canvas with responsive DPI scaling (720p, 1080p, 1440p).
- Single authentic checkered start/finish line at waypoint 0.
- Live follow-camera, minimap, driver cam inset (84x96), speedometer, steer dial, throttle/brake gauges.
