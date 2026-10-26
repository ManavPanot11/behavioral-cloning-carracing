# 3D WebGL Replay Viewer & Telemetry Dashboard

An interactive, broadcast-style 3D WebGL race visualization built using Three.js (vendored offline r128) and plain ES modules.

> **3D VISUALISATION · SIM PHYSICS IS 2D**  
> *Notice: All vehicle dynamics, collisions, and lap telemetry are computed directly by the 2D Box2D physics simulation in Gymnasium `CarRacing-v3`. The 3D viewer is purely a real-time rendering layer. Unofficial F1-inspired motorsport simulation; not affiliated with Formula 1 or the FIA.*

---

## Key Features

1. **True-to-Scale 3D Circuit**:
   - Road ribbon extracted directly from CarRacing Box2D track geometry (`TRACK_WIDTH = 6.667`, road half-width = $3.333$m, total road width = $13.333$m).
   - High-curvature red/white kerbs, barrier walls, floodlight poles, night sky dome with 1,800 stars.
   - **Single checkered start/finish line** at waypoint 0 with overhead 5-light start gantry and staggered starting grid slots.
2. **Procedural Low-Poly F1 Cars**:
   - Scaled from the real CarRacing hull bounds ($5.00$m $\times$ $2.40$m, wheelbase $3.24$m).
   - Front and rear wings, halo cockpit protection, driver helmet in constructor colors.
   - Steerable front wheels, spinning tires, dynamic brake lights, and 3D driver code billboard tags.
3. **6 Broadcast Camera Angles**:
   - **[1] Chase Cam**: Trailing third-person perspective with smooth lookahead tracking.
   - **[2] Onboard Nose Cam**: Low-slung front cockpit view capturing close racing action.
   - **[3] TV Helicopter Cam**: Cinematic sweeping camera orbiting the selected car.
   - **[4] Top-Down Orthographic**: Overhead view matching the 2D race dashboard orientation.
   - **[5] Trackside Handover**: Fixed trackside posts that dynamically hand over as the car passes.
   - **[6] Free Orbit**: Full 360° mouse / touch orbital inspection.
4. **FIA Manual 5-Light Start Sequence**:
   - Cars line up and idle on the starting grid.
   - Manual start button initiates the 5 red light sequence (1..5 LEDs illuminate, random hold, lights out to launch).
5. **F1 Broadcast HUD & Timing Tower**:
   - Live timing tower with positions, constructor badges, interval gaps, and fastest lap indicators.
   - Click any driver in the tower to focus the camera without resetting playback.
   - Live telemetry: Speed ($\text{km/h} = 3.6 \times v$), throttle/brake bars, steering dial, track progress %, and 4-wheel track-limit indicators.
   - Interactive circuit minimap with real-time colored car tracking dots.
   - Starting grid customizer modal allowing full starting order configuration.

---

## Keyboard Shortcuts

| Key | Action |
|:---:|:---|
| **SPACE** | Play / Pause playback |
| **1** | Select Chase Camera |
| **2** | Select Onboard Nose Camera |
| **3** | Select TV Helicopter Camera |
| **4** | Select Top-Down Orthographic Camera |
| **5** | Select Trackside Handover Camera |
| **6** | Select Free Orbit Camera (Click & drag to rotate, scroll to zoom) |
| **F** | Toggle Fullscreen Mode |
| **H** | Toggle Help & Controls Overlay |

---

## Truth vs. Cosmetic Effects

| Element | Classification | Source & Method |
|:---|:---:|:---|
| **Car Coordinates $(x, y)$** | **SIM TRUTH** | Exact Box2D hull position from 2D sim log ($X_{\text{3D}} = x_{\text{sim}}, Z_{\text{3D}} = -y_{\text{sim}}$). |
| **Heading & Yaw Angle** | **SIM TRUTH** | Exact Box2D hull angle ($\text{rotation.y} = \text{hull.angle}$). |
| **Speed** | **SIM TRUTH** | Real Box2D linear velocity magnitude ($\text{km/h} = \text{speed} \times 3.6$). |
| **Throttle & Brake** | **SIM TRUTH** | Direct action vector components $a[1]$ (gas) and $a[2]$ (brake). |
| **Steering Angle** | **SIM TRUTH** | Direct action vector component $a[0]$ in $[-1.0, 1.0]$. |
| **Wheel Track Limits** | **SIM TRUTH** | Four independent wheel contact booleans `[FL, FR, RL, RR]` from `FrictionDetector`. |
| **Body Pitch & Roll** | *Cosmetic* | Subtle chassis tilt derived from longitudinal acceleration and lateral G ($\pm 2.5^\circ$). |
| **Tire Spin & Steer** | *Cosmetic* | Wheels rotate based on linear speed; front wheel hubs steer based on logged steering action. |
| **Floodlights & Shadows** | *Cosmetic* | Directional night lighting and soft shadow maps for visual depth. |

---

## Running Offline

The 3D viewer is completely self-contained with no external CDN dependencies:
- Three.js r128 is vendored in `docs/viewer/vendor/three.min.js` (MIT License).
- OFL BigShoulders font is vendored in `docs/viewer/fonts/`.

To launch locally:
```bash
python -m http.server 8080
```
Open your browser to:
[http://localhost:8080/docs/viewer/index.html](http://localhost:8080/docs/viewer/index.html)

---

## Exporting & Recording Replays

1. **Export a new 3D replay from simulation**:
   ```bash
   python -m f1sim export3d --seed 3003 --laps 2 --out docs/viewer/data/race_3003.json
   ```
2. **Capture deterministic MP4 video**:
   ```bash
   python -m f1sim video3d --replay docs/viewer/data/race_3003.json --out docs/media/race3d.mp4 --camera mixed --duration 35 --size 1280x720 --fps 30
   ```
3. **Capture hero GIF**:
   ```bash
   python -m f1sim video3d --replay docs/viewer/data/race_3003.json --out docs/media/hero.gif --camera mixed --duration 20 --size 800x450 --fps 12
   ```
4. **Capture 6 showcase screenshots**:
   ```bash
   python -m f1sim video3d --replay docs/viewer/data/race_3003.json --screenshots
   ```
