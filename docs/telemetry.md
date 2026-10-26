# Telemetry Specification & Broadcast HUD

This document specifies the genuine real-time telemetry signals captured and displayed in the `f1sim` simulation engine and broadcast dashboard.

---

## 1. Non-Negotiable Telemetry Integrity

In accordance with strict scientific and engineering rigor:
- **NO FAKE TELEMETRY**: Box2D's `CarRacing-v3` environment does NOT simulate gearboxes, DRS wings, pit lanes, tire thermal degradation, or fuel consumption. None of these are synthesized or fabricated.
- **Genuine Physical Measurements**: All metrics displayed on the HUD or logged into telemetry traces derive directly from Box2D vehicle dynamics, track geometry, and the neural network / expert policies.

---

## 2. Telemetry Channels & Formulations

| Channel | Units | Source | Description |
|:---|:---:|:---|:---|
| **Speed** | km/h | `hull.linearVelocity` | Physical scalar speed calculated as $\sqrt{v_x^2 + v_y^2} \times 1.6$. |
| **Steering Angle** | $[-1.0, +1.0]$ | Model / Controller | Normalized steering deflection command. Negative = Left, Positive = Right. |
| **Throttle (Gas)** | $[0.0, 1.0]$ | Speed Controller | Engine forward driving torque request. Rendered as green LED bar. |
| **Brake** | $[0.0, 1.0]$ | Speed Controller | Wheel braking torque request. Rendered as red LED bar. |
| **Lateral G-Force** | G ($9.81 \, \text{m/s}^2$) | `hull.angularVelocity` | Lateral dynamic acceleration: $|a_{lat}| = |\omega \cdot v| / 9.81$. |
| **Centre-Line Distance** | meters (m) | Track Spline | Perpendicular Euclidean distance from vehicle centroid to circuit centerline. |
| **Track Progress** | % $[0, 100)$ | Windowed Waypoints | Continuous interpolated progress along circuit waypoints. |
| **Micro-Sector Checkpoints**| Index $[0, 99]$ | Timing System | 100 spatial checkpoints per lap for gap & delta computation. |
| **Autonomy %** | % $[0, 100]$ | Intervention Protocol | Nvidia standard: $\max\left(0, \left(1 - \frac{\text{excursions} \times 6}{\text{time}}\right) \times 100\right)$. |

---

## 3. F1 Broadcast Sector Color Logic

1. 🟣 **Session Purple**:
   - Awarded when a car sets the fastest individual sector time (Sector 1, 2, or 3) recorded across all vehicles during the session.
2. 🟢 **Personal Green**:
   - Awarded when a driver improves upon their own personal best time in that sector.
3. 🟡 **Yellow**:
   - Displayed when a sector is completed slower than the driver's current personal best.

---

## 4. Gap & Interval Precision

- Traditional video games compute gap by dividing physical distance by speed, causing wild fluctuations in corners.
- `f1sim` implements **Checkpoint Passage Timing**:
  - Whenever a vehicle passes one of the **100 spatial checkpoints**, its frame step timestamp is recorded.
  - The interval to the vehicle ahead is calculated as the true clock difference $(\Delta t = t_{\text{chaser}} - t_{\text{leader}})$ when both cars crossed the identical geographic checkpoint.
