# Experimental Protocol & Evaluation Pre-Registration

**Timestamp**: 2026-10-25T18:38:30+05:30 (Local Time) / 2026-10-25T13:08:30Z (UTC)  
**Project**: End-to-End Behavioral Cloning & DAgger in Gymnasium CarRacing-v3  
**Repository**: `E:\car`

---

## 1. The Unified Evaluation Rules (Applied Identically to Every Driver)

1. **Step Limit**:
   - Explicitly configured as `max_episode_steps=2000` passed directly to `gym.make("CarRacing-v3", ..., max_episode_steps=2000)` in all evaluation scripts (`evaluate.py`, `diagnose.py`, disturbance test, speed stress test, and final test evaluation).
   - Gymnasium's implicit default `TimeLimit` wrapper (which silently truncated episodes at step 1000) is strictly overridden.
   - Every episode is permitted to execute up to 2000 steps if needed.

2. **Episode Lifecycle & Driver Parameters**:
   - **50-Frame Intro**: At steps $0 \le t < 50$, the car zooms in, steering is forced to $0.0$, and the speed controller initializes the car.
   - **Lost Rule**: If all four wheels are completely off-track for $> 150$ consecutive frames (`offroad_streak >= 150`), the episode terminates immediately with ending `LOST`.
   - **Speed Controller**: The pure-pursuit speed controller from `expert.py` is applied identically to all drivers (expert, zero, plain, aug, dagger), calculating target speed as a smooth function of steering magnitude:
     $$\text{Target Speed} = f(|\text{steer}|; v_{\text{straight}}=70.0, v_{\text{turn}}=38.0, v_{\text{min}}=26.0)$$
   - **Lap-Detection Patch**: Every script imports `from expert import ...`, activating the patched `car_racing.FrictionDetector.BeginContact` that sets `env.unwrapped.new_lap = True` whenever tile 0 is visited and $\ge 95\%$ of track tiles have been covered.

3. **Metrics Recorded Per Episode**:
   - `lap_completed`: Boolean indicating if the car crossed the finish line after $\ge 95\%$ track coverage.
   - `steps`: Total frames in the episode ($50 \le \text{steps} \le 2000$).
   - `track_progress`: Percentage of track tiles visited at termination ($\frac{\text{tiles\_visited}}{\text{total\_tiles}} \times 100\%$).
   - `offroad_events`: Number of discrete incidents where the car transitions from on-track to having 0 wheels on the track.
   - `longest_offroad_streak`: Maximum consecutive frames with 0 wheels on the track.
   - `mean_speed`: Average linear velocity magnitude $\sqrt{v_x^2 + v_y^2}$ across the episode.
   - `mean_abs_steer`: Average steering magnitude $\frac{1}{T} \sum |s_t|$.
   - `steer_jitter`: High-frequency oscillation metric $\frac{1}{T-1} \sum |s_t - s_{t-1}|$.
   - `mean_centre_distance`: Average Euclidean distance from the car center to the nearest track waypoint across all post-intro steps.
   - `max_centre_distance`: Maximum Euclidean distance from the car center to the nearest track waypoint.
   - `ending_type`: One of `LAP` (finished lap), `TRUNCATED` (reached step 2000 while still on road), `LOST` (all wheels off >150 frames), or `OTHER`.
   - `autonomy`: Standard Nvidia metric $(1 - \frac{\text{offroad\_events} \times 6}{\text{seconds}}) \times 100\%$, reported for completeness.

4. **Data Collection Verification**:
   - In `collect.py`, data collection recorded episodes up to 1000 environment steps (950 post-intro frames per episode).
   - In `data/clean.npz`: 48 episodes, min length 742 frames, max length 950 frames, mean 927.3 frames (44,512 total frames).
   - In `data/noisy.npz`: 48 episodes, min length 950 frames, max length 950 frames, mean 950.0 frames (45,600 total frames).

---

## 2. Decision Rule for Claims

With 3 independent random seeds per training mode (`plain`, `aug`, `dagger`):
- A performance difference between Mode A and Mode B on any metric is designated **"real"** if and only if:
  $$\min_{s \in \{0,1,2\}}(\text{Metric}(A_s)) > \max_{s \in \{0,1,2\}}(\text{Metric}(B_s)) \quad (\text{or vice versa for lower-is-better metrics})$$
  *(i.e. every seed of Mode A strictly outperforms every seed of Mode B).*
- If there is any overlap between the seed distributions, the result must be explicitly stated as **"no clear difference"**.
- Mechanistic explanations: No explanation of a result may be claimed as fact unless validated by an empirical experiment in this repository. Any untested explanation must be explicitly labeled as **"hypothesis, untested"**.

---

## 3. Track Split Isolation & Non-Contamination Rule

- **Dev Tracks (Seeds 1000–1019)**: Exclusively used for hyperparameter verification, diagnostic testing, disturbance verification, and speed stress testing.
- **Test Tracks (Seeds 2000–2019)**: Strictly held out for final evaluation in Stage 6. Never used for tuning, model selection, or intermediate checks. Evaluated exactly once per condition.

---

## 4. Pre-Registered Test Conditions

1. **Clean Driving**: Evaluated on all 20 test tracks (seeds 2000–2019) under standard speed controller ($1.0\times$) and zero disturbance.
2. **Disturbance (Recovery) Test**:
   - Disturbance triggers at fractional progress points $[0.25, 0.50, 0.75]$ of the lap, indexed by track waypoint number (identical track locations for all drivers regardless of vehicle speed).
   - Steering perturbation: additive $\Delta s = \pm 0.6$ applied to executed steering for 20 frames, with sign determined by a seeded RNG per `(track_seed, trigger_idx)`. Learner camera inputs and network steering predictions remain unmodified.
   - Additional metrics: `recovered_each_disturbance` (boolean indicating if the car returned within 3.0 world units of the track center line within 60 frames after each perturbation), `max_centre_distance`.
3. **Speed Stress Test**:
   - Target speeds $(v_{\text{straight}}, v_{\text{turn}}, v_{\text{min}})$ scaled uniformly by factors:
     - Baseline: $1.0\times$ (70.0, 38.0, 26.0)
     - Medium Stress: $1.25\times$ (87.5, 47.5, 32.5)
     - High Stress: $1.50\times$ (105.0, 57.0, 39.0)
   - Evaluated across all drivers. Expert performance establishes track feasibility.
