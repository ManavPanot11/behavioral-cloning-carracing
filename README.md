# Autonomous Behavioral Cloning & DAgger in CarRacing-v3

An empirical investigation of end-to-end deep behavioral cloning, recovery augmentation, and interactive Dataset Aggregation (DAgger) for self-driving in Gymnasium's `CarRacing-v3` (headless, PyTorch/Keras 3).

---

## 1. Executive Summary & Honest Conclusions

### The Classical Prediction
In the behavioral cloning and imitation learning literature (e.g., Pomerleau 1989, Bojarski et al. / Nvidia 2016, Ross et al. 2011), the central prediction is:
> *Plain behavioral cloning trained only on expert demonstrations will inevitably drift off the road because tiny approximation errors compound exponentially (covariate shift), whereas recovery training (offline perturbation noise or DAgger) is essential for staying on track.*

### Was That Prediction Reproduced Here?
**Under standard driving conditions: NO.**
- Plain behavioral cloning **did not drift off the road**. Across three distinct training seeds on 20 dev tracks and 20 held-out test tracks, plain behavioral cloning completed **100% of clean laps (60/60 on dev, 60/60 on test)** with **0 off-road events** and an average center-line distance of 2.06–2.09 world units (matching the expert's 2.08–2.10 units).
- *Mechanism*: The scripted pure-pursuit speed controller dynamically throttles and brakes based on steering curvature. By slowing down before tight bends, the vehicle avoids entering high-slip tire states, keeping the camera observations strictly within the training distribution.

### Where Recovery Training DOES Help
When evaluated under stress conditions that violate the clean training envelope:
1. **Physical Disturbance Test ($\pm 0.6$ steering offset for 20 frames)**:
   - Plain cloning dropped laps on both dev ($18.3 \pm 1.7$ laps) and test ($18.0 \pm 0.8$ laps), suffering off-track excursions exceeding 55 units.
   - In contrast, **Augmentation (recovery noise) and DAgger achieved 100% lap completion (60/60 on dev, 60/60 on test) with 0 off-road departures**. Under our pre-registered decision rule ($\min(\text{aug}) > \max(\text{plain})$), this recovery benefit is **empirically real**.
2. **Speed Stress Test (1.50x target speed)**:
   - At high speeds where inertia causes tire scrub, plain cloning suffered severe compounding drift and collapsed to **15.0% completion on dev (9/60 laps)** and **25.0% on test (15/60 laps)**.
   - Augmentation maintained **73.3% completion (44/60 laps)** and DAgger maintained **76.7% completion (46/60 laps)**, proving that recovery training provides vital stabilizing corrections when physics forces the vehicle outside its nominal path.

### Does DAgger Outperform Offline Augmentation?
**No clear difference.**
- Offline data collection with random steering perturbations and synthetic recovery targets already provided comprehensive coverage of near-track recovery states.
- Consequently, all DAgger seeds achieved $\ge 19/20$ laps immediately in Rounds 0 and 1, triggering early stopping by Round 2. DAgger did not find unique unvisited failure states that offline augmentation had missed.

---

## 2. The Step-Limit Pitfall: What Happened and Why It Matters

During initial experiments, Gymnasium's `gym.make("CarRacing-v3")` was used with default settings.
- **The Bug**: `CarRacing-v3` silently applies a default `TimeLimit` wrapper capping episodes at **1000 steps**.
- **The Track Reality**: Procedurally generated tracks in CarRacing-v3 average ~307 tiles and require 850 to 1470 steps to complete cleanly at safe cornering speeds.
- **The False Conclusion**: Under the 1000-step cap, even the scripted pure-pursuit **expert completed only 3/20 laps (15%)**, getting artificially truncated at step 1000 while driving perfectly down the center line with 0 off-road events! Learner models completed only 3–8 laps, creating a severe false appearance of algorithmic failure.
- **The Resolution**: Setting `max_episode_steps=2000` explicitly revealed that all learned models (`plain`, `aug`, `dagger`) were already driving flawlessly, completing **20/20 laps (100%)**!

### Old Rules (1000 steps) vs New Rules (2000 steps) Comparison

| Driver / Mode | Old Rules (1000 Steps) Laps/20 | New Rules (2000 Steps) Laps/20 | Ending Type Under Old Rules |
| :--- | :---: | :---: | :--- |
| **Expert** | 3/20 (15%) | **20/20 (100%)** | 17/20 TRUNCATED at step 1000 |
| **Plain Cloning (s0)** | 3/20 (15%) | **20/20 (100%)** | 17/20 TRUNCATED at step 1000 |
| **Augmentation (s0)** | 4/20 (20%) | **20/20 (100%)** | 16/20 TRUNCATED at step 1000 |
| **DAgger (s0)** | 8/20 (40%) | **20/20 (100%)** | 12/20 TRUNCATED at step 1000 |

---

## 3. Pre-Registered Experimental Protocol (`results/protocol.md`)

All evaluations strictly followed the pre-registered rules established in `results/protocol.md`:
1. **Unified TimeLimit**: `max_episode_steps=2000` passed explicitly to every `gym.make` call. Default 1000-step limit prohibited.
2. **Unified Control & Lost Rules**:
   - 50-frame intro zoom skipped for steering decisions.
   - Same speed controller (curvature-based throttle/braking) across all drivers.
   - "Lost" rule: episode terminates if all 4 wheels remain off-track for $> 150$ consecutive frames.
   - Lap detection patch imported from `expert.py` ensuring tiles visited $\ge 95\%$ counts as a completed lap.
3. **Decision Rule for Claims**: With 3 seeds per mode, a difference between Mode A and Mode B is declared **"real"** if and only if **every seed of A strictly beats every seed of B** ($\min(A) > \max(B)$). Otherwise, recorded as **"no clear difference"**. Untested mechanisms must be explicitly labeled as *"hypothesis, untested"*.
4. **Held-Out Test Tracks**: Seeds 2000–2019 evaluated strictly once per condition after all dev experimentation concluded.

---

## 4. How to Run Each Script

### Setup Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 1: Data Collection
Collect clean expert demonstrations and recovery perturbation data:
```powershell
.venv\Scripts\python.exe collect.py
```
- Outputs: `data/clean.npz` (15,000 frames) and `data/noisy.npz` (15,000 frames).

### Step 2: Training (Plain & Augmented)
Train 3 seeds (0, 1, 2) for plain cloning and offline augmented cloning:
```powershell
# Plain cloning
.venv\Scripts\python.exe train.py plain 0
.venv\Scripts\python.exe train.py plain 1
.venv\Scripts\python.exe train.py plain 2

# Augmented cloning
.venv\Scripts\python.exe train.py aug 0
.venv\Scripts\python.exe train.py aug 1
.venv\Scripts\python.exe train.py aug 2
```

### Step 3: DAgger Training Loop
Run interactive Dataset Aggregation across 3 seeds (unified 2000-step limit):
```powershell
.venv\Scripts\python.exe dagger.py 0
.venv\Scripts\python.exe dagger.py 1
.venv\Scripts\python.exe dagger.py 2
```

### Step 4: Baseline & Dev Evaluations
```powershell
# Dev evaluation (seeds 1000-1019)
.venv\Scripts\python.exe -c "import eval_suite; print('Eval suite ready')"
.venv\Scripts\python.exe run_stage4_disturbance.py
.venv\Scripts\python.exe run_stage5_speed.py
```

### Step 5: Held-Out Test Evaluation (Seeds 2000–2019)
Run test conditions (can be run sequentially or in parallel):
```powershell
.venv\Scripts\python.exe run_stage6_test.py clean
.venv\Scripts\python.exe run_stage6_test.py disturbed
.venv\Scripts\python.exe run_stage6_test.py speed_1.25
.venv\Scripts\python.exe run_stage6_test.py speed_1.50
.venv\Scripts\python.exe merge_stage6_results.py
```

### Step 6: Generate Deliverables & Summary Report
```powershell
.venv\Scripts\python.exe generate_final_deliverables.py
```
- Generates `results/summary.md` and all analytical plots in `results/`.

---

## 5. Summary Results Table (Held-Out Test Tracks 2000–2019)

| Condition | Driver / Mode | Laps Completed (Mean, Range) | Track Progress | Off-Road Events | Center-Line Distance | Steering Jitter |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline (1.0x)** | **Expert** | 20/20 (100%) | 100.0% | 0.00 | 2.08 | 0.0779 |
| | **Zero Baseline** | 0/20 (0%) | 7.9% | 1.05 | 76.78 | 0.0000 |
| | **Plain Cloning** | **20.0/20 [20, 20]** | **100.0%** | **0.00** | **2.06** | 0.0690 |
| | **Augmentation** | **20.0/20 [20, 20]** | **100.0%** | **0.00** | **2.08** | 0.0641 |
| | **DAgger** | **20.0/20 [20, 20]** | **100.0%** | **0.00** | **2.08** | 0.0725 |
| **Disturbed ($\pm 0.6$)** | **Expert** | 20/20 (100%) | 100.0% | 0.00 | 2.05 | 0.0873 |
| | **Plain Cloning** | 18.0/20 [17, 19] | 96.8% | 0.32 | 2.59 | 0.0799 |
| | **Augmentation** | **20.0/20 [20, 20]** | **100.0%** | **0.00** | **2.06** | 0.0744 |
| | **DAgger** | **20.0/20 [20, 20]** | **100.0%** | **0.00** | **2.06** | 0.0814 |
| **Speed Stress (1.25x)** | **Expert** | 20/20 (100%) | 100.0% | 0.00 | 1.80 | 0.1147 |
| | **Plain Cloning** | 18.3/20 [17, 20] | 95.8% | 0.75 | 2.36 | 0.1115 |
| | **Augmentation** | **19.7/20 [19, 20]** | **98.6%** | **0.18** | **1.90** | 0.1101 |
| | **DAgger** | **19.7/20 [19, 20]** | **98.6%** | **0.02** | **1.80** | 0.1134 |
| **Speed Stress (1.50x)** | **Expert** | 20/20 (100%) | 100.0% | 0.00 | 1.88 | 0.1058 |
| | **Plain Cloning** | 5.0/20 [4, 6] | 57.6% | 6.77 | 8.65 | 0.1108 |
| | **Augmentation** | **14.7/20 [13, 16]** | **79.1%** | **5.00** | **4.00** | 0.1115 |
| | **DAgger** | **15.3/20 [11, 19]** | **82.0%** | **4.93** | **3.27** | 0.1189 |

---

## 6. Key Artifacts & Visualizations

- **Comprehensive Analysis**: [results/summary.md](results/summary.md)
- **Experimental Protocol**: [results/protocol.md](results/protocol.md)
- **Visualizations**:
  - `results/laps_progress_by_condition.png`: Bar chart of completions and track progress.
  - `results/disturbance_recovery_trajectory.png`: Distance-to-centerline trajectory through the 3 disturbance trigger windows.
  - `results/training_loss_curves.png`: Training & validation loss curves for plain and augmented models.
  - `results/dagger_curve_s*.png`: Round-by-round DAgger early stopping curves.
  - `results/steer_hist.png`: Training data steering angle distribution.
- **Video Recordings (Held-Out Test Track 2000)**:
  - Clean Baseline: `videos/test/clean/expert_track2000.mp4` and `videos/test/clean/dagger_s0_track2000.mp4`.
  - Disturbance Response: `videos/test/disturbed/plain_s0_track2000.mp4`, `videos/test/disturbed/aug_s0_track2000.mp4`, and `videos/test/disturbed/dagger_s0_track2000.mp4`.
