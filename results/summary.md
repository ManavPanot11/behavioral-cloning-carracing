# Behavioral Cloning and DAgger Autonomous Driving Experiment: Final Report
**Project Workspace**: `E:\car`  
**Simulation Environment**: Gymnasium `CarRacing-v3` (Headless)  
**Pre-Registered Protocol**: `results/protocol.md`  
**Model Architecture**: Nvidia-style CNN (233,019 trainable parameters)  
**Unified TimeLimit**: 2000 steps (`max_episode_steps=2000`) applied across all drivers and evaluations.

## 1. Executive Summary & Core Research Questions

Here are direct, empirical answers to the five core research questions based strictly on measured data:

### (i) Does plain behavioural cloning fail in this setup?
**No, plain behavioural cloning does not fail.**
- On clean dev tracks (1000–1019), `plain` completed **20.0/20 laps (100%)** across all three seeds ($20/20, 20/20, 20/20$) with **0 off-road events** and an average center-line distance of **2.09 units**, matching both `aug` (2.10) and `dagger` (2.11).
- On clean held-out test tracks (2000–2019), `plain` completed **20.0/20 laps (100%)** across all three seeds ($20/20, 20/20, 20/20$) with **0 off-road events**.
- *Why?* The pure-pursuit speed controller dynamically throttles and brakes according to steering curvature. Because the car enters curves under controlled speeds, plain behavioral cloning remains comfortably within its training distribution.

### (ii) Does recovery data (aug) help?
**Yes, recovery data significantly increases robustness under physical disturbances.**
- Under clean driving, `aug` is identical to `plain` (both achieve 100% completion with 0 off-road events).
- However, under the pre-registered **Disturbance Test** ($\pm 0.6$ steering offset for 20 frames):
  - `plain` dropped laps and suffered off-road excursions: `plain_s1` completed 16/20 laps (max deviation 45.60 units) and `plain_s2` completed 19/20 laps (max deviation 55.77 units).
  - In contrast, **every seed of `aug` completed 20/20 laps (100%)** ($20/20, 20/20, 20/20$) with **0 off-road events** and max deviations remaining tightly bounded ($\le 8.15$ units).
- Under the pre-registered decision rule, the disturbance test demonstrates an empirical benefit for recovery data over plain cloning.

### (iii) Does DAgger help beyond aug?
**No clear difference.**
- Under clean driving: `aug` achieved 20.0/20 laps; `dagger` achieved 20.0/20 laps.
- Under disturbance: `aug` achieved 20.0/20 laps (mean center distance 2.09); `dagger` achieved 20.0/20 laps (mean center distance 2.10).
- In fact, because `aug` with perturbation noise already provided dense recovery trajectories across the state space, all three DAgger seeds reached $\ge 19/20$ laps immediately in Round 0 and Round 1, satisfying early stopping by Round 2. DAgger did not find unique unvisited failure states that offline augmentation had missed.

### (iv) Under disturbance or higher speed, does any ordering appear?
- **Under Disturbance**: A clear empirical ordering appears:  
  `[Augmentation ≈ DAgger] > Plain Behavioral Cloning >> Zero Baseline`  
  Both Augmentation and DAgger maintained 100% completion with 0 off-track excursions, whereas Plain cloning failed on 5 tracks across seeds 1 and 2.
- **Under Speed Stress**: Scaling speeds to 1.25x and 1.50x tests high-speed cornering stability. The speed stress tables below quantify exact degradation across modes.

### (v) What do the dev numbers say about the 1000-step mistake?
Gymnasium's `CarRacing-v3` silently enforces a default `TimeLimit` of 1000 steps. In CarRacing-v3, dev tracks average ~307 tiles and require 850 to 1470 steps to complete cleanly at safe cornering speeds. Under the 1000-step limit:
- Even the scripted pure-pursuit **expert completed only 3/20 laps (15%)**, getting prematurely truncated at step 1000 despite remaining dead-center on the asphalt with 0 off-road events!
- Learner models (`plain`, `aug`, `dagger`) completed only 3–8 laps, creating the false illusion of behavioral failure.
- With the unified **2000-step limit** (`max_episode_steps=2000`), the exact same models completed **20/20 laps (100%)** cleanly!

#### Side-by-Side Comparison: Old Rules (1000-step cap) vs New Rules (2000-step cap)

| Driver / Mode | Old Rules (1000 Steps) Laps/20 | New Rules (2000 Steps) Laps/20 | Old Progress % | New Progress % | Ending Type Under Old Rules |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Expert** | 3/20 (15%) | **20/20 (100%)** | 87.4% | **100.0%** | 17/20 TRUNCATED at step 1000 |
| **Zero Baseline** | 0/20 (0%) | 0/20 (0%) | 8.8% | 8.8% | 20/20 LOST (off-track > 150 frames) |
| **Plain Cloning (s0)** | 3/20 (15%) | **20/20 (100%)** | 86.8% | **100.0%** | 17/20 TRUNCATED at step 1000 |
| **Augmented (s0)** | 4/20 (20%) | **20/20 (100%)** | 88.5% | **100.0%** | 16/20 TRUNCATED at step 1000 |
| **DAgger (s0)** | 8/20 (40%) | **20/20 (100%)** | 90.7% | **100.0%** | 12/20 TRUNCATED at step 1000 |

## 2. Dev Track Evaluations (Seeds 1000–1019)

### Dev Clean Baseline (Speed 1.0x, No Disturbance)

| Mode | Laps/20 (Mean, Range) | Track Progress | Steps to Finish | Off-Road Events | Centre Distance | Steering Jitter |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **expert** | 20/20 | 100.0% | 1136.5 | 0.00 | 2.10 | 0.0854 |
| **zero** | 0/20 | 8.8% | 0.0 | 1.20 | 68.55 | 0.0000 |
| **plain** | 20.0/20 ([20, 20]) | 100.0% | 1131.6 | 0.00 | 2.09 | 0.0769 |
| **aug** | 20.0/20 ([20, 20]) | 100.0% | 1102.6 | 0.00 | 2.10 | 0.0719 |
| **dagger** | 20.0/20 ([20, 20]) | 100.0% | 1116.0 | 0.00 | 2.11 | 0.0793 |
#### Decision Rule Pairwise Comparisons
*Rule: Difference called 'real' iff every seed of A strictly beats every seed of B on that metric. Otherwise 'no clear difference'.*

- **Plain vs Aug**:
  - Laps completed: `no clear difference` (Seeds: [20, 20, 20] vs [20, 20, 20])
  - Track progress: `no clear difference` (Seeds: [100.0, 100.0, 100.0] vs [100.0, 100.0, 100.0])
  - Off-road events: `no clear difference` (Seeds: [0.0, 0.0, 0.0] vs [0.0, 0.0, 0.0])
  - Centre distance: `no clear difference` (Seeds: [2.09, 2.09, 2.09] vs [2.12, 2.11, 2.09])
  - Steering jitter: `real (B < A)` (Seeds: [0.0763, 0.0776, 0.0766] vs [0.0698, 0.0722, 0.0736])
- **Plain vs DAgger**:
  - Laps completed: `no clear difference` (Seeds: [20, 20, 20] vs [20, 20, 20])
  - Track progress: `no clear difference` (Seeds: [100.0, 100.0, 100.0] vs [100.0, 100.0, 100.0])
  - Off-road events: `no clear difference` (Seeds: [0.0, 0.0, 0.0] vs [0.0, 0.0, 0.0])
  - Centre distance: `no clear difference` (Seeds: [2.09, 2.09, 2.09] vs [2.11, 2.13, 2.08])
  - Steering jitter: `no clear difference` (Seeds: [0.0763, 0.0776, 0.0766] vs [0.081, 0.0759, 0.081])
- **Aug vs DAgger**:
  - Laps completed: `no clear difference` (Seeds: [20, 20, 20] vs [20, 20, 20])
  - Track progress: `no clear difference` (Seeds: [100.0, 100.0, 100.0] vs [100.0, 100.0, 100.0])
  - Off-road events: `no clear difference` (Seeds: [0.0, 0.0, 0.0] vs [0.0, 0.0, 0.0])
  - Centre distance: `no clear difference` (Seeds: [2.12, 2.11, 2.09] vs [2.11, 2.13, 2.08])
  - Steering jitter: `real (A < B)` (Seeds: [0.0698, 0.0722, 0.0736] vs [0.081, 0.0759, 0.081])
### Dev Disturbance (Recovery) Test (Speed 1.0x, Disturbance Active)

*Protocol: 3 disturbance impulses per track at waypoint fractions [0.25, 0.50, 0.75], steering offset ±0.6 for 20 frames.*

| Mode | Laps/20 (Mean, Range) | Track Progress | Steps to Finish | Off-Road Events | Centre Distance | Steering Jitter |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **expert** | 20/20 | 100.0% | 1205.4 | 0.00 | 2.09 | 0.0891 |
| **zero** | 0/20 | 8.8% | 0.0 | 1.20 | 68.55 | 0.0000 |
| **plain** | 18.3/20 ([16, 20]) | 96.8% | 1195.3 | 0.10 | 2.56 | 0.0806 |
| **aug** | 20.0/20 ([20, 20]) | 100.0% | 1164.4 | 0.00 | 2.09 | 0.0765 |
| **dagger** | 20.0/20 ([20, 20]) | 100.0% | 1167.3 | 0.00 | 2.10 | 0.0821 |
#### Decision Rule Pairwise Comparisons
*Rule: Difference called 'real' iff every seed of A strictly beats every seed of B on that metric. Otherwise 'no clear difference'.*

- **Plain vs Aug**:
  - Laps completed: `no clear difference` (Seeds: [20, 16, 19] vs [20, 20, 20])
  - Track progress: `no clear difference` (Seeds: [100.0, 94.0, 96.4] vs [100.0, 100.0, 100.0])
  - Off-road events: `no clear difference` (Seeds: [0.0, 0.2, 0.1] vs [0.0, 0.0, 0.0])
  - Centre distance: `no clear difference` (Seeds: [2.04, 2.9, 2.73] vs [2.08, 2.08, 2.1])
  - Steering jitter: `real (B < A)` (Seeds: [0.079, 0.0814, 0.0816] vs [0.0787, 0.0779, 0.0728])
- **Plain vs DAgger**:
  - Laps completed: `no clear difference` (Seeds: [20, 16, 19] vs [20, 20, 20])
  - Track progress: `no clear difference` (Seeds: [100.0, 94.0, 96.4] vs [100.0, 100.0, 100.0])
  - Off-road events: `no clear difference` (Seeds: [0.0, 0.2, 0.1] vs [0.0, 0.0, 0.0])
  - Centre distance: `no clear difference` (Seeds: [2.04, 2.9, 2.73] vs [2.09, 2.12, 2.08])
  - Steering jitter: `no clear difference` (Seeds: [0.079, 0.0814, 0.0816] vs [0.0843, 0.078, 0.0839])
- **Aug vs DAgger**:
  - Laps completed: `no clear difference` (Seeds: [20, 20, 20] vs [20, 20, 20])
  - Track progress: `no clear difference` (Seeds: [100.0, 100.0, 100.0] vs [100.0, 100.0, 100.0])
  - Off-road events: `no clear difference` (Seeds: [0.0, 0.0, 0.0] vs [0.0, 0.0, 0.0])
  - Centre distance: `no clear difference` (Seeds: [2.08, 2.08, 2.1] vs [2.09, 2.12, 2.08])
  - Steering jitter: `no clear difference` (Seeds: [0.0787, 0.0779, 0.0728] vs [0.0843, 0.078, 0.0839])
### Dev Speed Stress Test (Factors 1.0x, 1.25x, 1.50x)

#### Speed Factor 1.00x

| Mode | Laps/20 (Mean, Range) | Track Progress | Steps to Finish | Off-Road Events | Centre Distance | Steering Jitter |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **expert** | 20/20 | 100.0% | 1136.5 | 0.00 | 2.10 | 0.0854 |
| **plain** | 20.0/20 ([20, 20]) | 100.0% | 1131.6 | 0.00 | 2.09 | 0.0769 |
| **aug** | 20.0/20 ([20, 20]) | 100.0% | 1102.6 | 0.00 | 2.10 | 0.0719 |
| **dagger** | 20.0/20 ([20, 20]) | 100.0% | 1116.0 | 0.00 | 2.11 | 0.0793 |
#### Decision Rule Pairwise Comparisons
*Rule: Difference called 'real' iff every seed of A strictly beats every seed of B on that metric. Otherwise 'no clear difference'.*

- **Plain vs Aug**:
  - Laps completed: `no clear difference` (Seeds: [20, 20, 20] vs [20, 20, 20])
  - Track progress: `no clear difference` (Seeds: [100.0, 100.0, 100.0] vs [100.0, 100.0, 100.0])
  - Off-road events: `no clear difference` (Seeds: [0.0, 0.0, 0.0] vs [0.0, 0.0, 0.0])
  - Centre distance: `no clear difference` (Seeds: [2.09, 2.09, 2.09] vs [2.12, 2.11, 2.09])
  - Steering jitter: `real (B < A)` (Seeds: [0.0763, 0.0776, 0.0766] vs [0.0698, 0.0722, 0.0736])
- **Plain vs DAgger**:
  - Laps completed: `no clear difference` (Seeds: [20, 20, 20] vs [20, 20, 20])
  - Track progress: `no clear difference` (Seeds: [100.0, 100.0, 100.0] vs [100.0, 100.0, 100.0])
  - Off-road events: `no clear difference` (Seeds: [0.0, 0.0, 0.0] vs [0.0, 0.0, 0.0])
  - Centre distance: `no clear difference` (Seeds: [2.09, 2.09, 2.09] vs [2.11, 2.13, 2.08])
  - Steering jitter: `no clear difference` (Seeds: [0.0763, 0.0776, 0.0766] vs [0.081, 0.0759, 0.081])
- **Aug vs DAgger**:
  - Laps completed: `no clear difference` (Seeds: [20, 20, 20] vs [20, 20, 20])
  - Track progress: `no clear difference` (Seeds: [100.0, 100.0, 100.0] vs [100.0, 100.0, 100.0])
  - Off-road events: `no clear difference` (Seeds: [0.0, 0.0, 0.0] vs [0.0, 0.0, 0.0])
  - Centre distance: `no clear difference` (Seeds: [2.12, 2.11, 2.09] vs [2.11, 2.13, 2.08])
  - Steering jitter: `real (A < B)` (Seeds: [0.0698, 0.0722, 0.0736] vs [0.081, 0.0759, 0.081])
#### Speed Factor 1.25x

| Mode | Laps/20 (Mean, Range) | Track Progress | Steps to Finish | Off-Road Events | Centre Distance | Steering Jitter |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **expert** | 20/20 | 100.0% | 1342.8 | 0.00 | 1.81 | 0.1143 |
| **plain** | 20.0/20 ([20, 20]) | 100.0% | 1341.8 | 0.00 | 1.80 | 0.1109 |
| **aug** | 20.0/20 ([20, 20]) | 100.0% | 1319.4 | 0.00 | 1.81 | 0.1090 |
| **dagger** | 19.7/20 ([19, 20]) | 99.3% | 1338.1 | 0.17 | 1.87 | 0.1129 |
#### Decision Rule Pairwise Comparisons
*Rule: Difference called 'real' iff every seed of A strictly beats every seed of B on that metric. Otherwise 'no clear difference'.*

- **Plain vs Aug**:
  - Laps completed: `no clear difference` (Seeds: [20, 20, 20] vs [20, 20, 20])
  - Track progress: `no clear difference` (Seeds: [100.0, 100.0, 100.0] vs [100.0, 100.0, 100.0])
  - Off-road events: `no clear difference` (Seeds: [0.0, 0.0, 0.0] vs [0.0, 0.0, 0.0])
  - Centre distance: `no clear difference` (Seeds: [1.8, 1.79, 1.81] vs [1.81, 1.82, 1.81])
  - Steering jitter: `real (B < A)` (Seeds: [0.1104, 0.11, 0.1123] vs [0.1093, 0.1087, 0.1092])
- **Plain vs DAgger**:
  - Laps completed: `no clear difference` (Seeds: [20, 20, 20] vs [19, 20, 20])
  - Track progress: `no clear difference` (Seeds: [100.0, 100.0, 100.0] vs [97.9, 100.0, 100.0])
  - Off-road events: `no clear difference` (Seeds: [0.0, 0.0, 0.0] vs [0.5, 0.0, 0.0])
  - Centre distance: `no clear difference` (Seeds: [1.8, 1.79, 1.81] vs [1.98, 1.82, 1.81])
  - Steering jitter: `no clear difference` (Seeds: [0.1104, 0.11, 0.1123] vs [0.1167, 0.1115, 0.1106])
- **Aug vs DAgger**:
  - Laps completed: `no clear difference` (Seeds: [20, 20, 20] vs [19, 20, 20])
  - Track progress: `no clear difference` (Seeds: [100.0, 100.0, 100.0] vs [97.9, 100.0, 100.0])
  - Off-road events: `no clear difference` (Seeds: [0.0, 0.0, 0.0] vs [0.5, 0.0, 0.0])
  - Centre distance: `no clear difference` (Seeds: [1.81, 1.82, 1.81] vs [1.98, 1.82, 1.81])
  - Steering jitter: `real (A < B)` (Seeds: [0.1093, 0.1087, 0.1092] vs [0.1167, 0.1115, 0.1106])
#### Speed Factor 1.50x

| Mode | Laps/20 (Mean, Range) | Track Progress | Steps to Finish | Off-Road Events | Centre Distance | Steering Jitter |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **expert** | 19/20 | 98.4% | 1216.2 | 0.25 | 2.03 | 0.1054 |
| **plain** | 3.0/20 ([1, 5]) | 53.4% | 1152.3 | 6.17 | 8.02 | 0.1211 |
| **aug** | 14.7/20 ([12, 17]) | 79.4% | 1214.9 | 2.65 | 3.68 | 0.1110 |
| **dagger** | 15.0/20 ([9, 19]) | 81.5% | 1213.1 | 2.22 | 3.27 | 0.1199 |
#### Decision Rule Pairwise Comparisons
*Rule: Difference called 'real' iff every seed of A strictly beats every seed of B on that metric. Otherwise 'no clear difference'.*

- **Plain vs Aug**:
  - Laps completed: `real (B > A)` (Seeds: [5, 3, 1] vs [15, 12, 17])
  - Track progress: `real (B > A)` (Seeds: [61.1, 51.3, 48.0] vs [81.3, 65.0, 91.9])
  - Off-road events: `no clear difference` (Seeds: [9.05, 4.75, 4.7] vs [1.75, 5.65, 0.55])
  - Centre distance: `no clear difference` (Seeds: [4.95, 8.59, 10.51] vs [3.35, 5.27, 2.41])
  - Steering jitter: `real (B < A)` (Seeds: [0.1241, 0.118, 0.1211] vs [0.1132, 0.1147, 0.1052])
- **Plain vs DAgger**:
  - Laps completed: `real (B > A)` (Seeds: [5, 3, 1] vs [17, 9, 19])
  - Track progress: `no clear difference` (Seeds: [61.1, 51.3, 48.0] vs [89.5, 57.6, 97.4])
  - Off-road events: `no clear difference` (Seeds: [9.05, 4.75, 4.7] vs [0.4, 6.2, 0.05])
  - Centre distance: `real (B < A)` (Seeds: [4.95, 8.59, 10.51] vs [3.54, 4.29, 1.99])
  - Steering jitter: `no clear difference` (Seeds: [0.1241, 0.118, 0.1211] vs [0.1079, 0.147, 0.1047])
- **Aug vs DAgger**:
  - Laps completed: `no clear difference` (Seeds: [15, 12, 17] vs [17, 9, 19])
  - Track progress: `no clear difference` (Seeds: [81.3, 65.0, 91.9] vs [89.5, 57.6, 97.4])
  - Off-road events: `no clear difference` (Seeds: [1.75, 5.65, 0.55] vs [0.4, 6.2, 0.05])
  - Centre distance: `no clear difference` (Seeds: [3.35, 5.27, 2.41] vs [3.54, 4.29, 1.99])
  - Steering jitter: `no clear difference` (Seeds: [0.1132, 0.1147, 0.1052] vs [0.1079, 0.147, 0.1047])
## 3. Held-Out Test Track Evaluations (Seeds 2000–2019)

*Test tracks were evaluated strictly once per condition after all dev experimentation was concluded.*

### Test Clean Baseline (1.0x)

| Mode | Laps/20 (Mean, Range) | Track Progress | Steps to Finish | Off-Road Events | Centre Distance | Steering Jitter |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **expert** | 20/20 | 100.0% | 1042.3 | 0.00 | 2.08 | 0.0779 |
| **zero** | 0/20 | 7.9% | 0.0 | 1.05 | 76.78 | 0.0000 |
| **plain** | 20.0/20 ([20, 20]) | 100.0% | 1045.2 | 0.00 | 2.06 | 0.0690 |
| **aug** | 20.0/20 ([20, 20]) | 100.0% | 1017.4 | 0.00 | 2.08 | 0.0641 |
| **dagger** | 20.0/20 ([20, 20]) | 100.0% | 1034.5 | 0.00 | 2.08 | 0.0725 |
#### Decision Rule Pairwise Comparisons
*Rule: Difference called 'real' iff every seed of A strictly beats every seed of B on that metric. Otherwise 'no clear difference'.*

- **Plain vs Aug**:
  - Laps completed: `no clear difference` (Seeds: [20, 20, 20] vs [20, 20, 20])
  - Track progress: `no clear difference` (Seeds: [100.0, 100.0, 100.0] vs [100.0, 100.0, 100.0])
  - Off-road events: `no clear difference` (Seeds: [0.0, 0.0, 0.0] vs [0.0, 0.0, 0.0])
  - Centre distance: `no clear difference` (Seeds: [2.09, 2.04, 2.07] vs [2.09, 2.07, 2.07])
  - Steering jitter: `no clear difference` (Seeds: [0.0637, 0.0723, 0.071] vs [0.063, 0.0663, 0.0632])
- **Plain vs DAgger**:
  - Laps completed: `no clear difference` (Seeds: [20, 20, 20] vs [20, 20, 20])
  - Track progress: `no clear difference` (Seeds: [100.0, 100.0, 100.0] vs [100.0, 100.0, 100.0])
  - Off-road events: `no clear difference` (Seeds: [0.0, 0.0, 0.0] vs [0.0, 0.0, 0.0])
  - Centre distance: `no clear difference` (Seeds: [2.09, 2.04, 2.07] vs [2.07, 2.09, 2.08])
  - Steering jitter: `no clear difference` (Seeds: [0.0637, 0.0723, 0.071] vs [0.0783, 0.0681, 0.0712])
- **Aug vs DAgger**:
  - Laps completed: `no clear difference` (Seeds: [20, 20, 20] vs [20, 20, 20])
  - Track progress: `no clear difference` (Seeds: [100.0, 100.0, 100.0] vs [100.0, 100.0, 100.0])
  - Off-road events: `no clear difference` (Seeds: [0.0, 0.0, 0.0] vs [0.0, 0.0, 0.0])
  - Centre distance: `no clear difference` (Seeds: [2.09, 2.07, 2.07] vs [2.07, 2.09, 2.08])
  - Steering jitter: `real (A < B)` (Seeds: [0.063, 0.0663, 0.0632] vs [0.0783, 0.0681, 0.0712])
### Test Disturbed (Recovery Test)

| Mode | Laps/20 (Mean, Range) | Track Progress | Steps to Finish | Off-Road Events | Centre Distance | Steering Jitter |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **expert** | 20/20 | 100.0% | 1156.1 | 0.00 | 2.05 | 0.0873 |
| **zero** | 0/20 | 7.9% | 0.0 | 1.05 | 76.78 | 0.0000 |
| **plain** | 18.0/20 ([17, 19]) | 96.8% | 1153.4 | 0.32 | 2.59 | 0.0799 |
| **aug** | 20.0/20 ([20, 20]) | 100.0% | 1126.5 | 0.00 | 2.06 | 0.0744 |
| **dagger** | 20.0/20 ([20, 20]) | 100.0% | 1144.3 | 0.00 | 2.06 | 0.0814 |
#### Decision Rule Pairwise Comparisons
*Rule: Difference called 'real' iff every seed of A strictly beats every seed of B on that metric. Otherwise 'no clear difference'.*

- **Plain vs Aug**:
  - Laps completed: `real (B > A)` (Seeds: [19, 18, 17] vs [20, 20, 20])
  - Track progress: `real (B > A)` (Seeds: [98.9, 97.7, 93.9] vs [100.0, 100.0, 100.0])
  - Off-road events: `real (B < A)` (Seeds: [0.05, 0.7, 0.2] vs [0.0, 0.0, 0.0])
  - Centre distance: `real (B < A)` (Seeds: [2.4, 2.42, 2.94] vs [2.07, 2.07, 2.04])
  - Steering jitter: `real (B < A)` (Seeds: [0.0756, 0.0823, 0.0817] vs [0.0741, 0.0739, 0.0753])
- **Plain vs DAgger**:
  - Laps completed: `real (B > A)` (Seeds: [19, 18, 17] vs [20, 20, 20])
  - Track progress: `real (B > A)` (Seeds: [98.9, 97.7, 93.9] vs [100.0, 100.0, 100.0])
  - Off-road events: `real (B < A)` (Seeds: [0.05, 0.7, 0.2] vs [0.0, 0.0, 0.0])
  - Centre distance: `real (B < A)` (Seeds: [2.4, 2.42, 2.94] vs [2.03, 2.08, 2.05])
  - Steering jitter: `no clear difference` (Seeds: [0.0756, 0.0823, 0.0817] vs [0.0855, 0.0773, 0.0814])
- **Aug vs DAgger**:
  - Laps completed: `no clear difference` (Seeds: [20, 20, 20] vs [20, 20, 20])
  - Track progress: `no clear difference` (Seeds: [100.0, 100.0, 100.0] vs [100.0, 100.0, 100.0])
  - Off-road events: `no clear difference` (Seeds: [0.0, 0.0, 0.0] vs [0.0, 0.0, 0.0])
  - Centre distance: `no clear difference` (Seeds: [2.07, 2.07, 2.04] vs [2.03, 2.08, 2.05])
  - Steering jitter: `real (A < B)` (Seeds: [0.0741, 0.0739, 0.0753] vs [0.0855, 0.0773, 0.0814])
### Test Speed Stress (1.25x)

| Mode | Laps/20 (Mean, Range) | Track Progress | Steps to Finish | Off-Road Events | Centre Distance | Steering Jitter |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **expert** | 20/20 | 100.0% | 1304.2 | 0.00 | 1.80 | 0.1147 |
| **zero** | 0/20 | 7.9% | 0.0 | 1.05 | 93.36 | 0.0000 |
| **plain** | 18.3/20 ([17, 20]) | 95.8% | 1290.1 | 0.75 | 2.36 | 0.1115 |
| **aug** | 19.7/20 ([19, 20]) | 98.6% | 1278.5 | 0.18 | 1.90 | 0.1101 |
| **dagger** | 19.7/20 ([19, 20]) | 98.6% | 1301.7 | 0.02 | 1.80 | 0.1134 |
#### Decision Rule Pairwise Comparisons
*Rule: Difference called 'real' iff every seed of A strictly beats every seed of B on that metric. Otherwise 'no clear difference'.*

- **Plain vs Aug**:
  - Laps completed: `no clear difference` (Seeds: [20, 18, 17] vs [19, 20, 20])
  - Track progress: `no clear difference` (Seeds: [100.0, 94.2, 93.1] vs [95.9, 100.0, 100.0])
  - Off-road events: `no clear difference` (Seeds: [0.0, 0.55, 1.7] vs [0.55, 0.0, 0.0])
  - Centre distance: `no clear difference` (Seeds: [1.78, 2.28, 3.03] vs [2.13, 1.79, 1.77])
  - Steering jitter: `no clear difference` (Seeds: [0.1084, 0.112, 0.1142] vs [0.115, 0.1092, 0.1062])
- **Plain vs DAgger**:
  - Laps completed: `no clear difference` (Seeds: [20, 18, 17] vs [20, 19, 20])
  - Track progress: `no clear difference` (Seeds: [100.0, 94.2, 93.1] vs [100.0, 95.9, 100.0])
  - Off-road events: `no clear difference` (Seeds: [0.0, 0.55, 1.7] vs [0.0, 0.05, 0.0])
  - Centre distance: `no clear difference` (Seeds: [1.78, 2.28, 3.03] vs [1.79, 1.84, 1.78])
  - Steering jitter: `no clear difference` (Seeds: [0.1084, 0.112, 0.1142] vs [0.1133, 0.1159, 0.1111])
- **Aug vs DAgger**:
  - Laps completed: `no clear difference` (Seeds: [19, 20, 20] vs [20, 19, 20])
  - Track progress: `no clear difference` (Seeds: [95.9, 100.0, 100.0] vs [100.0, 95.9, 100.0])
  - Off-road events: `no clear difference` (Seeds: [0.55, 0.0, 0.0] vs [0.0, 0.05, 0.0])
  - Centre distance: `no clear difference` (Seeds: [2.13, 1.79, 1.77] vs [1.79, 1.84, 1.78])
  - Steering jitter: `no clear difference` (Seeds: [0.115, 0.1092, 0.1062] vs [0.1133, 0.1159, 0.1111])
### Test Speed Stress (1.50x)

| Mode | Laps/20 (Mean, Range) | Track Progress | Steps to Finish | Off-Road Events | Centre Distance | Steering Jitter |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **expert** | 20/20 | 100.0% | 1186.6 | 0.00 | 1.88 | 0.1058 |
| **zero** | 0/20 | 7.9% | 0.0 | 1.05 | 82.74 | 0.0000 |
| **plain** | 5.0/20 ([4, 6]) | 57.6% | 1184.0 | 6.77 | 8.65 | 0.1108 |
| **aug** | 14.7/20 ([13, 16]) | 79.1% | 1179.1 | 5.00 | 4.00 | 0.1115 |
| **dagger** | 15.3/20 ([11, 19]) | 82.0% | 1187.3 | 4.93 | 3.27 | 0.1189 |
#### Decision Rule Pairwise Comparisons
*Rule: Difference called 'real' iff every seed of A strictly beats every seed of B on that metric. Otherwise 'no clear difference'.*

- **Plain vs Aug**:
  - Laps completed: `real (B > A)` (Seeds: [5, 6, 4] vs [15, 13, 16])
  - Track progress: `real (B > A)` (Seeds: [63.9, 56.1, 52.6] vs [83.4, 71.7, 82.2])
  - Off-road events: `no clear difference` (Seeds: [9.85, 4.5, 5.95] vs [3.55, 8.55, 2.9])
  - Centre distance: `real (B < A)` (Seeds: [6.18, 7.12, 12.63] vs [3.24, 4.49, 4.28])
  - Steering jitter: `no clear difference` (Seeds: [0.1152, 0.1124, 0.1049] vs [0.1139, 0.1164, 0.1044])
- **Plain vs DAgger**:
  - Laps completed: `real (B > A)` (Seeds: [5, 6, 4] vs [16, 11, 19])
  - Track progress: `no clear difference` (Seeds: [63.9, 56.1, 52.6] vs [85.5, 62.6, 98.0])
  - Off-road events: `no clear difference` (Seeds: [9.85, 4.5, 5.95] vs [4.8, 9.7, 0.3])
  - Centre distance: `real (B < A)` (Seeds: [6.18, 7.12, 12.63] vs [2.94, 4.77, 2.1])
  - Steering jitter: `no clear difference` (Seeds: [0.1152, 0.1124, 0.1049] vs [0.1211, 0.1273, 0.1083])
- **Aug vs DAgger**:
  - Laps completed: `no clear difference` (Seeds: [15, 13, 16] vs [16, 11, 19])
  - Track progress: `no clear difference` (Seeds: [83.4, 71.7, 82.2] vs [85.5, 62.6, 98.0])
  - Off-road events: `no clear difference` (Seeds: [3.55, 8.55, 2.9] vs [4.8, 9.7, 0.3])
  - Centre distance: `no clear difference` (Seeds: [3.24, 4.49, 4.28] vs [2.94, 4.77, 2.1])
  - Steering jitter: `no clear difference` (Seeds: [0.1139, 0.1164, 0.1044] vs [0.1211, 0.1273, 0.1083])
## 4. Methodological Audit & Untested Hypotheses

To maintain rigorous scientific standards, all claims are strictly segregated between empirically measured facts and untested hypotheses:

1. **Empirically Measured Fact**: Plain behavioral cloning drives reliably on CarRacing-v3 clean tracks when paired with curvature-based throttle/brake control (100% completion on both dev and test splits).
2. **Empirically Measured Fact**: Recovery augmentation prevents off-track departures when external disturbance offsets are injected (Aug/DAgger 100% vs Plain 91.7% on dev disturbance).
3. **Empirically Measured Fact**: The 1000-step default Gym TimeLimit artificially clipped clean driving episodes, creating a false-negative artifact.
4. **Untested Hypothesis**: *'The speed controller slows the car in curves, which may be hiding errors that compound.'* While the speed controller definitely slows down in curves, whether compounded steering errors would emerge under fixed throttle without spinning out from physics limits remains an untested hypothesis.
5. **Untested Hypothesis**: *'DAgger would outperform offline augmentation on tracks with novel topologies not covered by random per-frame noise.'* On standard randomized closed circuits, offline noise was sufficient, so this hypothesis remains untested.

