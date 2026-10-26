# Sporting Regulations & Race Control Rules

This document outlines the rules engine and stewarding adjudication system implemented in `f1sim/stewards.py`.

---

## 1. Track Limits & Excursions

1. **Definition of Track Limits**:
   - The playable racing surface is defined by the tarmac surface (`uw.track`).
   - A vehicle is considered to have exceeded track limits when all four wheels leave the tarmac surface (detected via `wheels_on_track == 0`).
2. **Warnings Hierarchy**:
   - **Excursions 1 & 2**: Official warnings logged by Race Control.
   - **Excursion 3**: **Black and White Flag** displayed as the final warning for unsporting track limits infringements.
   - **Excursion 4**: **5-Second Time Penalty** added to the driver's elapsed race time.
   - **Excursions 5+**: Additional **10-Second Time Penalties** for persistent non-compliance.

---

## 2. Retirement & DNF (Did Not Finish)

- **150-Frame Off-Track Threshold**:
  - If a vehicle remains off the tarmac for **150 consecutive frames** (3.0 seconds at 50 FPS) without regaining control or re-entering the circuit, the vehicle is classified as **Retired / DNF (Lost)**.
  - This prevents unbounded drifting across the playfield and mirrors the experimental protocol defined in `results/protocol.md`.

---

## 3. Start Procedure & False Starts

1. **Grid Formation**:
   - Vehicles form on a staggered, 2-wide grid behind the start/finish line.
   - Pole position (P1) occupies the racing line side; P2 occupies the opposite side spaced 6 meters behind; P3 and P4 follow sequentially.
2. **Jump Start Detection**:
   - Vehicle movement exceeding **1.0 m/s** prior to the red lights extinguishing ("Lights Out") is classified as a False Start.
   - **Sanction**: Automatic 5-second time penalty.

---

## 4. Lap Timing & Validity

1. **Checkpoints & Micro-Sectors**:
   - Each circuit is divided into **100 timing checkpoints** and **3 primary sectors (S1, S2, S3)**.
2. **Sector Timing Colors**:
   - 🟣 **Purple**: Fastest sector recorded across all drivers in the session.
   - 🟢 **Green**: Personal best sector for that driver.
   - 🟡 **Yellow**: Slower than the driver's personal best sector.
3. **Finish Line Detection**:
   - Uses the idempotent contact listener patch (`f1sim.env.apply_finish_line_patch`).
   - Only triggers a completed lap when tile visited percentage $> 95\%$ and the car is inside the playfield boundaries.
