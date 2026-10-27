"""
f1sim/world.py - High-Performance Shared-World Multi-Car Box2D Simulation Engine.

Key Architecture:
- ONE Gymnasium CarRacing-v3 environment, ONE Box2D world instance.
- Car 0 is env.unwrapped.car; additional cars (1..N-1) are car_dynamics.Car in the SAME world.
- True Box2D rigid-body hull-hull contacts and collision dynamics.
- Real 2-wide staggered grid placement before lights-out.
- Per-car observations via temporary uw.car viewport swapping with env.t zoom-in handling.
- Windowed track progress tracking decoupled from global Gym tile flags.
- Comprehensive contact stewards (rear-end vs racing incidents) and clean deceleration after finish.
- Replay JSON export fully compatible with the 3D WebGL Replay Viewer.
"""

import math
import os
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import gymnasium as gym
from gymnasium.envs.box2d import car_racing, car_dynamics

from f1sim.drivers import (
    Driver,
    ExpertDriver,
    ModelDriver,
    TEAMS,
    compute_staggered_grid_positions,
    get_car_state,
    speed_controller,
)
from f1sim.env import make_env
from f1sim.metrics import compute_progress_pct
from f1sim.stewards import Stewards, RulesConfig
from f1sim.timing import TimingSystem
from f1sim.track import WindowedProgressTracker, extract_sectors


FPS = 50.0
TRACK_WIDTH = car_racing.TRACK_WIDTH  # 40 / 6 = 6.6667
FORBIDDEN_SEEDS = set(range(1000, 1020)) | set(range(2000, 2020)) | set(range(5000, 5010))


def validate_seed(seed: int):
    """
    Ensure seed stays out of benchmark sets (1000-1019, 2000-2019, 5000-5009).
    Allows showcase range [3000, 3019] or random seeds >= 6000.
    """
    assert seed not in FORBIDDEN_SEEDS, (
        f"Seed {seed} is strictly reserved for training/evaluation benchmarks."
    )
    assert (3000 <= seed <= 3019) or (seed >= 6000), (
        f"Seed must be in showcase range [3000, 3019] or >= 6000, got {seed}"
    )


class CarHullUserData:
    """
    Custom user-data attached to Box2D car hull bodies.
    Must be a class instance with __dict__ so Gymnasium FrictionDetector does not crash.
    """
    def __init__(self, car_id: str, car_obj: car_dynamics.Car):
        self.car_id = car_id
        self.car = car_obj
        self.is_car_hull = True


@dataclass
class ContactRecord:
    step: int
    time_s: float
    car_a: str
    car_b: str
    at_fault: Optional[str]
    relative_speed: float
    incident_type: str  # "rear_end" (+5s) or "racing_incident"


SIX_CAR_SPECS = [
    {
        "id": "car_apx",
        "team": "APX",
        "code": "VAN",
        "name": "Alex Vance",
        "type": "model",
        "model_file": "models/dagger_s0.keras",
        "number": 1,
        "grid_pos": 1,
    },
    {
        "id": "car_vel",
        "team": "VEL",
        "code": "SIL",
        "name": "Marco Silva",
        "type": "model",
        "model_file": "models/plain_s0.keras",
        "number": 16,
        "grid_pos": 2,
    },
    {
        "id": "car_ndx",
        "team": "NDX",
        "code": "LIN",
        "name": "Lukas Lind",
        "type": "model",
        "model_file": "models/aug_s0.keras",
        "number": 23,
        "grid_pos": 3,
    },
    {
        "id": "car_zen",
        "team": "ZEN",
        "code": "TAN",
        "name": "Kai Tanaka",
        "type": "expert",
        "number": 88,
        "grid_pos": 4,
    },
    {
        "id": "car_rdr",
        "team": "RDR",
        "code": "OCO",
        "name": "Liam O'Connor",
        "type": "model",
        "model_file": "models/dagger_s1.keras",
        "number": 11,
        "grid_pos": 5,
    },
    {
        "id": "car_orb",
        "team": "ORB",
        "code": "ROS",
        "name": "Elena Rostov",
        "type": "model",
        "model_file": "models/aug_s1.keras",
        "number": 77,
        "grid_pos": 6,
    },
]


class MultiCarWorld:
    """
    Synchronized single-world multi-car Box2D simulation engine.
    """

    def __init__(
        self,
        track_seed: int = 3003,
        total_laps: int = 10,
        car_specs: Optional[List[Dict[str, Any]]] = None,
        traffic_mode: str = "contact",  # "contact" or "follow"
        fps: float = 50.0,
        stewards_config: Optional[RulesConfig] = None,
        lights_hold_steps: int = 50,
        pace_variance: float = 0.0,
    ):
        validate_seed(track_seed)
        self.track_seed = track_seed
        self.total_laps = total_laps
        self.traffic_mode = traffic_mode
        self.fps = fps
        self.specs = car_specs or SIX_CAR_SPECS
        self.num_cars = len(self.specs)
        
        # RNG for deterministic variety (lights hold and pace variance)
        rng = random.Random(track_seed)
        if lights_hold_steps is None or lights_hold_steps <= 0:
            hold_sec = rng.uniform(0.5, 3.0)
            self.lights_hold_steps = int(hold_sec * fps)
        else:
            self.lights_hold_steps = lights_hold_steps

        self.pace_variance = pace_variance
        self.pace_multipliers: Dict[str, float] = {}
        for spec in self.specs:
            cid = spec["id"]
            if pace_variance > 0.0:
                delta = rng.uniform(-pace_variance, pace_variance)
            else:
                delta = 0.0
            self.pace_multipliers[cid] = 1.0 + delta

        # Systems
        self.timing = TimingSystem(fps=int(fps), num_checkpoints=100)
        self.stewards = Stewards(config=stewards_config, fps=int(fps))

        # Core state
        self.env = None
        self.uw = None
        self.world = None
        self.cars: Dict[str, car_dynamics.Car] = {}
        self.drivers: Dict[str, Driver] = {}
        self.trackers: Dict[str, WindowedProgressTracker] = {}
        self.driver_meta: Dict[str, Dict[str, Any]] = {}
        self.grid_boxes: List[Dict[str, Any]] = []

        # Lifecycle
        self.current_step = 0
        self.winner_id: Optional[str] = None
        self.winner_finish_step: int = -1
        self.last_finisher_step: int = -1
        self.session_end_step: int = -1
        self.race_state = "GRID"  # GRID, RACING, CHEQUERED, FINISHING, COOLDOWN, RESULTS

        # Contact monitoring
        self.contacts_log: List[ContactRecord] = []
        self._raw_contacts_this_step: List[Tuple[str, str]] = []
        self._last_contact_step: Dict[Tuple[str, str], int] = {}

        # Trajectories for replay export
        self.trajectories: Dict[str, Dict[str, list]] = {}
        self.race_events: List[Dict[str, Any]] = []

        # Previous states for yaw-rate & acceleration calculation
        self._prev_heading: Dict[str, float] = {}

        # Phase 3: Lifecycle and classification tracking
        self.finished_cars = set()
        self._lapped_announced = set()
        self.stopped_frames = {spec["id"]: 0 for spec in self.specs}
        self.timeout_seconds = 120.0

        self._init_simulation()

    def _init_simulation(self):
        """Construct the shared environment, cars on grid, and collision listeners."""
        # 1. Create single underlying Gym environment
        max_possible_steps = max(20000, self.total_laps * 2000)
        self.env = make_env(
            seed=self.track_seed,
            max_steps=max_possible_steps,
            render_mode="rgb_array",
            race_mode=True,
        )
        self.env.reset(seed=self.track_seed)
        self.uw = self.env.unwrapped
        self.world = self.uw.world
        self.track = self.uw.track
        self.track_pts = np.array([[p[2], p[3]] for p in self.track])

        # 2. Extract staggered starting grid
        self.grid_data = compute_staggered_grid_positions(
            self.track_pts,
            grid_slots=self.num_cars,
            spacing_meters=6.5,
            lateral_offset=1.75,
        )

        # 3. Instantiate drivers, cars, trackers, and telemetry
        for i, spec in enumerate(self.specs):
            cid = spec["id"]
            code = spec["code"]
            name = spec["name"]
            team = spec["team"]
            self.driver_meta[cid] = spec

            self.timing.register_driver(cid, code, name, team)
            self.stewards.register_car(cid, code)
            self.trackers[cid] = WindowedProgressTracker(self.track)

            if spec.get("type") == "expert":
                driver = ExpertDriver(name=name, code=code, team=team)
            else:
                driver = ModelDriver(spec["model_file"], name=name, code=code, team=team)
            self.drivers[cid] = driver

            # Calculate physical grid position & angle
            # Box2D heading convention: hull.angle = heading - pi/2 (COORDS.md)
            g_pos = self.grid_data[i]["pos"]
            g_heading = self.grid_data[i]["heading"]
            hull_angle = g_heading - math.pi / 2.0

            if i == 0:
                car = self.uw.car
                car.hull.position = (float(g_pos[0]), float(g_pos[1]))
                car.hull.angle = float(hull_angle)
            else:
                car = car_dynamics.Car(
                    self.world,
                    float(hull_angle),
                    float(g_pos[0]),
                    float(g_pos[1]),
                )

            # Assign livery color: solo car keeps standard Gym CarRacing red for bit-identity
            if self.num_cars == 1:
                car.hull.color = [0.8, 0.0, 0.0]
            else:
                team_info = TEAMS.get(team, {})
                hex_color = team_info.get("hex", "#FFFFFF")
                rgb_norm = [
                    int(hex_color[1:3], 16) / 255.0,
                    int(hex_color[3:5], 16) / 255.0,
                    int(hex_color[5:7], 16) / 255.0,
                ]
                car.hull.color = rgb_norm
            car.hull.userData = CarHullUserData(cid, car)
            self.cars[cid] = car

            # Initialize trajectory recording
            self.trajectories[cid] = {
                "x": [],
                "y": [],
                "heading": [],
                "hull_angle": [],
                "speed": [],
                "steer": [],
                "gas": [],
                "brake": [],
                "wheels": [],
                "progress": [],
                "lap": [],
                "offroad": [],
                "centre_dist": [],
                "lateral_g": [],
            }
            self._prev_heading[cid] = g_heading

        # 5. Intercept Box2D ContactListener for car-car collisions
        self._setup_collision_listener()

        # 6. Pre-start race events
        self.race_events = [
            {"step": 0, "time": 0.0, "type": "lights_on", "text": "RED LIGHTS ILLUMINATING"},
            {"step": 20, "time": 0.4, "type": "lights_seq", "text": "FORMATION GRID SET"},
            {"step": self.lights_hold_steps, "time": round(self.lights_hold_steps / self.fps, 2),
             "type": "lights_out", "text": "LIGHTS OUT AND AWAY WE GO!"},
        ]

    def _setup_checkered_finish_line(self):
        """Build checkered line on road_poly[0] for rendering authenticity."""
        if not hasattr(self.uw, "road_poly") or len(self.uw.road_poly) == 0:
            return
        poly0 = self.uw.road_poly[0][0]
        if len(poly0) < 4:
            return
        p1_l, p1_r, p2_r, p2_l = [np.array(p) for p in poly0[:4]]
        checkered_polys = []
        n_cols, n_rows = 12, 4
        for r in range(n_rows):
            for c in range(n_cols):
                u0, u1 = c / n_cols, (c + 1) / n_cols
                v0, v1 = r / n_rows, (r + 1) / n_rows
                pt_tl = (1 - v1) * ((1 - u0) * p2_l + u0 * p2_r) + v1 * ((1 - u0) * p1_l + u0 * p1_r)
                pt_tr = (1 - v1) * ((1 - u1) * p2_l + u1 * p2_r) + v1 * ((1 - u1) * p1_l + u1 * p1_r)
                pt_br = (1 - v0) * ((1 - u1) * p2_l + u1 * p2_r) + v0 * ((1 - u1) * p1_l + u1 * p1_r)
                pt_bl = (1 - v0) * ((1 - u0) * p2_l + u0 * p2_r) + v0 * ((1 - u0) * p1_l + u0 * p1_r)
                color = (0.95, 0.95, 0.95) if (r + c) % 2 == 0 else (0.1, 0.1, 0.1)
                checkered_polys.append(([pt_tl.tolist(), pt_tr.tolist(), pt_br.tolist(), pt_bl.tolist()], color))

        # Checkered start/finish line replacement
        self.uw.road_poly = checkered_polys + self.uw.road_poly[1:]

    def _setup_collision_listener(self):
        """Wrap Gymnasium contactListener to register hull-hull collisions."""
        orig_begin = self.world.contactListener.BeginContact

        def shared_world_begin_contact(contact):
            orig_begin(contact)
            bA = contact.fixtureA.body
            bB = contact.fixtureB.body
            u1 = bA.userData
            u2 = bB.userData
            if getattr(u1, "is_car_hull", False) and getattr(u2, "is_car_hull", False):
                self._raw_contacts_this_step.append((u1.car_id, u2.car_id))

        self.world.contactListener.BeginContact = shared_world_begin_contact

    def _get_car_observation(self, car: car_dynamics.Car) -> np.ndarray:
        """Render 96x96 viewport centered on the specified car."""
        orig_car = self.uw.car
        self.uw.car = car
        obs = self.uw._render("state_pixels")
        self.uw.car = orig_car
        return obs

    def step(self) -> bool:
        """
        Execute one physical simulation frame across all cars in the shared world.
        Returns False when the race session has reached its full conclusion.
        """
        step = self.current_step
        self.current_step += 1
        self._raw_contacts_this_step.clear()

        is_pre_start = step < self.lights_hold_steps

        # 1. State machine update
        if is_pre_start:
            self.race_state = "GRID"
        elif self.winner_id is None:
            self.race_state = "RACING"
        elif self.session_end_step < 0:
            self.race_state = "FINISHING"
        else:
            self.race_state = "COOLDOWN"

        # 2. Collect actions for each car
        actions: Dict[str, np.ndarray] = {}
        for i, (cid, car) in enumerate(self.cars.items()):
            timing_driver = self.timing.drivers[cid]
            steward_state = self.stewards.states[cid]
            has_finished = timing_driver.laps_completed >= self.total_laps
            is_dnf = steward_state.is_dnf

            x, y, heading, speed = get_car_state(car)

            if is_pre_start:
                # Pre-start: holding brake on starting grid
                actions[cid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            elif has_finished or is_dnf:
                # Post-race cool down: smooth deceleration, never freeze in place
                # Follow gentle track curve with engine braking to come to a stop
                steer = 0.0
                gas = 0.0
                brake = float(min(0.4, max(0.1, speed * 0.04)))
                actions[cid] = np.array([steer, gas, brake], dtype=np.float32)
            else:
                # Active racing driver
                driver = self.drivers[cid]
                if isinstance(driver, ModelDriver):
                    obs = self._get_car_observation(car)
                    act = driver.act(obs, self.env, step=step)
                else:
                    # Expert Driver
                    act = driver.act(None, self.env, step=step)

                # Traffic follow mode: avoid ramming car directly ahead
                if self.traffic_mode == "follow":
                    act = self._apply_follow_logic(cid, car, act)

                actions[cid] = act

        # 3. Apply control inputs and step each car dynamics
        for cid, car in self.cars.items():
            act = actions[cid]
            car.steer(-float(act[0]))
            pace_mult = self.pace_multipliers.get(cid, 1.0)
            car.gas(float(np.clip(act[1] * pace_mult, 0.0, 1.0)))
            car.brake(float(act[2]))
            car.step(1.0 / self.fps)

        # Pre-collision kinematics before Box2D solver impulse transfer
        pre_states = {cid: get_car_state(car) for cid, car in self.cars.items()}

        # 4. Advance Box2D physics simulation for the entire world ONCE
        self.world.Step(1.0 / self.fps, 6 * 30, 2 * 30)
        self.uw.t += 1.0 / self.fps

        # 5. Process car-car contacts from Box2D contact listener
        self._process_contacts(step, pre_states)

        # 6. Update tracking, timing, and stewards per car
        classified_finishers = []
        for i, (cid, car) in enumerate(self.cars.items()):
            timing_driver = self.timing.drivers[cid]
            steward_state = self.stewards.states[cid]
            x, y, heading, speed = get_car_state(car)
            hull_angle = float(car.hull.angle)
            ang_vel = float(car.hull.angularVelocity)

            # Wheels track limits (FL, FR, RL, RR)
            wheel_flags = [bool(len(w.tiles) > 0) for w in car.wheels]
            on_track_wheels = sum(wheel_flags)
            is_offroad = (on_track_wheels == 0)

            # Update windowed track progress
            tracker = self.trackers[cid]
            wp_idx, norm_dist, lap_finished = tracker.update(x, y)
            progress_pct = tracker.progress_pct

            # Stewards check (offtrack warnings, 150-frame DNF rule)
            if not is_pre_start and not steward_state.is_dnf:
                self.stewards.check_step(step, cid, is_offroad, speed)

            # Timing check
            if not is_pre_start and not steward_state.is_dnf:
                self.timing.update_driver_progress(
                    cid,
                    current_step=step,
                    progress_pct=progress_pct,
                    lap_finished=lap_finished,
                )

            # Centre-line distance
            diff = self.track_pts - np.array([x, y])
            cdist = float(np.sqrt(np.min(np.sum(diff ** 2, axis=1))))

            # Lateral acceleration: speed * |yaw_rate| / 9.81 (estimated G)
            lat_g = round(float((speed * abs(ang_vel)) / 9.81), 2)

            # Record finish timestamps: winner or lapped car crossing finish line
            is_winner_crossing = (timing_driver.laps_completed >= self.total_laps) and (self.winner_id is None)
            is_lapped_finish = (self.winner_id is not None) and lap_finished and (cid not in self.finished_cars)

            if is_winner_crossing or is_lapped_finish or (timing_driver.laps_completed >= self.total_laps and cid not in self.finished_cars):
                self.finished_cars.add(cid)
                if self.winner_id is None:
                    self.winner_id = cid
                    self.winner_finish_step = step
                    self.race_state = "CHEQUERED"
                    self.race_events.append({
                        "step": step,
                        "time": round(step / self.fps, 2),
                        "type": "chequered_flag",
                        "text": f"CHEQUERED FLAG: {timing_driver.code} WINS THE GRAND PRIX!",
                        "car_id": cid,
                    })
                elif is_lapped_finish and cid not in self._lapped_announced:
                    self._lapped_announced.add(cid)
                    laps_down = max(1, self.total_laps - timing_driver.laps_completed)
                    lap_str = f"+{laps_down} LAP{'S' if laps_down > 1 else ''}"
                    self.race_events.append({
                        "step": step,
                        "time": round(step / self.fps, 2),
                        "type": "chequered_flag",
                        "text": f"CHEQUERED FLAG: {timing_driver.code} FINISHES ({lap_str})",
                        "car_id": cid,
                    })

            # Stopped car on track check (>500 frames / 10s at speed < 0.2 m/s triggers DNF)
            if not is_pre_start and not has_finished and not is_dnf:
                if speed < 0.8:
                    self.stopped_frames[cid] += 1
                    if self.stopped_frames[cid] >= 500:
                        steward_state.is_dnf = True
                        steward_state.dnf_reason = "STOPPED ON TRACK (>10s)"
                        self.race_events.append({
                            "step": step,
                            "time": round(step / self.fps, 2),
                            "type": "DNF",
                            "text": f"CAR {timing_driver.code} - RETIRED (STOPPED ON TRACK)",
                            "car_id": cid,
                        })
                else:
                    self.stopped_frames[cid] = 0

            # Append telemetry state
            act = actions[cid]
            traj = self.trajectories[cid]
            traj["x"].append(round(float(x), 2))
            traj["y"].append(round(float(y), 2))
            traj["heading"].append(round(float(heading), 3))
            traj["hull_angle"].append(round(float(hull_angle), 3))
            traj["speed"].append(round(float(speed), 1))
            traj["steer"].append(round(float(act[0]), 2))
            traj["gas"].append(round(float(act[1]), 2))
            traj["brake"].append(round(float(act[2]), 2))
            traj["wheels"].append(wheel_flags)
            traj["progress"].append(round(float(progress_pct), 1))
            traj["lap"].append(int(timing_driver.current_lap))
            traj["offroad"].append(is_offroad)
            traj["centre_dist"].append(round(float(cdist), 1))
            traj["lateral_g"].append(lat_g)

            self._prev_heading[cid] = heading

        # 7. Lifecycle & Session End Resolution
        all_eligible_finished = all(
            (cid in self.finished_cars or self.stewards.states[cid].is_dnf)
            for cid in self.cars
        )

        # 120s cutoff after winner for stranded cars (default 120 s)
        time_since_winner = (step - self.winner_finish_step) if self.winner_id else 0
        timed_out = (self.winner_id is not None) and (time_since_winner >= int(self.timeout_seconds * self.fps))

        if timed_out:
            for cid in self.cars:
                if cid not in self.finished_cars and not self.stewards.states[cid].is_dnf:
                    self.stewards.states[cid].is_dnf = True
                    self.stewards.states[cid].dnf_reason = "NOT CLASSIFIED (>120s BEHIND WINNER)"
                    self.race_events.append({
                        "step": step,
                        "time": round(step / self.fps, 2),
                        "type": "NOT_CLASSIFIED",
                        "text": f"CAR {self.driver_meta[cid]['code']} - NOT CLASSIFIED (TIME LIMIT EXCEEDED)",
                        "car_id": cid,
                    })

        if (all_eligible_finished or timed_out) and self.session_end_step < 0:
            self.last_finisher_step = step
            # Requirement: Session ends exactly 750 steps (15s @ 50 FPS) after the last finisher
            self.session_end_step = self.last_finisher_step + 750
            self.race_state = "COOLDOWN"
            self.race_events.append({
                "step": step,
                "time": round(step / self.fps, 2),
                "type": "cooldown_start",
                "text": "ALL CLASSIFIED CARS FINISHED - 15s COOLDOWN TO RESULTS",
            })

        # Broadcast visible countdown during COOLDOWN
        if self.session_end_step > 0 and step < self.session_end_step:
            rem_steps = self.session_end_step - step
            if rem_steps % 50 == 0:  # Every second (50 FPS)
                rem_s = int(rem_steps / self.fps)
                self.race_events.append({
                    "step": step,
                    "time": round(step / self.fps, 2),
                    "type": "countdown",
                    "text": f"SESSION ENDS IN 00:{rem_s:02d}",
                })

        # Append new steward notices
        recent_notices = self.stewards.get_recent_notices(step, window_steps=1)
        for notice in recent_notices:
            if notice.step == step:
                self.race_events.append({
                    "step": step,
                    "time": round(step / self.fps, 2),
                    "type": notice.level,
                    "text": notice.message,
                    "car_id": notice.car_id,
                })

        # Terminate when session end step is reached
        if self.session_end_step > 0 and step >= self.session_end_step:
            self.race_state = "RESULTS"
            return False

        # Hard safety cap (generous cap against infinite loops)
        hard_cap = max(4000, self.total_laps * 2500)
        if step >= hard_cap:
            print(f"[WARN] Hard safety cap reached at {hard_cap} steps. Terminating session.")
            return False

        return True

    def _apply_follow_logic(self, my_cid: str, my_car: car_dynamics.Car, act: np.ndarray) -> np.ndarray:
        """Modulate speed in 'follow' traffic mode to prevent rear-ending."""
        my_x, my_y, my_heading, my_speed = get_car_state(my_car)
        fwd = np.array([math.cos(my_heading), math.sin(my_heading)])
        right = np.array([math.sin(my_heading), -math.cos(my_heading)])

        steer, gas, brake = float(act[0]), float(act[1]), float(act[2])

        for other_cid, other_car in self.cars.items():
            if other_cid == my_cid:
                continue
            ox, oy, ohead, ospeed = get_car_state(other_car)
            rel = np.array([ox - my_x, oy - my_y])
            d_fwd = float(np.dot(rel, fwd))
            d_lat = float(np.dot(rel, right))

            # Car ahead within forward cone
            if 0.5 < d_fwd < 20.0 and abs(d_lat) < 3.2:
                if d_fwd < 7.5:
                    gas = 0.0
                    brake = 0.85
                elif d_fwd < 13.0 and my_speed > ospeed:
                    gas = 0.0
                    brake = float(min(0.6, 0.2 + (13.0 - d_fwd) * 0.08))
                elif d_fwd < 20.0 and my_speed > ospeed:
                    gas *= 0.35

        return np.array([steer, gas, brake], dtype=np.float32)

    def _process_contacts(self, step: int, pre_states: Optional[Dict[str, Any]] = None):
        """Analyze car-car hull collisions and assign steward penalties."""
        cooldown_frames = 25  # Debounce repeated contact impulses
        for cid_a, cid_b in self._raw_contacts_this_step:
            pair_key = tuple(sorted([cid_a, cid_b]))
            last_step = self._last_contact_step.get(pair_key, -100)
            if step - last_step < cooldown_frames:
                continue
            self._last_contact_step[pair_key] = step

            car_a = self.cars[cid_a]
            car_b = self.cars[cid_b]
            if pre_states and cid_a in pre_states and cid_b in pre_states:
                xa, ya, heada, speeda = pre_states[cid_a]
                xb, yb, headb, speedb = pre_states[cid_b]
            else:
                xa, ya, heada, speeda = get_car_state(car_a)
                xb, yb, headb, speedb = get_car_state(car_b)

            # Vector from A to B
            vec_ab = np.array([xb - xa, yb - ya])
            dist = float(np.hypot(vec_ab[0], vec_ab[1]))

            fwd_a = np.array([math.cos(heada), math.sin(heada)])
            fwd_b = np.array([math.cos(headb), math.sin(headb)])

            dot_a = float(np.dot(vec_ab / max(dist, 1e-3), fwd_a))
            dot_b = float(np.dot(-vec_ab / max(dist, 1e-3), fwd_b))

            at_fault = None
            incident_type = "racing_incident"

            # If car A hit car B from behind (vector AB aligns with forward A, and speed A > speed B)
            if dot_a > 0.70 and speeda > (speedb + 1.5):
                at_fault = cid_a
                incident_type = "rear_end"
            elif dot_b > 0.70 and speedb > (speeda + 1.5):
                at_fault = cid_b
                incident_type = "rear_end"

            record = ContactRecord(
                step=step,
                time_s=round(step / self.fps, 2),
                car_a=cid_a,
                car_b=cid_b,
                at_fault=at_fault,
                relative_speed=round(abs(speeda - speedb), 2),
                incident_type=incident_type,
            )
            self.contacts_log.append(record)

            code_a = self.driver_meta[cid_a]["code"]
            code_b = self.driver_meta[cid_b]["code"]

            if at_fault:
                fault_code = self.driver_meta[at_fault]["code"]
                self.stewards.apply_penalty(at_fault, penalty_seconds=5, reason="CAUSING A COLLISION", step=step)
                self.race_events.append({
                    "step": step,
                    "time": round(step / self.fps, 2),
                    "type": "penalty_5s",
                    "text": f"COLLISION: {fault_code} REAR-ENDED {code_b if at_fault==cid_a else code_a} (+5s PENALTY)",
                    "car_id": at_fault,
                })
            else:
                self.race_events.append({
                    "step": step,
                    "time": round(step / self.fps, 2),
                    "type": "contact_incident",
                    "text": f"CAR CONTACT: {code_a} & {code_b} (RACING INCIDENT)",
                    "car_id": cid_a,
                })

    def run_full_race(self, progress_callback=None) -> Dict[str, Any]:
        """Execute the entire race until conclusion and return replay payload."""
        t0 = time.time()
        while self.step():
            if progress_callback and self.current_step % 250 == 0:
                lb = self.timing.compute_leaderboard(current_step=self.current_step)
                p1 = lb[0] if lb else {}
                progress_callback(self.current_step, p1)

        t_elapsed = time.time() - t0
        sim_duration = self.current_step / self.fps
        rtf = sim_duration / max(t_elapsed, 1e-4)

        return self.build_replay_payload(rtf=rtf)

    def _extract_track_geometry(self) -> Dict[str, Any]:
        """Extract full 3D track ribbon, bounds, checkpoints and kerbs."""
        from f1sim.export3d import extract_track_geometry
        return extract_track_geometry(self.env)

    def export_replay_dict(self, rtf: float = 1.0) -> Dict[str, Any]:
        """Export replay dictionary payload."""
        return self.build_replay_payload(rtf=rtf)

    def build_replay_payload(self, rtf: float = 1.0) -> Dict[str, Any]:
        """Format the complete 3D WebGL Replay Viewer JSON payload."""
        track_geom = self._extract_track_geometry()

        # Leaderboard serialization
        final_lb = self.timing.compute_leaderboard(current_step=self.current_step)
        serializable_lb = []
        for row in final_lb:
            r = dict(row)
            for s_key in ("s1", "s2", "s3"):
                sec = r.get(s_key)
                if sec is not None and hasattr(sec, "time_s"):
                    r[s_key] = {"time_s": round(float(sec.time_s), 3), "status": sec.status}
                else:
                    r[s_key] = None
            serializable_lb.append(r)

        # Drivers metadata
        drivers_list = []
        for spec in self.specs:
            cid = spec["id"]
            team_info = TEAMS.get(spec["team"], {})
            drivers_list.append({
                "id": cid,
                "code": spec["code"],
                "name": spec["name"],
                "team": spec["team"],
                "team_name": team_info.get("name", spec["team"]),
                "number": spec.get("number", 0),
                "color": team_info.get("hex", "#FFFFFF"),
                "model": spec.get("model_file", "expert"),
                "grid_pos": spec.get("grid_pos", 1),
                "pace_variance": f"{(self.pace_multipliers.get(cid, 1.0) - 1.0)*100:+.1f}%",
            })

        payload = {
            "meta": {
                "title": f"Autonomous Grand Prix — 3D Replay Seed {self.track_seed}",
                "track_seed": self.track_seed,
                "total_laps": self.total_laps,
                "fps": int(self.fps),
                "num_cars": self.num_cars,
                "total_steps": self.current_step,
                "duration_s": round(self.current_step / self.fps, 2),
                "lights_out_step": self.lights_hold_steps,
                "traffic_mode": self.traffic_mode,
                "contacts_count": len(self.contacts_log),
                "real_time_factor": round(rtf, 2),
                "sim_constants": {
                    "SCALE": 6.0,
                    "TRACK_WIDTH": TRACK_WIDTH,
                    "ROAD_HALF_WIDTH": TRACK_WIDTH,
                    "PLAYFIELD": 2000.0 / 6.0,
                    "CAR_LENGTH": 5.0,
                    "CAR_WIDTH": 2.4,
                    "WHEEL_RADIUS": 0.54,
                    "WHEEL_WIDTH": 0.56,
                    "WHEELBASE": 3.24,
                },
                "disclaimer": "3D VISUALISATION - SIM PHYSICS IS 2D. Box2D shared-world collisions enabled.",
                "speed_conversion": "KM/H = world_speed * 3.6 (exact m/s to km/h conversion)",
            },
            "track": track_geom,
            "drivers": drivers_list,
            "events": self.race_events,
            "contacts": [
                {
                    "step": c.step,
                    "time_s": c.time_s,
                    "car_a": c.car_a,
                    "car_b": c.car_b,
                    "at_fault": c.at_fault,
                    "relative_speed": c.relative_speed,
                    "incident_type": c.incident_type,
                }
                for c in self.contacts_log
            ],
            "classification": serializable_lb,
            "cars": self.trajectories,
        }
        return payload

    def close(self):
        """Cleanup environment and Box2D bodies."""
        if self.env:
            self.env.close()
