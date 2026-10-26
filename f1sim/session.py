"""
f1sim/session.py - Race Weekend orchestrator: Qualifying, Grand Prix Race, and Media Generator.

Provides:
- RaceSession: Synchronized multi-car simulation on fixed circuit seeds.
- QualifyingSession: Flying lap shootout to determine starting grid.
- WeekendRunner: End-to-end race weekend execution with live rendering, hero MP4, and telemetry capture.
"""

import os
import sys
import time
import imageio
import numpy as np
import pygame

from f1sim.env import make_env
from f1sim.drivers import ExpertDriver, ModelDriver, TEAMS, compute_staggered_grid_positions
from f1sim.timing import TimingSystem
from f1sim.stewards import Stewards, RulesConfig
from f1sim.renderer import DashboardRenderer
from f1sim.replay import TelemetryRecorder
from expert import wheels_on_track, get_car_state


# Default 4-Car Showcase Grid:
# 1. Apex Racing GP: DAgger Seed 0 (Champion)
# 2. Veloce Corse: Plain Cloning Seed 0
# 3. Nordic Dynamics: Augmented Cloning Seed 0
# 4. Zenith Motorsport: Scripted Pure-Pursuit Expert
DEFAULT_CAR_SPECS = [
    {
        "id": "car_apx",
        "team": "APX",
        "code": "VAN",
        "name": "Alex Vance",
        "type": "model",
        "model_file": "models/dagger_s0.keras",
    },
    {
        "id": "car_vel",
        "team": "VEL",
        "code": "SIL",
        "name": "Marco Silva",
        "type": "model",
        "model_file": "models/plain_s0.keras",
    },
    {
        "id": "car_ndx",
        "team": "NDX",
        "code": "LIN",
        "name": "Lukas Lind",
        "type": "model",
        "model_file": "models/aug_s0.keras",
    },
    {
        "id": "car_zen",
        "team": "ZEN",
        "code": "TAN",
        "name": "Kai Tanaka",
        "type": "expert",
    },
]


class RaceSession:
    """
    Multi-car race session running synchronized parallel physics on a shared circuit seed.
    """

    def __init__(
        self,
        track_seed=3003,
        total_laps=3,
        fps=50,
        render_width=1920,
        render_height=1080,
        stewards_config=None,
    ):
        # Strict rule: Showcase seeds must be 3000-3019
        assert 3000 <= track_seed <= 3019, f"Showcase seed must be in [3000, 3019], got {track_seed}"
        self.track_seed = track_seed
        self.total_laps = total_laps
        self.fps = fps
        self.width = render_width
        self.height = render_height

        self.timing = TimingSystem(fps=fps, num_checkpoints=100)
        self.stewards = Stewards(config=stewards_config, fps=fps)
        self.renderer = DashboardRenderer(width=render_width, height=render_height)
        self.recorder = TelemetryRecorder(track_seed=track_seed, fps=fps, num_cars=4)

        self.drivers = {}
        self.envs = {}
        self.driver_metadata = {}

    def setup_drivers(self, car_specs=None):
        """Instantiate drivers and parallel environments."""
        specs = car_specs or DEFAULT_CAR_SPECS
        for spec in specs:
            cid = spec["id"]
            code = spec["code"]
            name = spec["name"]
            team = spec["team"]
            self.driver_metadata[cid] = spec

            # Register in timing and stewards
            self.timing.register_driver(cid, code, name, team)
            self.stewards.register_car(cid, code)

            # Create environment for this car
            env = make_env(seed=self.track_seed, max_steps=15000, render_mode="rgb_array", race_mode=True)
            self.envs[cid] = env

            # Create driver
            if spec["type"] == "expert":
                driver = ExpertDriver(name=name, code=code, team=team)
            else:
                driver = ModelDriver(spec["model_file"], name=name, code=code, team=team)
            self.drivers[cid] = driver

    def run_race(
        self,
        record_video=False,
        video_path=None,
        record_gif=False,
        gif_path=None,
        max_steps=5000,
        progress_callback=None,
    ):
        """
        Execute the race simulation.
        Returns race results dict.
        """
        obs_dict = {}
        cars_state = {}

        # Reset all environments with identical track seed
        for cid, env in self.envs.items():
            obs, info = env.reset(seed=self.track_seed)
            obs_dict[cid] = obs

        video_writer = None
        if record_video and video_path:
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            video_writer = imageio.get_writer(video_path, fps=self.fps, codec="libx264", macro_block_size=1)

        gif_frames = []
        selected_car_id = list(self.drivers.keys())[0]

        # Reference environment for track geometry
        ref_env = list(self.envs.values())[0]
        uw = ref_env.unwrapped
        waypoints = np.array([[p[2], p[3]] for p in uw.track], dtype=np.float32)

        start_time = time.time()
        winner_id = None
        winner_finish_step = -1

        for step in range(max_steps):
            # 1. Collect actions from each driver
            actions = {}
            for cid, driver in self.drivers.items():
                if self.stewards.states[cid].is_dnf:
                    actions[cid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Parked
                    continue
                actions[cid] = driver.act(obs_dict[cid], self.envs[cid], step=step)

            # 2. Step each environment
            for cid, env in self.envs.items():
                if self.stewards.states[cid].is_dnf:
                    continue

                act = actions[cid]
                obs, reward, terminated, truncated, info = env.step(act)
                obs_dict[cid] = obs

                car_x, car_y, heading, speed = get_car_state(env)
                on_track_wheels = wheels_on_track(env)
                is_offroad = (on_track_wheels == 0)

                # Stewards rule check
                self.stewards.check_step(step, cid, is_offroad, speed)

                # Timing system update
                lap_finished = info.get("lap_finished", False)
                progress_pct = info.get("progress_pct", 0.0)
                self.timing.update_driver_progress(cid, current_step=step, progress_pct=progress_pct, lap_finished=lap_finished)

                # Centre line distance
                diff = waypoints - np.array([car_x, car_y])
                cdist = float(np.sqrt(np.min(np.sum(diff ** 2, axis=1))))

                spec = self.driver_metadata[cid]
                cars_state[cid] = {
                    "pos": (car_x, car_y),
                    "heading": heading,
                    "speed": speed,
                    "steer": float(act[0]),
                    "gas": float(act[1]),
                    "brake": float(act[2]),
                    "team_code": spec["team"],
                    "code": spec["code"],
                    "offroad": is_offroad,
                    "centre_dist": cdist,
                    "lateral_g": float(abs(act[0]) * speed * 0.015),
                    "progress_pct": progress_pct,
                    "lap": self.timing.drivers[cid].current_lap,
                    "sector": 1,
                }

                # Check race win condition
                if self.timing.drivers[cid].laps_completed >= self.total_laps and winner_id is None:
                    winner_id = cid
                    winner_finish_step = step

            # Compute leaderboard
            leaderboard = self.timing.compute_leaderboard(current_step=step)

            # Auto-follow leader if not manually switched
            if leaderboard:
                selected_car_id = leaderboard[0]["car_id"]

            # Record telemetry frame
            self.recorder.record_frame(
                step=step,
                cars_data=cars_state,
                leaderboard=leaderboard,
                race_control_events=[ev.to_dict() for ev in self.stewards.get_recent_notices(step, window_steps=100)],
            )

            # Render frame for video / inspection
            if video_writer is not None or (record_gif and step % 2 == 0):
                frame_surf = self.renderer.render_frame(
                    env=ref_env,
                    cars_state=cars_state,
                    leaderboard=leaderboard,
                    selected_car_id=selected_car_id,
                    current_step=step,
                    total_laps=self.total_laps,
                    race_events=self.stewards.get_recent_notices(step, window_steps=150),
                    active_cam_obs=obs_dict.get(selected_car_id),
                )
                rgb_frame = pygame.surfarray.array3d(frame_surf)
                rgb_frame = np.transpose(rgb_frame, (1, 0, 2))

                if video_writer is not None:
                    video_writer.append_data(rgb_frame)
                if record_gif and gif_path and len(gif_frames) < 180 and step % 3 == 0:
                    # Resize for GIF efficiency
                    small_frame = pygame.transform.scale(frame_surf, (640, 360))
                    g_arr = pygame.surfarray.array3d(small_frame)
                    gif_frames.append(np.transpose(g_arr, (1, 0, 2)))

            # Chequered flag logic: All cars finish OR 300 steps (6s) after winner crosses line
            all_done = all(
                (self.timing.drivers[cid].laps_completed >= self.total_laps or self.stewards.states[cid].is_dnf)
                for cid in self.drivers
            )
            if all_done or (winner_id is not None and (step - winner_finish_step) >= 300):
                break

            if progress_callback and step % 250 == 0:
                progress_callback(step, max_steps, leaderboard)

        # Close resources
        if video_writer is not None:
            video_writer.close()
        if record_gif and gif_frames and gif_path:
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            imageio.mimsave(gif_path, gif_frames, fps=15)
        for env in self.envs.values():
            env.close()

        final_leaderboard = self.timing.compute_leaderboard(current_step=step)
        return {
            "winner_id": winner_id,
            "total_steps": step + 1,
            "elapsed_s": round(time.time() - start_time, 2),
            "leaderboard": final_leaderboard,
            "stewards_events": [ev.to_dict() for ev in self.stewards.event_log],
        }
