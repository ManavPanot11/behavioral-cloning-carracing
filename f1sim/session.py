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


from f1sim.world import MultiCarWorld, validate_seed, CarHullUserData, SIX_CAR_SPECS


class CarEnvWrapper:
    """
    Wrapper around the shared MultiCarWorld environment presenting standard Gym env
    semantics for a specific car (e.g. for inspection and compatibility).
    """

    def __init__(self, mcw: MultiCarWorld, car_id: str):
        self.mcw = mcw
        self.car_id = car_id
        self.car = mcw.cars[car_id]
        self._env = mcw.env

    @property
    def unwrapped(self):
        class UnwrappedProxy:
            def __init__(proxy_self, uw, car, mcw):
                proxy_self._uw = uw
                proxy_self.car = car
                proxy_self.world = mcw.world
                proxy_self.track = mcw.track
                proxy_self.t = uw.t

            def __getattr__(proxy_self, item):
                return getattr(proxy_self._uw, item)

        return UnwrappedProxy(self._env.unwrapped, self.car, self.mcw)

    def reset(self, seed=None):
        # Already placed on staggered starting grid in MultiCarWorld
        return self.mcw._get_car_observation(self.car), {}

    def step(self, action):
        # Stepping is handled synchronously by MultiCarWorld
        return None, 0.0, False, False, {}

    def close(self):
        pass

    def __getattr__(self, item):
        return getattr(self._env, item)


class RaceSession:
    """
    Multi-car race session running synchronized parallel physics on a shared circuit seed
    using the unified Box2D shared-world engine (MultiCarWorld).
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
        validate_seed(track_seed)
        self.track_seed = track_seed
        self.total_laps = total_laps
        self.fps = fps
        self.width = render_width
        self.height = render_height
        self.stewards_config = stewards_config

        self.timing = TimingSystem(fps=fps, num_checkpoints=100)
        self.stewards = Stewards(config=stewards_config, fps=fps)
        self.renderer = DashboardRenderer(width=render_width, height=render_height)
        self.recorder = TelemetryRecorder(track_seed=track_seed, fps=fps, num_cars=4)

        self.drivers = {}
        self.envs = {}
        self.driver_metadata = {}
        self.mcw = None

    def setup_drivers(self, car_specs=None):
        """Instantiate drivers and single shared MultiCarWorld."""
        specs = car_specs or DEFAULT_CAR_SPECS
        self.mcw = MultiCarWorld(
            track_seed=self.track_seed,
            total_laps=self.total_laps,
            car_specs=specs,
            fps=self.fps,
            stewards_config=self.stewards_config,
        )
        self.timing = self.mcw.timing
        self.stewards = self.mcw.stewards

        for spec in specs:
            cid = spec["id"]
            self.driver_metadata[cid] = spec
            self.drivers[cid] = self.mcw.drivers[cid]
            self.envs[cid] = CarEnvWrapper(self.mcw, cid)

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
        Execute the race simulation using the shared-world Box2D engine.
        Returns race results dict.
        """
        if self.mcw is None:
            self.setup_drivers()

        video_writer = None
        if record_video and video_path:
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            video_writer = imageio.get_writer(video_path, fps=self.fps, codec="libx264", macro_block_size=1)

        gif_frames = []
        selected_car_id = list(self.drivers.keys())[0]
        start_time = time.time()
        ref_env = self.mcw.env

        while self.mcw.step():
            step = self.mcw.current_step
            if step >= max_steps:
                break

            leaderboard = self.timing.compute_leaderboard(current_step=step)
            if leaderboard:
                selected_car_id = leaderboard[0]["car_id"]

            cars_state = {}
            for cid, car in self.mcw.cars.items():
                x, y, heading, speed = get_car_state(car)
                ang_vel = float(car.hull.angularVelocity)
                wheel_flags = [bool(len(w.tiles) > 0) for w in car.wheels]
                is_offroad = sum(wheel_flags) == 0
                diff = self.mcw.track_pts - np.array([x, y])
                cdist = float(np.sqrt(np.min(np.sum(diff ** 2, axis=1))))
                spec = self.driver_metadata[cid]
                timing_d = self.timing.drivers[cid]

                # Estimated lateral G: speed * |yaw_rate| / 9.81
                lat_g = round(float((speed * abs(ang_vel)) / 9.81), 2)

                cars_state[cid] = {
                    "pos": (x, y),
                    "heading": heading,
                    "speed": speed,
                    "steer": float(self.mcw.trajectories[cid]["steer"][-1]),
                    "gas": float(self.mcw.trajectories[cid]["gas"][-1]),
                    "brake": float(self.mcw.trajectories[cid]["brake"][-1]),
                    "team_code": spec["team"],
                    "code": spec["code"],
                    "offroad": is_offroad,
                    "centre_dist": cdist,
                    "lateral_g": lat_g,
                    "progress_pct": self.mcw.trackers[cid].progress_pct,
                    "lap": timing_d.current_lap,
                    "sector": 1,
                }

            # Record telemetry frame
            self.recorder.record_frame(
                step=step,
                cars_data=cars_state,
                leaderboard=leaderboard,
                race_control_events=[ev.to_dict() for ev in self.stewards.get_recent_notices(step, window_steps=100)],
            )

            # Render frame for video / inspection
            if video_writer is not None or (record_gif and step % 2 == 0):
                obs = self.mcw._get_car_observation(self.mcw.cars[selected_car_id])
                frame_surf = self.renderer.render_frame(
                    env=ref_env,
                    cars_state=cars_state,
                    leaderboard=leaderboard,
                    selected_car_id=selected_car_id,
                    current_step=step,
                    total_laps=self.total_laps,
                    race_events=self.stewards.get_recent_notices(step, window_steps=150),
                    active_cam_obs=obs,
                )
                rgb_frame = pygame.surfarray.array3d(frame_surf)
                rgb_frame = np.transpose(rgb_frame, (1, 0, 2))

                if video_writer is not None:
                    video_writer.append_data(rgb_frame)
                if record_gif and gif_path and len(gif_frames) < 180 and step % 3 == 0:
                    small_frame = pygame.transform.scale(frame_surf, (640, 360))
                    g_arr = pygame.surfarray.array3d(small_frame)
                    gif_frames.append(np.transpose(g_arr, (1, 0, 2)))

            if progress_callback and step % 250 == 0:
                progress_callback(step, max_steps, leaderboard)

        # Close resources
        if video_writer is not None:
            video_writer.close()
        if record_gif and gif_frames and gif_path:
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            imageio.mimsave(gif_path, gif_frames, fps=15)

        final_step = self.mcw.current_step
        final_leaderboard = self.timing.compute_leaderboard(current_step=final_step)
        results = {
            "winner_id": self.mcw.winner_id,
            "total_steps": final_step,
            "elapsed_s": round(time.time() - start_time, 2),
            "leaderboard": final_leaderboard,
            "stewards_events": [ev.to_dict() for ev in self.stewards.event_log],
        }
        self.mcw.close()
        return results
