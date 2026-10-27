"""
f1sim/env.py - Unified, reliable environment factory for CarRacing-v3

Key guarantees:
1. Idempotent FrictionDetector patch with sentinel attribute (_f1sim_patched).
2. Explicit max_episode_steps (Gymnasium's silent 1000-step TimeLimit is strictly overridden).
3. Enriched step info: info['lap_finished'], info['out_of_bounds'], info['progress_pct'].
4. Leaving the playfield (out of bounds) terminates the episode but NEVER counts as a completed lap.
5. Multi-lap race mode support (suppresses single-lap auto-termination).
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium.envs.box2d import car_racing

# Headless video driver safety default
if "SDL_VIDEODRIVER" not in os.environ:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


def apply_finish_line_patch():
    """
    Apply finish-line detection patch to FrictionDetector.BeginContact idempotently.
    Sentinel attribute '_f1sim_patched' ensures it is applied at most once,
    even if modules are reloaded repeatedly.
    """
    if getattr(car_racing.FrictionDetector, "_f1sim_patched", False):
        return False

    orig_begin_contact = car_racing.FrictionDetector.BeginContact

    def fixed_begin_contact(self, contact):
        orig_begin_contact(self, contact)
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        tile = (
            u1
            if (u1 and hasattr(u1, "road_friction"))
            else (u2 if (u2 and hasattr(u2, "road_friction")) else None)
        )
        if tile and hasattr(tile, "idx") and tile.idx == 0:
            if hasattr(self, "env") and hasattr(self.env, "track") and len(self.env.track) > 0:
                if (self.env.tile_visited_count / len(self.env.track)) > self.lap_complete_percent:
                    self.env.new_lap = True

    fixed_begin_contact._orig_func = orig_begin_contact
    car_racing.FrictionDetector.BeginContact = fixed_begin_contact
    car_racing.FrictionDetector._f1sim_patched = True
    return True


# Apply patch upon module import
apply_finish_line_patch()


class F1EnvWrapper(gym.Wrapper):
    """
    Custom wrapper providing audit-proof step accounting, lap distinction,
    and out-of-bounds safety.
    """

    def __init__(self, env, race_mode=False, max_steps=2000):
        super().__init__(env)
        self.race_mode = race_mode
        self.configured_max_steps = max_steps
        self.episode_steps = 0
        self.laps_completed = 0

    def reset(self, **kwargs):
        self.episode_steps = 0
        self.laps_completed = 0
        obs, info = self.env.reset(**kwargs)
        uw = self.env.unwrapped
        info["lap_finished"] = False
        info["out_of_bounds"] = False
        info["tiles_visited"] = getattr(uw, "tile_visited_count", 0)
        info["total_tiles"] = len(uw.track) if hasattr(uw, "track") else 0
        info["progress_pct"] = 0.0
        info["step_count"] = 0
        return obs, info

    def step(self, action):
        self.episode_steps += 1
        uw = self.env.unwrapped

        if not isinstance(action, np.ndarray):
            action = np.asarray(action, dtype=np.float32)

        prev_new_lap = getattr(uw, "new_lap", False)
        obs, reward, terminated, truncated, info = self.env.step(action)

        hull = uw.car.hull
        x, y = hull.position
        playfield = getattr(car_racing, "PLAYFIELD", 2000 / car_racing.SCALE)
        is_out_of_bounds = bool(abs(x) > playfield or abs(y) > playfield)

        # In standard Gym, terminated=True can be triggered by BOTH out-of-bounds AND lap completion.
        # We explicitly decouple them:
        total_tiles = len(uw.track) if hasattr(uw, "track") else 1
        tiles_visited = getattr(uw, "tile_visited_count", 0)
        progress_pct = (tiles_visited / total_tiles) * 100.0

        is_lap_finished = bool(
            (uw.new_lap or (not self.race_mode and tiles_visited == total_tiles))
            and not is_out_of_bounds
        )

        if is_lap_finished:
            self.laps_completed += 1

        info["lap_finished"] = is_lap_finished
        info["out_of_bounds"] = is_out_of_bounds
        info["tiles_visited"] = tiles_visited
        info["total_tiles"] = total_tiles
        info["progress_pct"] = progress_pct
        info["step_count"] = self.episode_steps
        info["laps_completed"] = self.laps_completed

        # In race_mode (multi-lap), reset tiles for the subsequent lap
        if self.race_mode and is_lap_finished and not is_out_of_bounds:
            terminated = False
            uw.new_lap = False
            uw.tile_visited_count = 0
            for t in getattr(uw, "road", []):
                t.road_visited = False

        return obs, reward, terminated, truncated, info


def make_env(
    seed=None,
    max_steps=2000,
    render_mode="rgb_array",
    lap_complete_percent=0.95,
    continuous=True,
    race_mode=False,
):
    """
    Standard environment factory. Guarantees explicit max_episode_steps and
    idempotent finish line patch.
    """
    apply_finish_line_patch()

    env = gym.make(
        "CarRacing-v3",
        render_mode=render_mode,
        lap_complete_percent=lap_complete_percent,
        continuous=continuous,
        max_episode_steps=max_steps,
    )

    wrapped_env = F1EnvWrapper(env, race_mode=race_mode, max_steps=max_steps)

    if seed is not None:
        wrapped_env.reset(seed=seed)

    return wrapped_env
