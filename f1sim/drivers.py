"""
f1sim/drivers.py - Driver agents, model inference, and fictional F1 team metadata.

Implements:
- Driver (Base class)
- ExpertDriver (Pure pursuit with speed controller)
- ModelDriver (84x96 cropped RGB behavioral cloning inference)
- BatchedModelDriver (Multi-car concurrent forward passes)
- Fictional F1 Teams & Drivers registry (unofficial branding compliance)
- Grid positioning logic for multi-car starts
"""

import os
import math
from abc import ABC, abstractmethod
import numpy as np

# PyTorch / Keras backend initialization
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras

from expert import expert_steer, speed_controller, get_car_state


# Fictional Team & Driver Registry
# Rule: Unofficial branding only. Fictional names, 3-letter codes, curated hex colors.
TEAMS = {
    "APX": {
        "name": "Apex Racing GP",
        "color": (0, 229, 255),       # Cyan / Electric Teal (#00E5FF)
        "hex": "#00E5FF",
        "secondary": "#0A192F",
        "driver": "Alex Vance",
        "code": "VAN",
        "number": 4,
    },
    "VEL": {
        "name": "Veloce Corse",
        "color": (255, 23, 68),       # Neon Crimson (#FF1744)
        "hex": "#FF1744",
        "secondary": "#210006",
        "driver": "Marco Silva",
        "code": "SIL",
        "number": 16,
    },
    "NDX": {
        "name": "Nordic Dynamics",
        "color": (41, 121, 255),      # Electric Cobalt (#2979FF)
        "hex": "#2979FF",
        "secondary": "#001033",
        "driver": "Lukas Lind",
        "code": "LIN",
        "number": 23,
    },
    "ZEN": {
        "name": "Zenith Motorsport",
        "color": (255, 214, 0),       # Cyber Gold / Amber (#FFD600)
        "hex": "#FFD600",
        "secondary": "#1E1800",
        "driver": "Kai Tanaka",
        "code": "TAN",
        "number": 88,
    },
}


class Driver(ABC):
    """
    Abstract driver interface.
    """

    def __init__(self, name="Driver", code="DRV", team="APX"):
        self.name = name
        self.code = code
        self.team = team
        self.team_info = TEAMS.get(team, TEAMS["APX"])

    @abstractmethod
    def act(self, obs, env, step=None):
        """
        Compute action [steer, gas, brake] given current observation and environment.
        """
        pass

    def reset(self):
        """Reset internal driver state if any."""
        pass


class ExpertDriver(Driver):
    """
    Scripted expert driver utilizing pure pursuit lookahead and speed control.
    """

    def __init__(self, name="Scripted Expert", code="EXP", team="ZEN",
                 lookahead=11, gain=0.55, target_straight=70.0, target_turn=38.0):
        super().__init__(name=name, code=code, team=team)
        self.lookahead = lookahead
        self.gain = gain
        self.target_straight = target_straight
        self.target_turn = target_turn

    def act(self, obs, env, step=None):
        if step is not None and step < 50:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        steer = expert_steer(env, lookahead=self.lookahead, gain=self.gain)
        gas, brake = speed_controller(
            env, steer,
            target_speed_straight=self.target_straight,
            target_speed_turn=self.target_turn
        )
        return np.array([steer, gas, brake], dtype=np.float32)


class ModelDriver(Driver):
    """
    Neural behavioral cloning driver.
    Receives 96x96 RGB gym observation, crops to 84x96, normalizes to [0, 1],
    and executes single forward pass.
    """

    def __init__(self, model_path_or_model, name="Model", code="MDL", team="APX",
                 target_straight=70.0, target_turn=38.0):
        super().__init__(name=name, code=code, team=team)
        self.target_straight = target_straight
        self.target_turn = target_turn

        if isinstance(model_path_or_model, str):
            assert os.path.exists(model_path_or_model), f"Model file not found: {model_path_or_model}"
            self.model = keras.models.load_model(model_path_or_model)
        else:
            self.model = model_path_or_model

    def preprocess_obs(self, obs):
        """
        Strict non-negotiable rule: Model must receive 84x96 cropped RGB gym observation.
        The model architecture includes layers.Rescaling(1/127.5, offset=-1), so inputs
        must be float32 in range [0, 255].
        """
        crop = obs[:84, :, :].astype(np.float32)
        return crop

    def act(self, obs, env, step=None):
        # Initial zoom-in skip frames: car dropped from height, camera zooming in
        if step is not None and step < 50:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        crop = self.preprocess_obs(obs)
        batch = np.expand_dims(crop, axis=0)

        # Single forward pass without training gradients
        pred = self.model(batch, training=False)
        pred_steer = float(pred.detach().cpu().numpy()[0, 0])
        pred_steer = float(np.clip(pred_steer, -1.0, 1.0))

        gas, brake = speed_controller(
            env, pred_steer,
            target_speed_straight=self.target_straight,
            target_speed_turn=self.target_turn
        )
        return np.array([pred_steer, gas, brake], dtype=np.float32)


class BatchedModelDriver:
    """
    High-throughput batched inference driver for multi-car simulations.
    Combines observations from multiple cars into a single (B, 84, 96, 3) forward pass.
    """

    def __init__(self, models_dict):
        """
        models_dict: dict mapping car_id -> keras Model
        """
        self.models = models_dict

    def act_batch(self, obs_dict, env_dict, target_straight=70.0, target_turn=38.0):
        """
        obs_dict: dict mapping car_id -> obs
        env_dict: dict mapping car_id -> env
        Returns dict mapping car_id -> np.array([steer, gas, brake])
        """
        actions = {}
        for car_id, model in self.models.items():
            obs = obs_dict[car_id]
            env = env_dict[car_id]
            crop = obs[:84, :, :].astype(np.float32) / 255.0
            batch = np.expand_dims(crop, axis=0)

            pred = model(batch, training=False)
            pred_steer = float(pred.detach().cpu().numpy()[0, 0])
            pred_steer = float(np.clip(pred_steer, -1.0, 1.0))

            gas, brake = speed_controller(
                env, pred_steer,
                target_speed_straight=target_straight,
                target_speed_turn=target_turn
            )
            actions[car_id] = np.array([pred_steer, gas, brake], dtype=np.float32)

        return actions


def compute_staggered_grid_positions(track_nodes, grid_slots=4, spacing_meters=6.0, lateral_offset=1.5):
    """
    Computes staggered 2-wide F1-style grid positions behind the start line (node 0).
    Grid slots:
      Slot 0 (Pole / P1): on the racing line side, spacing * 1 behind line
      Slot 1 (P2): on opposite side, spacing * 2 behind line
      Slot 2 (P3): on P1 side, spacing * 3 behind line
      Slot 3 (P4): on P2 side, spacing * 4 behind line
    """
    n = len(track_nodes)
    positions = []

    # Node 0 is start/finish line. Grid forms backwards from node 0 (nodes n-1, n-2...)
    # Calculate backward cumulative distance
    diffs = track_nodes - np.roll(track_nodes, 1, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])

    for i in range(grid_slots):
        target_dist = (i + 1) * spacing_meters
        # Search backwards along track nodes
        cum = 0.0
        curr_idx = 0
        while cum < target_dist and curr_idx < n:
            prev_idx = (curr_idx - 1) % n
            seg_len = seg_lens[curr_idx]
            if cum + seg_len >= target_dist:
                frac = (target_dist - cum) / max(seg_len, 1e-4)
                base_pos = track_nodes[curr_idx] * (1.0 - frac) + track_nodes[prev_idx] * frac
                # Heading tangent
                tangent = track_nodes[curr_idx] - track_nodes[prev_idx]
                tangent_len = np.hypot(tangent[0], tangent[1])
                if tangent_len > 1e-4:
                    tangent /= tangent_len
                    # Lateral normal: (-tangent[1], tangent[0])
                    lat_sign = -1.0 if (i % 2 == 0) else 1.0
                    lat_normal = np.array([-tangent[1], tangent[0]]) * (lateral_offset * lat_sign)
                    pos = base_pos + lat_normal
                    heading = math.atan2(tangent[1], tangent[0])
                else:
                    pos = base_pos
                    heading = 0.0
                positions.append({"pos": pos, "heading": heading, "slot": i + 1})
                break
            cum += seg_len
            curr_idx = prev_idx

    return positions
