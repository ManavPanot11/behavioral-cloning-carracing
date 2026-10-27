"""
f1sim/replay.py - Telemetry recorder and deterministic replay loader.

Provides:
- TelemetryRecorder: Captures per-frame physics, inputs, and timing for all cars.
- Replay: Storage and serialization format (compressed json or npz) for web viewers and post-race analysis.
"""

import gzip
import json


class TelemetryRecorder:
    """
    Records per-frame telemetry traces across all cars in a session.
    """

    def __init__(self, track_seed=3000, fps=50, num_cars=4):
        self.metadata = {
            "track_seed": track_seed,
            "fps": fps,
            "num_cars": num_cars,
            "total_frames": 0,
        }
        self.frames = []

    def record_frame(self, step, cars_data, leaderboard=None, race_control_events=None):
        """
        Record a single simulation frame.
        cars_data: dict mapping car_id -> {
            'pos': (x, y),
            'heading': rad,
            'speed': float,
            'action': [steer, gas, brake],
            'progress_pct': float,
            'sector': int,
            'lap': int,
            'offroad': bool,
            'centre_dist': float,
            'lateral_g': float,
            'slip': float,
        }
        """
        # Ensure leaderboard entries are JSON serializable
        clean_lb = []
        for entry in (leaderboard or []):
            item = dict(entry)
            for s_key in ("s1", "s2", "s3"):
                sec = item.get(s_key)
                if hasattr(sec, "to_dict"):
                    item[s_key] = sec.to_dict()
                elif hasattr(sec, "status"):
                    item[s_key] = {"time_s": sec.time_s, "status": sec.status}
            clean_lb.append(item)

        frame = {
            "step": step,
            "time": step / self.metadata["fps"],
            "cars": {},
            "leaderboard": clean_lb,
            "events": race_control_events or [],
        }

        for car_id, data in cars_data.items():
            pos = data.get("pos", (0.0, 0.0))
            action = data.get("action", [0.0, 0.0, 0.0])
            frame["cars"][car_id] = {
                "x": round(float(pos[0]), 3),
                "y": round(float(pos[1]), 3),
                "heading": round(float(data.get("heading", 0.0)), 4),
                "speed": round(float(data.get("speed", 0.0)), 2),
                "steer": round(float(action[0]), 3),
                "gas": round(float(action[1]), 3),
                "brake": round(float(action[2]), 3),
                "progress_pct": round(float(data.get("progress_pct", 0.0)), 2),
                "sector": int(data.get("sector", 1)),
                "lap": int(data.get("lap", 1)),
                "offroad": bool(data.get("offroad", False)),
                "centre_dist": round(float(data.get("centre_dist", 0.0)), 2),
                "lateral_g": round(float(data.get("lateral_g", 0.0)), 2),
                "slip": round(float(data.get("slip", 0.0)), 3),
            }

        self.frames.append(frame)
        self.metadata["total_frames"] = len(self.frames)

    def save_json_gz(self, filepath):
        """Save telemetry data as compressed JSON for the web dashboard."""
        payload = {
            "metadata": self.metadata,
            "frames": self.frames,
        }
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            json.dump(payload, f, default=self._json_default)

    def save_json(self, filepath):
        """Save telemetry data as JSON."""
        payload = {
            "metadata": self.metadata,
            "frames": self.frames,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, default=self._json_default)

    @staticmethod
    def _json_default(obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    @classmethod
    def load_json(cls, filepath):
        """Load telemetry trace from json or json.gz."""
        if filepath.endswith(".gz"):
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        rec = cls(
            track_seed=data["metadata"].get("track_seed", 3000),
            fps=data["metadata"].get("fps", 50),
            num_cars=data["metadata"].get("num_cars", 4)
        )
        rec.metadata = data["metadata"]
        rec.frames = data["frames"]
        return rec
