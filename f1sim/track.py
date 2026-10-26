"""
f1sim/track.py - Track geometry, sector segmentation, and robust progress tracking.

Provides:
- Track: Geometric representation of CarRacing-v3 track.
- Local-windowed progress tracker preventing wrap-around or teleportation artifacts.
- Exact center-line distance (signed and unsigned).
- F1-style Sector 1, 2, 3 boundary definitions.
"""

import math
import numpy as np


class Track:
    """
    Representation of the circuit track geometry extracted from CarRacing.
    """

    def __init__(self, raw_track):
        """
        raw_track: list of tuples (alpha, beta, x, y) as stored in env.unwrapped.track
        """
        self.raw_track = raw_track
        self.num_tiles = len(raw_track)
        assert self.num_tiles > 0, "Track cannot be empty"

        # Extract (x, y) coordinates of center-line nodes
        self.nodes = np.array([[p[2], p[3]] for p in raw_track], dtype=np.float32)

        # Compute segment lengths and cumulative distances
        diffs = np.roll(self.nodes, -1, axis=0) - self.nodes
        self.segment_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        self.total_length = float(np.sum(self.segment_lengths))
        self.cum_distances = np.zeros(self.num_tiles, dtype=np.float32)
        self.cum_distances[1:] = np.cumsum(self.segment_lengths[:-1])

        # Sector divisions (3 equal-distance or equal-tile sectors)
        self.sector_boundaries = [
            int(self.num_tiles / 3.0),
            int(2.0 * self.num_tiles / 3.0),
            self.num_tiles,
        ]

    def get_sector(self, tile_idx):
        """
        Return sector index (1, 2, or 3) for a given tile index.
        """
        idx = tile_idx % self.num_tiles
        if idx < self.sector_boundaries[0]:
            return 1
        elif idx < self.sector_boundaries[1]:
            return 2
        else:
            return 3

    def find_nearest_tile_windowed(self, pos, current_idx, backward_window=6, forward_window=20):
        """
        Find closest waypoint to `pos` within a constrained local window [current_idx - backward, current_idx + forward].
        Prevents teleporting / shortcut exploit or backwards wrap-around.
        """
        pos = np.asarray(pos, dtype=np.float32)
        n = self.num_tiles

        # Build candidate indices in order from backward to forward
        offsets = list(range(-backward_window, forward_window + 1))
        candidate_indices = [(current_idx + off) % n for off in offsets]
        candidate_nodes = self.nodes[candidate_indices]

        # Compute squared distances
        dists_sq = np.sum((candidate_nodes - pos) ** 2, axis=1)
        best_cand_idx = int(np.argmin(dists_sq))
        best_tile_idx = candidate_indices[best_cand_idx]

        return best_tile_idx

    def compute_continuous_progress(self, pos, tile_idx):
        """
        Compute continuous progress percentage [0.0, 100.0) around the track.
        Projects position onto the segment between tile_idx and (tile_idx + 1).
        """
        n = self.num_tiles
        p1 = self.nodes[tile_idx]
        p2 = self.nodes[(tile_idx + 1) % n]
        seg = p2 - p1
        seg_len_sq = float(np.dot(seg, seg))

        frac = 0.0
        if seg_len_sq > 1e-6:
            v = np.asarray(pos, dtype=np.float32) - p1
            frac = float(np.clip(np.dot(v, seg) / seg_len_sq, 0.0, 1.0))

        progress_tiles = (tile_idx + frac) % n
        progress_pct = (progress_tiles / n) * 100.0
        return progress_pct

    def compute_centre_line_distance(self, pos, tile_idx):
        """
        Compute perpendicular distance from pos to the segment at tile_idx.
        Returns:
            abs_dist: distance in world units (meters)
            signed_dist: positive if to the right of travel direction, negative if to left.
        """
        n = self.num_tiles
        p1 = self.nodes[tile_idx]
        p2 = self.nodes[(tile_idx + 1) % n]
        seg = p2 - p1
        seg_len = float(np.hypot(seg[0], seg[1]))

        if seg_len < 1e-6:
            return float(np.hypot(pos[0] - p1[0], pos[1] - p1[1])), 0.0

        # Normal vector pointing to the right of (p2 - p1)
        # Tangent: (dx, dy) -> Right Normal: (dy, -dx) / seg_len
        r_norm = np.array([seg[1], -seg[0]], dtype=np.float32) / seg_len

        v = np.asarray(pos, dtype=np.float32) - p1
        signed_dist = float(np.dot(v, r_norm))
        abs_dist = abs(signed_dist)

        return abs_dist, signed_dist
