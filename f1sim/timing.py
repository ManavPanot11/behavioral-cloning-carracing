"""
f1sim/timing.py - High-precision F1 timing system with sector coloring, gaps, and intervals.

Implements:
- 100 timing checkpoints per lap for smooth micro-sector tracking.
- Authentic F1 Sector Colors:
    PURPLE: Session best (fastest across all cars)
    GREEN:  Personal best (driver's best sector)
    YELLOW: Slower than personal best
- Genuine interval to car ahead and gap to leader calculated at common checkpoint crossings.
- Live session telemetry timing tower data structures.
"""

# F1 Timing Colors (RGB tuples & Hex strings)
COLOR_PURPLE = (176, 38, 255)    # Fastest sector in session (#B026FF)
COLOR_GREEN  = (0, 230, 118)     # Personal best sector (#00E676)
COLOR_YELLOW = (255, 214, 0)     # Slower than personal best (#FFD600)
COLOR_WHITE  = (255, 255, 255)
COLOR_GREY   = (158, 158, 158)


class SectorTime:
    def __init__(self, time_s=0.0, status="yellow"):
        self.time_s = time_s
        self.status = status  # 'purple', 'green', 'yellow'

    @property
    def color(self):
        if self.status == "purple":
            return COLOR_PURPLE
        elif self.status == "green":
            return COLOR_GREEN
        elif self.status == "yellow":
            return COLOR_YELLOW
        return COLOR_WHITE

    def to_dict(self):
        return {
            "time_s": round(float(self.time_s), 3),
            "status": self.status,
        }


class LapRecord:
    def __init__(self, lap_number=1):
        self.lap_number = lap_number
        self.lap_time_s = 0.0
        self.sector_times = [None, None, None]  # S1, S2, S3
        self.valid = True  # Track limits violations invalidate lap time for pole/fastest lap


class DriverTiming:
    """
    Timing state for a single driver.
    """

    def __init__(self, car_id, code, name, team_code):
        self.car_id = car_id
        self.code = code
        self.name = name
        self.team_code = team_code

        self.current_lap = 1
        self.laps_completed = 0
        self.current_lap_start_step = 0
        self.last_lap_time_s = None
        self.best_lap_time_s = None

        # Personal best sectors
        self.pb_sectors = [None, None, None]
        # Current lap sectors
        self.current_sectors = [None, None, None]
        # Sector entry steps
        self.sector_entry_steps = [0, 0, 0]

        # 100 Checkpoint crossing timestamps (step index)
        # Format: dict mapping (lap_number, cp_index) -> step_index
        self.checkpoint_steps = {}

        # Last crossed checkpoint
        self.last_cp = -1
        self.total_checkpoints_passed = 0

        # Completed laps history
        self.lap_history = []


class TimingSystem:
    """
    Manages timing, intervals, gaps, and leaderboard positions for all cars in a session.
    """

    def __init__(self, fps=50, num_checkpoints=100):
        self.fps = fps
        self.num_checkpoints = num_checkpoints
        self.drivers = {}
        self.session_best_lap = None
        self.session_best_lap_driver = None
        # Session best sectors across all drivers
        self.session_best_sectors = [None, None, None]

    def register_driver(self, car_id, code, name, team_code):
        dt = DriverTiming(car_id, code, name, team_code)
        self.drivers[car_id] = dt
        return dt

    def update_driver_progress(self, car_id, current_step, progress_pct, lap_finished=False):
        """
        Update driver's timing state from their current continuous progress percentage.
        """
        dt = self.drivers[car_id]
        cp_idx = int((progress_pct / 100.0) * self.num_checkpoints) % self.num_checkpoints

        # Record checkpoint crossing
        if cp_idx != dt.last_cp:
            key = (dt.current_lap, cp_idx)
            if key not in dt.checkpoint_steps:
                dt.checkpoint_steps[key] = current_step
                dt.total_checkpoints_passed += 1

            # Sector transitions (Sector 1: 0-33%, Sector 2: 33-66%, Sector 3: 66-100%)
            # Sector 1 completes at ~33%
            s1_cp = int(self.num_checkpoints / 3.0)
            s2_cp = int(2.0 * self.num_checkpoints / 3.0)

            # Completing Sector 1
            if dt.last_cp < s1_cp and cp_idx >= s1_cp:
                s1_time = (current_step - dt.current_lap_start_step) / self.fps
                dt.sector_entry_steps[1] = current_step
                status = self._evaluate_sector(car_id, sector_idx=0, time_s=s1_time)
                dt.current_sectors[0] = SectorTime(s1_time, status)

            # Completing Sector 2
            elif dt.last_cp < s2_cp and cp_idx >= s2_cp:
                s2_time = (current_step - dt.sector_entry_steps[1]) / self.fps
                dt.sector_entry_steps[2] = current_step
                status = self._evaluate_sector(car_id, sector_idx=1, time_s=s2_time)
                dt.current_sectors[1] = SectorTime(s2_time, status)

            dt.last_cp = cp_idx

        # Lap completion
        if lap_finished:
            s3_time = (current_step - dt.sector_entry_steps[2]) / self.fps
            status = self._evaluate_sector(car_id, sector_idx=2, time_s=s3_time)
            dt.current_sectors[2] = SectorTime(s3_time, status)

            lap_time = (current_step - dt.current_lap_start_step) / self.fps
            dt.last_lap_time_s = lap_time
            dt.laps_completed += 1

            # Personal best lap check
            if dt.best_lap_time_s is None or lap_time < dt.best_lap_time_s:
                dt.best_lap_time_s = lap_time

            # Session best lap check
            if self.session_best_lap is None or lap_time < self.session_best_lap:
                self.session_best_lap = lap_time
                self.session_best_lap_driver = dt.code

            rec = LapRecord(dt.current_lap)
            rec.lap_time_s = lap_time
            rec.sector_times = list(dt.current_sectors)
            dt.lap_history.append(rec)

            # Reset for next lap
            dt.current_lap += 1
            dt.current_lap_start_step = current_step
            dt.sector_entry_steps = [current_step, current_step, current_step]
            dt.current_sectors = [None, None, None]
            dt.last_cp = 0

    def _evaluate_sector(self, car_id, sector_idx, time_s):
        """Determine purple, green, or yellow sector status."""
        dt = self.drivers[car_id]
        pb = dt.pb_sectors[sector_idx]
        sb = self.session_best_sectors[sector_idx]

        is_sb = (sb is None or time_s < sb)
        is_pb = (pb is None or time_s < pb)

        if is_sb:
            self.session_best_sectors[sector_idx] = time_s
            dt.pb_sectors[sector_idx] = time_s
            return "purple"
        elif is_pb:
            dt.pb_sectors[sector_idx] = time_s
            return "green"
        else:
            return "yellow"

    def compute_leaderboard(self, current_step):
        """
        Compute positions, intervals, and gaps.
        Order is determined by total checkpoints passed descending.
        """
        # Sort drivers by total checkpoints passed
        sorted_drivers = sorted(
            self.drivers.values(),
            key=lambda d: d.total_checkpoints_passed,
            reverse=True
        )

        leaderboard = []
        leader = sorted_drivers[0] if sorted_drivers else None

        for pos, dt in enumerate(sorted_drivers, start=1):
            gap_str = "LEADER" if pos == 1 else "+0.000"
            interval_str = "-" if pos == 1 else "+0.000"

            if pos > 1 and leader:
                # Calculate genuine gap to leader at most recent common checkpoint
                gap_val = self._compute_checkpoint_delta(leader, dt)
                car_ahead = sorted_drivers[pos - 2]
                interval_val = self._compute_checkpoint_delta(car_ahead, dt)

                gap_str = f"+{gap_val:.3f}" if gap_val is not None else "+LAP"
                interval_str = f"+{interval_val:.3f}" if interval_val is not None else "+LAP"

            leaderboard.append({
                "pos": pos,
                "car_id": dt.car_id,
                "code": dt.code,
                "name": dt.name,
                "team_code": dt.team_code,
                "laps": dt.laps_completed,
                "gap": gap_str,
                "interval": interval_str,
                "last_lap": f"{dt.last_lap_time_s:.3f}" if dt.last_lap_time_s else "--.---",
                "best_lap": f"{dt.best_lap_time_s:.3f}" if dt.best_lap_time_s else "--.---",
                "is_fastest_lap": (dt.code == self.session_best_lap_driver and self.session_best_lap is not None),
                "s1": dt.current_sectors[0],
                "s2": dt.current_sectors[1],
                "s3": dt.current_sectors[2],
            })

        return leaderboard

    def _compute_checkpoint_delta(self, car_fast, car_slow):
        """
        Compute true time difference (seconds) between car_fast and car_slow
        at the most recent checkpoint both cars have completed.
        """
        # Check backward from car_slow's last checkpoint
        slow_lap = car_slow.current_lap
        slow_cp = car_slow.last_cp

        for lookback in range(self.num_checkpoints):
            cp = (slow_cp - lookback) % self.num_checkpoints
            lap = slow_lap if cp <= slow_cp else (slow_lap - 1)
            key = (lap, cp)

            if key in car_slow.checkpoint_steps and key in car_fast.checkpoint_steps:
                delta_steps = car_slow.checkpoint_steps[key] - car_fast.checkpoint_steps[key]
                if delta_steps >= 0:
                    return delta_steps / self.fps

        # Fallback approximation if no common checkpoint recorded yet
        missing_cps = car_fast.total_checkpoints_passed - car_slow.total_checkpoints_passed
        avg_step_per_cp = 10.0  # approximate
        return max(0.0, (missing_cps * avg_step_per_cp) / self.fps)
