"""
f1sim/stewards.py - FIA/F1-inspired Stewards and Race Control Rules Engine.

Implements:
- RulesConfig: Configurable regulatory parameters.
- Stewards: Real-time adjudicator monitoring track limits, time penalties,
  black-and-white flags, jump starts, and DNF classifications.
- Race control broadcast notification stream.
"""

from collections import defaultdict


class RulesConfig:
    """
    Stewarding regulatory constants.
    """

    def __init__(
        self,
        max_track_limit_warnings=3,
        penalty_seconds_first=5.0,
        penalty_seconds_repeat=10.0,
        dnf_offroad_streak=150,
        jump_start_speed_threshold=1.0,
    ):
        self.max_track_limit_warnings = max_track_limit_warnings
        self.penalty_seconds_first = penalty_seconds_first
        self.penalty_seconds_repeat = penalty_seconds_repeat
        self.dnf_offroad_streak = dnf_offroad_streak
        self.jump_start_speed_threshold = jump_start_speed_threshold


class CarInfractionState:
    def __init__(self, car_id, code):
        self.car_id = car_id
        self.code = code
        self.offroad_streak = 0
        self.longest_offroad_streak = 0
        self.total_offroad_events = 0
        self.was_offroad = False
        self.track_limit_warnings = 0
        self.penalty_seconds = 0.0
        self.is_dnf = False
        self.dnf_reason = None
        self.black_and_white_flag = False
        self.jump_start = False


class RaceControlEvent:
    """
    Official notice from Race Control.
    """

    def __init__(self, step, time_s, car_id, code, level, message):
        self.step = step
        self.time_s = time_s
        self.car_id = car_id
        self.code = code
        self.level = level  # 'INFO', 'WARNING', 'PENALTY', 'FLAG', 'DNF'
        self.message = message

    def to_dict(self):
        return {
            "step": self.step,
            "time_s": round(self.time_s, 2),
            "car_id": self.car_id,
            "code": self.code,
            "level": self.level,
            "message": self.message,
        }


class Stewards:
    """
    Active adjudicator for race control.
    """

    def __init__(self, config=None, fps=50):
        self.config = config or RulesConfig()
        self.fps = fps
        self.states = {}
        self.event_log = []

    def register_car(self, car_id, code):
        self.states[car_id] = CarInfractionState(car_id, code)

    def check_step(self, step, car_id, is_offroad, speed, lights_out=True):
        """
        Evaluate single step state for a car.
        Returns list of new RaceControlEvents generated this step.
        """
        st = self.states[car_id]
        if st.is_dnf:
            return []

        time_s = step / self.fps
        new_events = []

        # 1. Jump Start Detection
        if not lights_out and speed > self.config.jump_start_speed_threshold and not st.jump_start:
            st.jump_start = True
            st.penalty_seconds += self.config.penalty_seconds_first
            ev = RaceControlEvent(
                step, time_s, car_id, st.code, "PENALTY",
                f"CAR {st.code} - FALSE START: +{self.config.penalty_seconds_first:.0f}S TIME PENALTY"
            )
            self.event_log.append(ev)
            new_events.append(ev)

        # 2. Track Limits & Offroad Streak Accounting
        if is_offroad:
            st.offroad_streak += 1
            if st.offroad_streak > st.longest_offroad_streak:
                st.longest_offroad_streak = st.offroad_streak

            # New off-road excursion event
            if not st.was_offroad:
                st.total_offroad_events += 1
                st.track_limit_warnings += 1

                if st.track_limit_warnings < self.config.max_track_limit_warnings:
                    ev = RaceControlEvent(
                        step, time_s, car_id, st.code, "WARNING",
                        f"CAR {st.code} - TRACK LIMITS WARNING ({st.track_limit_warnings}/{self.config.max_track_limit_warnings})"
                    )
                    self.event_log.append(ev)
                    new_events.append(ev)
                elif st.track_limit_warnings == self.config.max_track_limit_warnings:
                    st.black_and_white_flag = True
                    ev = RaceControlEvent(
                        step, time_s, car_id, st.code, "FLAG",
                        f"CAR {st.code} - BLACK & WHITE FLAG: FINAL WARNING FOR TRACK LIMITS"
                    )
                    self.event_log.append(ev)
                    new_events.append(ev)
                elif st.track_limit_warnings == self.config.max_track_limit_warnings + 1:
                    st.penalty_seconds += self.config.penalty_seconds_first
                    ev = RaceControlEvent(
                        step, time_s, car_id, st.code, "PENALTY",
                        f"CAR {st.code} - TRACK LIMITS: +{self.config.penalty_seconds_first:.0f}S TIME PENALTY"
                    )
                    self.event_log.append(ev)
                    new_events.append(ev)
                else:
                    st.penalty_seconds += self.config.penalty_seconds_repeat
                    ev = RaceControlEvent(
                        step, time_s, car_id, st.code, "PENALTY",
                        f"CAR {st.code} - REPEAT TRACK LIMITS: +{self.config.penalty_seconds_repeat:.0f}S TIME PENALTY"
                    )
                    self.event_log.append(ev)
                    new_events.append(ev)

            # DNF check: exceeding 150 consecutive offroad frames
            if st.offroad_streak >= self.config.dnf_offroad_streak:
                st.is_dnf = True
                st.dnf_reason = "EXCEEDED 150 CONSECUTIVE OFF-ROAD FRAMES"
                ev = RaceControlEvent(
                    step, time_s, car_id, st.code, "DNF",
                    f"CAR {st.code} - RETIRED (OFF TRACK > 150 FRAMES)"
                )
                self.event_log.append(ev)
                new_events.append(ev)

        else:
            st.offroad_streak = 0

        st.was_offroad = is_offroad
        return new_events

    def apply_penalty(self, car_id, penalty_seconds=5.0, reason="COLLISION", step=None):
        """
        Manually assess a time penalty to a car (e.g. from collision steward investigation).
        """
        if car_id in self.states:
            st = self.states[car_id]
            st.penalty_seconds += penalty_seconds
            step_val = step if step is not None else 0
            ev = RaceControlEvent(
                step_val, step_val / self.fps, car_id, st.code, "PENALTY",
                f"CAR {st.code} - {reason}: +{penalty_seconds:.0f}S TIME PENALTY"
            )
            self.event_log.append(ev)
            return ev
        return None

    def get_recent_notices(self, current_step, window_steps=200):
        """
        Return notices active within the last window_steps (for on-screen banners).
        """
        return [
            ev for ev in self.event_log
            if current_step - ev.step <= window_steps
        ]
