"""
verify_disturbance.py - Pre-registered check for Stage 4
Run zero and expert on 3 dev tracks (1000, 1001, 1002) with and without disturbance.
Report max centre-line distance with and without disturbance to confirm disturbance is real.
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
from eval_suite import run_episode

def main():
    print("=" * 65)
    print("STAGE 4: PRE-REGISTERED DISTURBANCE VERIFICATION (3 DEV TRACKS)")
    print("=" * 65)
    tracks = [1000, 1001, 1002]
    drivers = ["expert", "zero"]

    for track in tracks:
        for d in drivers:
            r_clean = run_episode(d, seed=track, disturb=False, max_steps=2000)
            r_dist = run_episode(d, seed=track, disturb=True, max_steps=2000)
            clean_max = r_clean["max_centre_distance"]
            dist_max = r_dist["max_centre_distance"]
            clean_mean = r_clean["mean_centre_distance"]
            dist_mean = r_dist["mean_centre_distance"]
            print(f"Track {track} | Driver {d:6s} | Clean Max: {clean_max:5.2f} (mean: {clean_mean:5.2f}) -> Disturbed Max: {dist_max:5.2f} (mean: {dist_mean:5.2f})")

if __name__ == "__main__":
    main()
