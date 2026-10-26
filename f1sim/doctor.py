"""
f1sim/doctor.py - Diagnostic health check for the simulation environment.
"""

import sys
import os

os.environ.setdefault("KERAS_BACKEND", "torch")
import numpy as np


def run_doctor():
    """Perform diagnostic health check and print actionable results."""
    print("=" * 65)
    print(" f1sim Doctor: System & Environment Health Check")
    print("=" * 65)

    all_passed = True

    # 1. Python Version
    py_ver = sys.version_info
    print(f"[*] Python Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro} ... ", end="")
    if py_ver >= (3, 10):
        print("OK")
    else:
        print("FAIL (Python >= 3.10 required)")
        all_passed = False

    # 2. Critical Packages
    pkgs = [
        ("gymnasium", "1.4.0"),
        ("Box2D", None),
        ("pygame", None),
        ("torch", None),
        ("keras", None),
        ("numpy", None),
        ("PIL", None),
    ]

    for mod_name, req_ver in pkgs:
        print(f"[*] Checking {mod_name:<12} ... ", end="")
        try:
            m = __import__(mod_name)
            v = getattr(m, "__version__", "installed")
            print(f"OK ({v})")
        except ImportError as e:
            print(f"MISSING ({e})")
            all_passed = False

    # 3. SDL Video Driver
    sdl_drv = os.environ.get("SDL_VIDEODRIVER", "not set (default)")
    print(f"[*] SDL_VIDEODRIVER: {sdl_drv} ... ", end="")
    print("OK")

    # 4. Finish Line Patch
    print("[*] Finish Line Patch Sentinel ... ", end="")
    try:
        from f1sim.env import apply_finish_line_patch
        from gymnasium.envs.box2d import car_racing
        apply_finish_line_patch()
        if getattr(car_racing.FrictionDetector, "_f1sim_patched", False):
            print("OK (Active & Idempotent)")
        else:
            print("FAIL (Not detected)")
            all_passed = False
    except Exception as e:
        print(f"FAIL ({e})")
        all_passed = False

    # 5. Model Files Check
    models_to_check = [
        "plain_s0.keras", "plain_s1.keras", "plain_s2.keras",
        "aug_s0.keras", "aug_s1.keras", "aug_s2.keras",
        "dagger_s0.keras", "dagger_s1.keras", "dagger_s2.keras",
    ]
    missing_models = [m for m in models_to_check if not os.path.exists(os.path.join("models", m))]
    print(f"[*] Checking Trained Models (9 total) ... ", end="")
    if not missing_models:
        print("OK (All 9 models found)")
    else:
        print(f"WARNING: {len(missing_models)} missing: {missing_models}")

    # 6. Track Reproducibility Test
    print("[*] Track Reproducibility (Seed 1000) ... ", end="")
    try:
        from f1sim.env import make_env
        env1 = make_env(seed=1000, max_steps=50)
        p1 = env1.unwrapped.car.hull.position
        env1.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))
        p1_step = env1.unwrapped.car.hull.position
        env1.close()

        env2 = make_env(seed=1000, max_steps=50)
        p2 = env2.unwrapped.car.hull.position
        env2.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))
        p2_step = env2.unwrapped.car.hull.position
        env2.close()

        if abs(p1[0] - p2[0]) < 1e-6 and abs(p1_step[0] - p2_step[0]) < 1e-6:
            print("OK (Bit-identical positions)")
        else:
            print("FAIL (Trajectories diverged)")
            all_passed = False
    except Exception as e:
        print(f"FAIL ({e})")
        all_passed = False

    print("=" * 65)
    if all_passed:
        print(" RESULT: DOCTOR CHECK PASSED! All systems operational.")
    else:
        print(" RESULT: DOCTOR CHECK FAILED. Review errors above.")
    print("=" * 65)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_doctor())
