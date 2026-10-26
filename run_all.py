"""
run_all.py - Orchestrate the full experiment
Runs all training modes with seeds 0, 1, 2, plus reference evaluations.
Skips any step whose output already exists.
"""
import os, sys, subprocess, time
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("KERAS_BACKEND", "torch")


def run_cmd(cmd, desc=""):
    """Run a command and log output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {desc}")
    print(f"CMD: {cmd}")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    env = os.environ.copy()
    env["SDL_VIDEODRIVER"] = "dummy"
    env["KERAS_BACKEND"] = "torch"
    
    result = subprocess.run(
        cmd, shell=True, env=env,
        stdout=sys.stdout, stderr=sys.stderr,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    if result.returncode != 0:
        print(f"WARNING: Command exited with code {result.returncode}")
    
    return result.returncode


def main():
    t_start = time.time()
    python = os.path.join(".venv", "Scripts", "python.exe")
    
    # Stage 2: Data collection
    if not (os.path.exists("data/clean.npz") and os.path.exists("data/noisy.npz")):
        run_cmd(f"{python} collect.py", "Data Collection")
    else:
        print("Data already collected, skipping.")
    
    seeds = [0, 1, 2]
    
    # Stage 4: Train plain and aug models
    for mode in ["plain", "aug"]:
        for seed in seeds:
            model_path = f"models/{mode}_s{seed}.keras"
            if os.path.exists(model_path):
                print(f"Model {model_path} exists, skipping.")
                continue
            run_cmd(f"{python} train.py {mode} {seed}",
                   f"Train {mode} seed={seed}")
    
    # Stage 6: DAgger for each seed
    for seed in seeds:
        model_path = f"models/dagger_s{seed}.keras"
        if os.path.exists(model_path):
            print(f"DAgger model {model_path} exists, skipping.")
            continue
        run_cmd(f"{python} dagger.py {seed}",
               f"DAgger seed={seed}")
    
    # Stage 5: Evaluate all on TEST tracks
    # Reference drivers
    run_cmd(f"{python} evaluate.py expert test", "Evaluate expert")
    run_cmd(f"{python} evaluate.py zero test", "Evaluate zero")
    
    # All trained models
    for mode in ["plain", "aug", "dagger"]:
        for seed in seeds:
            name = f"{mode}_s{seed}"
            run_cmd(f"{python} evaluate.py {name} test",
                   f"Evaluate {name}")
    
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
