import os, time
from concurrent.futures import ProcessPoolExecutor

def worker(driver_info):
    driver_name, mode, seed_num = driver_info
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    from eval_suite import run_episode
    model = None
    driver_type = mode
    if mode in ["plain", "aug", "dagger"]:
        driver_type = "model"
        model_path = f"models/{driver_name}.keras"
        model = keras.models.load_model(model_path)
    res = run_episode(driver_type, model=model, seed=1000, max_steps=2000)
    return driver_name, res["lap_completed"], res["steps"], res["mean_centre_distance"]

def main():
    drivers = [
        ("expert", "expert", None),
        ("plain_s0", "plain", 0),
    ]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(worker, drivers))
    for r in results:
        print(f"Parallel Worker Result: {r}")
    print(f"Elapsed: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
