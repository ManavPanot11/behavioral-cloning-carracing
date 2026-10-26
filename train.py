"""
train.py <mode> <seed>
Train a steering model using one of three recipes:
  plain: clean data, no augmentation
  aug:   noisy data + flip + straight-road balancing
  dagger: same as aug (DAgger rounds handled by dagger.py)
"""
import os, sys, json, time
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import build_model

# ── Seeds for reproducibility ────────────────────────────────────────
def set_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    # Keras / torch seeding
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    print(f"Seeds set to {seed} (python, numpy, keras backend)")


# ── Data loading ────────────────────────────────────────────────────
VAL_SEEDS = set(range(100, 108))

def load_data(mode, extra_obs=None, extra_steers=None):
    """Load train/val split. For 'plain', use clean.npz; for 'aug'/'dagger', use noisy.npz."""
    if mode == "plain":
        data = np.load("data/clean.npz")
    else:
        data = np.load("data/noisy.npz")
    
    obs = data["observations"]  # uint8
    steers = data["steers"]     # float32
    seeds = data["seeds"]       # int32
    
    # Split by seed
    val_mask = np.isin(seeds, list(VAL_SEEDS))
    train_mask = ~val_mask
    
    X_train, y_train = obs[train_mask], steers[train_mask]
    X_val, y_val = obs[val_mask], steers[val_mask]
    
    # Add extra data (for DAgger)
    if extra_obs is not None and extra_steers is not None:
        X_train = np.concatenate([X_train, extra_obs], axis=0)
        y_train = np.concatenate([y_train, extra_steers], axis=0)
    
    # Apply balancing (for aug/dagger only)
    if mode in ("aug", "dagger"):
        straight_mask = np.abs(y_train) < 0.05
        keep_mask = ~straight_mask | (np.random.random(len(y_train)) < 0.3)
        X_train = X_train[keep_mask]
        y_train = y_train[keep_mask]
        print(f"  Balancing: kept {keep_mask.sum()}/{len(keep_mask)} training frames "
              f"(dropped {(~keep_mask).sum()} straight frames)")
    
    print(f"  Training: {len(X_train)} frames, Validation: {len(X_val)} frames")
    return X_train, y_train, X_val, y_val


# ── Batch generator with augmentation ──────────────────────────────
class SteeringDataset(keras.utils.PyDataset):
    def __init__(self, X, y, batch_size=64, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(X))
        np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[batch_idx].astype(np.float32)
        y_batch = self.y[batch_idx].copy()
        
        if self.augment:
            # Random horizontal flip
            flip_mask = np.random.random(len(X_batch)) > 0.5
            X_batch[flip_mask] = X_batch[flip_mask, :, ::-1, :]
            y_batch[flip_mask] = -y_batch[flip_mask]
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# ── Training ────────────────────────────────────────────────────────
def train(mode, seed, extra_obs=None, extra_steers=None):
    """Train a model. Returns (model, history_dict)."""
    set_seeds(seed)
    
    print(f"\n{'='*60}")
    print(f"Training mode={mode}, seed={seed}")
    print(f"{'='*60}")
    
    # Load data
    X_train, y_train, X_val, y_val = load_data(mode, extra_obs, extra_steers)
    
    # Augment only for aug/dagger
    augment = mode in ("aug", "dagger")
    
    train_ds = SteeringDataset(X_train, y_train, batch_size=64, augment=augment)
    val_ds = SteeringDataset(X_val, y_val, batch_size=64, augment=False)
    
    # Build model
    model = build_model()
    
    # Callbacks
    model_path = f"models/{mode}_s{seed}.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
    ]
    
    t0 = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=30, callbacks=callbacks, verbose=1
    )
    elapsed = time.time() - t0
    print(f"\nTraining time: {elapsed:.1f}s")
    
    # Save model
    model.save(model_path)
    print(f"Saved model to {model_path}")
    
    # Validation MSE analysis
    val_pred = model.predict(X_val.astype(np.float32), verbose=0).flatten()
    model_mse = np.mean((val_pred - y_val) ** 2)
    zero_mse = np.mean(y_val ** 2)
    
    print(f"\nValidation MSE (model): {model_mse:.6f}")
    print(f"Validation MSE (always-0): {zero_mse:.6f}")
    print(f"Model beats always-0: {model_mse < zero_mse}")
    
    # Check overfitting
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    best_epoch = np.argmin(val_loss)
    
    if val_loss[-1] > train_loss[-1] * 2:
        print("WARNING: Possible overfitting (val_loss >> train_loss)")
    else:
        print("Training and validation loss fell together (no severe overfitting)")
    
    # Save loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_loss, label="Train Loss", color="blue")
    ax.plot(val_loss, label="Val Loss", color="orange")
    ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.5, label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Loss Curves: {mode} (seed {seed})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = f"results/loss_{mode}_s{seed}.png"
    plt.savefig(loss_path, dpi=100)
    plt.close()
    print(f"Saved {loss_path}")
    
    # Save history as JSON
    hist_path = f"results/history_{mode}_s{seed}.json"
    with open(hist_path, "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, f, indent=2)
    print(f"Saved {hist_path}")
    
    return model, history.history


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train.py <mode> <seed>")
        print("  mode: plain | aug | dagger")
        sys.exit(1)
    
    mode = sys.argv[1]
    seed = int(sys.argv[2])
    
    assert mode in ("plain", "aug", "dagger"), f"Unknown mode: {mode}"
    
    model_path = f"models/{mode}_s{seed}.keras"
    if os.path.exists(model_path):
        print(f"Model {model_path} already exists, skipping.")
        sys.exit(0)
    
    train(mode, seed)
