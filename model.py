"""
model.py - Nvidia-style CNN for steering prediction
Architecture adapted from "End to End Learning for Self-Driving Cars" (2016)
Input: 84x96x3 (cropped CarRacing observation, status bar removed)
Output: 1 (steering angle)
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import layers

def build_model():
    m = keras.Sequential([
        keras.Input(shape=(84, 96, 3)),
        layers.Rescaling(1/127.5, offset=-1),            # normalise pixels to [-1, 1]
        layers.Conv2D(24, 5, strides=2, activation="relu"),
        layers.Conv2D(36, 5, strides=2, activation="relu"),
        layers.Conv2D(48, 5, strides=2, activation="relu"),
        layers.Conv2D(64, 3, activation="relu"),
        layers.Conv2D(64, 3, activation="relu"),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(100, activation="relu"),
        layers.Dense(50, activation="relu"),
        layers.Dense(10, activation="relu"),
        layers.Dense(1),                                  # steering
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m

if __name__ == "__main__":
    m = build_model()
    m.summary()
    total_params = m.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Expected ~233k (article ~250k)")
