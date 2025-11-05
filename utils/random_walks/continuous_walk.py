import numpy as np


def generate_random_walk(X=[[0]], n_steps=50, step_scale=None, seed=None):
    X = np.array(X, dtype=float).reshape(-1, 1)
    rng = np.random.default_rng(seed)

    # Automatically infer step_scale if not provided
    if step_scale is None:
        value_range = np.ptp(X)  # max - min
        length = len(X)
        # Scale step size relative to variability and number of points
        step_scale = (value_range / max(length, 1)) if value_range > 0 else 1.0

    steps = rng.normal(0, step_scale, n_steps)
    walk = np.cumsum(steps) + X[-1][0]
    return walk.reshape(-1, 1)
