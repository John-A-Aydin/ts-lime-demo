import numpy as np


def generate_random_walk(n_steps=50, step_scale=1.0, seed=None):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, step_scale, n_steps)
    return np.cumsum(steps).reshape(-1, 1)
