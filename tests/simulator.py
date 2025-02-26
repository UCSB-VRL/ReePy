import numpy as np

def brownian_motion(n, dims=2):
    x = np.zeros((n, dims))
    x[0] = np.random.normal(size=dims)

    for i in range(1, n):
        x[i] = x[i-1] + np.random.normal(size=dims)
    return x