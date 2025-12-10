import reepy
import numpy as np

from matplotlib import pyplot as plt

def notest_sparse():
    N = 60 * 60 * 24 / 10
    K = 5
    data = np.random.rand(K, N, 3)
    # fix the time axes 
    data[:, :, 0] = np.arange(N)/60

    reeb = reepy.SparseReebGraph(dim=1, epsilon=0.5)
    for traj in data:
        reeb.append_trajectory(traj)

