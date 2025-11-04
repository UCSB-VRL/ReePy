import reepy
import numpy as np

from matplotlib import pyplot as plt

def test_sparse():
    N = 10
    K = 5
    data = np.random.rand(K, N, 2)
    # integer time
    data[:, :, 0] = np.arange(N)/2

    reeb = reepy.SparseReebGraph(dim=1, epsilon=0.5)
    for traj in data:
        reeb.append_trajectory(traj)

    plt.figure()
    for i in range(K):
        plt.plot(data[i, :, 0], data[i, :, 1])

    for bundle in reeb._bundles:
        plt.scatter(bundle[0], bundle[1], color="black")

    plt.savefig("image.png")
