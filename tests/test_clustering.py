import numpy as np
import pandas as pd
from reepy.SparseReebGraph import SparseReebGraph

long_trajs = [
    pd.DataFrame({'lat': np.random.rand(100), 'long': np.random.rand(100)}) for _ in range(66)
]

def dist(a, b):
    return np.sqrt(((a - b)**2).sum()).astype(np.float64)

def test_incr_reeb():
    # generate random trajectories of length 100, each as a pandas DataFrame
    trajs = [pd.DataFrame({'trajectory': np.random.rand(100)}) for _ in range(10)]

    all_trajs = pd.concat(trajs, ignore_index=True)

    reeb = SparseReebGraph(epsilon=0.1, dist=dist)
    reeb.append_trajectories(all_trajs)

    reeb_incr = SparseReebGraph(epsilon=0.1, dist=dist)
    reeb_incr.append_trajectories(pd.concat([traj.copy() for traj in trajs[:5]], ignore_index=True))
    reeb_incr.append_trajectories(pd.concat([traj.copy() for traj in trajs[5:]], ignore_index=True))

    for i in range(len(reeb.bundles["centroids"])):
        assert np.allclose(reeb.bundles["centroids"][i], reeb_incr.bundles["centroids"][i]), f"Centroid {i} does not match"