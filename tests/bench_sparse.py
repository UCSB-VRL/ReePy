import reepy
import numpy as np

import pyinstrument

# set up the data
N = 60 * 60 * 24 // 10 # 10 second sampling frequency
K = 33 # 33 trajectories
data = np.random.rand(K, N, 3) # 2D samples between 0 and 1
data[:, :, 0] = np.arange(N)/(60 * 60) # time in hours

reeb = reepy.SparseReebGraph(dim=2, epsilon=0.25) # ~15 minutes
with pyinstrument.profile():
    for i, traj in enumerate(data):
        reeb.append_trajectory(traj)

# report bundle statistics
bundle_count = len(reeb._bundles)
print(bundle_count, "bundles")
print(bundle_count / N, "bundles per index (SRG)")
