import reepy
import numpy as np

import pyinstrument

# set up the data
N = 60 * 60 * 24 // 10 # 10 second sampling frequency
K = 33 # 33 trajectories

data = np.empty((0, 4))
for k in range(K):
    length = np.random.randint(N * 0.9, N)
    traj = np.random.rand(length, 4)
    traj[:, 0] = k
    traj[:, 1] = np.arange(length)

    data = np.vstack((data, traj))

print(data.shape)

reeb = reepy.SparseReebGraph(dim=2, epsilon=0.25)
with pyinstrument.profile():
    reeb.append_trajectories(data)

# test incremental construction
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
