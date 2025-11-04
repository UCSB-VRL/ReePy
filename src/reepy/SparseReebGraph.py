from .ReebGraph import ReebGraph

import numpy as np
from sortedcontainers import SortedList

def dist(x, y):
    # return np.sqrt(np.sum((x - y) ** 2))
    return np.linalg.norm(x - y)

class SparseReebGraph(ReebGraph):
    # TODO: better characterization of epsilon beyond spherical ball
    def __init__(self, dim, epsilon: float):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon

        # maintains a sorted list of bundles
        # bundles are (t, x, y, idx) 4-tuples
        # NOTE: consider quantization if epsilon is large
        self._bundles = SortedList(key=lambda x: x[0])
        # NOTE: find a better data structure for this
        self._bundle_indices = {}

        self.trajectory_count = 0

    # each trajectory is N rows, dim + 1 columns (time, ...)
    def append_trajectory(self, traj):
        self.trajectory_count += 1

        # NOTE: remove these asserts for faster evaluation
        assert traj.shape[1] == self.dim + 1

        for row in range(traj.shape[0]):
            # base case
            if len(self._bundles) == 0:
                bundle_index = len(self._bundle_indices)
                self._bundles.add((*traj[row], bundle_index))
                self._bundle_indices[bundle_index] = [self.trajectory_count]
                continue

            # TODO: incremental tree structure for querying
            t_min, t_max = traj[row][0] - self.epsilon, traj[row][0] + self.epsilon
            candidates = self._bundles.irange_key(min_key=t_min, max_key=t_max)

            new_bundle = True
            for bundle in candidates:
                if dist(bundle[:self.dim + 1], traj[row]) < self.epsilon:
                    # add to this bundle
                    self._bundle_indices[bundle[-1]].append(self.trajectory_count)
                    new_bundle = False
                    break

            if new_bundle:
                bundle_index = len(self._bundle_indices)
                self._bundles.add((*traj[row], bundle_index))
                self._bundle_indices[bundle_index] = [self.trajectory_count]
    
    # each trajectory is N rows, dim + 2 columns
    def append_trajectories(self, trajs):
        pass

