from .ReebGraph import ReebGraph

import numpy as np
from sortedcontainers import SortedList

# TODO: replace with KD-Tree
from .naive import NaiveDS
from .kdtree import KDTree

def dist(x, y):
    return np.linalg.norm(x[:-1] - y[:-1])

class SparseReebGraph(ReebGraph):
    # TODO: better characterization of epsilon beyond spherical ball
    def __init__(self, dim, epsilon: float):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon

        # self.DS = NaiveDS
        self.DS = KDTree

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

        # initialize the data structure
        # sstruct = self.DS()
        t1_min, t1_max = None, None

        for row in range(traj.shape[0]):
            # base case
            if len(self._bundles) == 0:
                bundle_index = len(self._bundle_indices)
                self._bundles.add((*traj[row], bundle_index))
                self._bundle_indices[bundle_index] = [self.trajectory_count]
                continue

            t2_min, t2_max = traj[row][0] - self.epsilon, traj[row][0] + self.epsilon

            # non-incremental construction approach
            # candidates = np.array(list(self._bundles.irange_key(min_key=t2_min,
            #                                                     max_key=t2_max)))

            candidates = np.array(list(self._bundles))

            if len(candidates.shape) != 2:
                candidates = candidates.reshape(-1, 1)

            sstruct = self.DS(
                points=candidates,
                # last axis is spatially insignificant
                axes=np.arange(traj.shape[1] - 1, dtype=int),
                dist=dist
            )

            # incremental data structure construction only
            # if t1_min is None or t1_max is None:
            #     candidates = self._bundles.irange_key(min_key=t2_min, 
            #                                           max_key=t2_max)
            #
            #     for bundle in candidates:
            #         sstruct.insert(np.array(bundle))
            #
            # else:
            #     old_candidates = self._bundles.irange_key(min_key=t1_min,
            #                                               max_key=min(t2_min,
            #                                                           t1_max))
            #     new_candidates = self._bundles.irange_key(min_key=max(t1_max,
            #                                                           t2_min),
            #                                               max_key=t2_max)
            #
            #     # print("Inserting", len(list(new_candidates)))
            #     # print("Removing", len(list(old_candidates)))
            #
            #     for bundle in old_candidates:
            #         sstruct.remove(np.array(bundle))
            #     for bundle in new_candidates:
            #         sstruct.insert(np.array(bundle))
            #
            #     # print("Size of struct:", len(sstruct))
            #
            # t1_min, t1_max = t2_min, t2_max

            if len(sstruct) > 0:
                traj_np = np.array((*traj[row], 0.))
                nearest_bundle = sstruct.nearest(traj_np)
                distance = dist(nearest_bundle, traj_np)
            else:
                distance = np.inf

            if distance < self.epsilon:
                # merge bundles
                self._bundle_indices[int(nearest_bundle[-1])].append(self.trajectory_count)
            else:
                # add a new bundle
                bundle_index = len(self._bundle_indices)
                self._bundles.add((*traj[row], bundle_index))
                self._bundle_indices[bundle_index] = [self.trajectory_count]
    
    # each trajectory is N rows, dim + 2 columns
    def append_trajectories(self, trajs):
        _, masks = np.unique(trajs[:, 0], return_index=True)
        trajs_split = np.split(trajs, masks[1:])

        for traj in trajs_split:
            # remove the trajectory dimension
            self.append_trajectory(traj[:, 1:])

        self.build()

    # once trajectories are added, compute the reeb graph
    def build(self):
        self.clear()

        states = [None] * self.trajectory_count
        state_nodes = [None] * self.trajectory_count

        edge_count = 0

        for bundle in self._bundles:
            # get the trajectories present in this bundle
            cindices = self._bundle_indices[bundle[-1]]
            # get the trajectories present in the previous bundle
            pindices = self._bundle_indices[states[cindices[0] - 1]] \
                if states[cindices[0] - 1] is not None\
                else None

            if cindices != pindices:
                # create a new node
                self.add_node(bundle[-1], label=bundle[:-1])

                mask_indices = set()

                for traj in cindices:
                    # check if this trajectory has appeared -- if not, mark it
                    if states[traj - 1] is not None:
                        # if not, appear event (no edge needed)
                        mask_indices.add(states[traj - 1])
                    states[traj - 1] = bundle[-1]

                for dst in mask_indices:
                    self.add_edge(bundle[-1], dst)

