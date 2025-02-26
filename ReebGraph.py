from networkx.classes import DiGraph
from dataclasses import dataclass
import numpy as np

"""
Sequential Reeb Graph - assumes no explicit time dimension (fixed sampling rate)
TODO: generalize axis beyond default (0)
"""
class SequentialReebGraph(DiGraph):
    @staticmethod
    def euclidean_distance(x, y, axis=0):
        return np.linalg.norm(x - y, axis=axis)

    def __init__(self, dist=euclidean_distance, epsilon=0.3):
        super().__init__()
        self.dist = dist
        self.epsilon = epsilon

        self.trajectories = None

        self.bundle_centers = None 
        self.bundle_trajectories = None

    def append_trajectory(self, traj: np.ndarray):
        # append a trajectory to the reeb graph
        if self.trajectories is None:
            self.bundle_centers = [np.array([traj[i]], ndmin=1) for i in range(traj.shape[0])]
            self.bundle_trajectories = [[[0]] for i in range(traj.shape[0])]
            self.trajectories = traj.reshape(1, -1)
            return
        
        for t in range(traj.shape[0]):
            # find the bundle distances
            bundle_norms = self.dist(self.bundle_centers[t], traj[t].reshape(1, -1), axis=1)

            traj_bundle = np.argmin(bundle_norms)

            if bundle_norms[traj_bundle] < self.epsilon:
                # TODO: should we move the bundle around?
                self.bundle_trajectories[t][traj_bundle] = np.append(self.bundle_trajectories[t][traj_bundle], self.trajectories.shape[0])
            else:
                # create a new bundle
                self.bundle_centers[t] = np.vstack((self.bundle_centers[t], np.array([traj[t]], ndmin=1)))
                self.bundle_trajectories[t].append(np.array(self.trajectories.shape[0], ndmin=1))
            
        self.trajectories = np.vstack((self.trajectories, traj.reshape(1, -1)))
    
    def build_graph(self):
        # start by adding all the bundles at the start of the graph
        active_nodes = np.hstack((np.zeros((self.bundle_centers[0].shape[0], 1)), self.bundle_centers[0]))
        active_bundles = [tuple(bundle) for bundle in self.bundle_trajectories[0]]

        # print(type(active_bundles[0]))

        for row in active_nodes:
            self.add_node(tuple(row))

        for t in range(1, len(self.bundle_centers)):
            next_bundles_matrix = self.bundle_trajectories[t]
            next_bundles = [tuple(bundle) for bundle in next_bundles_matrix]

            old_bundles_index = []

            new_bundles = []
            new_nodes = np.empty((0, active_nodes.shape[1]))

            for i, bundle in enumerate(next_bundles):
                # Note: always add the last bundle
                if bundle not in active_bundles or t == len(self.bundle_centers) - 1:
                    # Add this bundle as a new node
                    new_node = np.hstack((t, self.bundle_centers[t][i]))
                    self.add_node(tuple(new_node))
                    # print(f"{t}: {bundle} not in active bundles. Adding new node {new_node}")

                    new_bundles.append(bundle)
                    new_nodes = np.vstack((new_nodes, new_node))


                    # find all bundles which this should replace
                    # for each trajectory, check all bundles in active_bundles and see if it matches
                    for traj in bundle: 
                        for j, active_bundle in enumerate(active_bundles):
                            if traj in active_bundle:
                                # print(f"Adding edge from {active_nodes[j]} to {new_node}")
                                self.add_edge(tuple(active_nodes[j]), tuple(new_node))
                                old_bundles_index.append(j)
                                break
                    
            # remove each of the old bundles, in reverse
            for j in np.unique(np.array(old_bundles_index))[::-1]:
                active_bundles.pop(j)
                active_nodes = np.delete(active_nodes, j, axis=0)

            # add the new bundles
            active_bundles += new_bundles
            active_nodes = np.vstack((active_nodes, new_nodes))
