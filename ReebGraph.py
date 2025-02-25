from networkx.classes import DiGraph
from dataclasses import dataclass
import numpy as np

"""
Sequential Reeb Graph - assumes no explicit time dimension (fixed sampling rate)
TODO: generalize axis beyond default (0)
"""
class SequentialReebGraph(DiGraph):
    def __init__(self, norm=np.linalg.norm, epsilon=0.3):
        super().__init__()
        self.norm = norm
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
            bundle_norms = self.norm(self.bundle_centers[t] - traj[t].reshape(1, -1), axis=1)

            traj_bundle = np.argmin(bundle_norms)

            if bundle_norms[traj_bundle] < self.epsilon:
                # TODO: should we move the bundle around?
                self.bundle_trajectories[t][traj_bundle] = np.append(self.bundle_trajectories[t][traj_bundle], self.trajectories.shape[0])
            else:
                # create a new bundle
                self.bundle_centers[t] = np.vstack((self.bundle_centers[t], np.array([traj[t]], ndmin=1)))
                self.bundle_trajectories[t].append(np.array(self.trajectories.shape[0], ndmin=1))
            
        self.trajectories = np.vstack((self.trajectories, traj.reshape(1, -1)))
    
    def _add_nodes_from_matrix(self, matrix):
        for row in matrix:
            self.add_node(tuple(row))

    def build_graph(self):
        # start by adding all the bundles at the start of the graph
        active_nodes = np.hstack((np.zeros((self.bundle_centers[0].shape[0], 1)), self.bundle_centers[0]))
        active_bundles = self.bundle_trajectories[0] 

        self._add_nodes_from_matrix(active_nodes)
