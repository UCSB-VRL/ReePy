from networkx.classes import DiGraph
from dataclasses import dataclass
import numpy as np
from collections.abc import Iterable

"""
Sequential Reeb Graph - assumes no explicit time dimension (fixed sampling rate)
TODO: generalize axis beyond default (0)
"""
class SequentialReebGraph(DiGraph):
    @staticmethod
    def euclidean_distance(x, y, axis=0):
        return np.linalg.norm(x - y, axis=axis)

    def __init__(self, dist=euclidean_distance, epsilon=1, store_trajectories=False):
        super().__init__()
        self.dist = dist
        self.epsilon = epsilon

        if store_trajectories:
            self.trajectories = None
        else:
            self.trajectories = 0

        self.store_trajectories = store_trajectories


        self.bundle_centers = None 
        self.bundle_trajectories = None
    

    def __append(self, traj: np.ndarray):
        # append a trajectory to the reeb graph
        if (self.store_trajectories and self.trajectories is None) or (not self.store_trajectories and self.trajectories == 0):
            self.bundle_centers = [np.array([traj[i]], ndmin=1) for i in range(traj.shape[0])]
            self.bundle_trajectories = [[[0]] for i in range(traj.shape[0])]
            if self.store_trajectories:
                self.trajectories = traj.reshape(1, -1)
            else:
                self.trajectories += 1
            return
        
        for t in range(traj.shape[0]):
            # find the bundle distances
            bundle_norms = self.dist(self.bundle_centers[t], traj[t], axis=1)

            traj_bundle = np.argmin(bundle_norms)

            if bundle_norms[traj_bundle] < self.epsilon:
                # TODO: should we move the bundle around?
                self.bundle_trajectories[t][traj_bundle] = np.append(self.bundle_trajectories[t][traj_bundle], self.trajectories.shape[0] if self.store_trajectories else self.trajectories)
            else:
                # create a new bundle
                self.bundle_centers[t] = np.vstack((self.bundle_centers[t], np.array([traj[t]], ndmin=1)))
                self.bundle_trajectories[t].append(np.array(self.trajectories.shape[0] if self.store_trajectories else self.trajectories, ndmin=1))
            
        if self.store_trajectories:
            self.trajectories = np.vstack((self.trajectories, traj.reshape(1, -1)))
        else:
            self.trajectories += 1

    def __build_graph(self):
        # delete all nodes and edges
        self.clear()

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

    def append_trajectory(self, traj: np.ndarray, compute_graph=True):
        self.__append(traj)
        if compute_graph:
            self.__build_graph()
        
    def append_trajectories(self, trajs: Iterable, compute_graph=True): 
        for traj in trajs:
            self.__append(traj)
        if compute_graph:
            self.__build_graph()
    
    def union(self, other, compute_graph=True, adjust_epsilon=1):
        assert self.dist == other.dist, "Distance functions must be the same"
        assert self.epsilon == other.epsilon, "Epsilon must be the same" 
        assert self.store_trajectories == other.store_trajectories, "Flag store_trajectories must be the same"

        union = SequentialReebGraph(self.dist, self.epsilon, self.store_trajectories)

        if union.store_trajectories:
            assert self.trajectories.shape[1:] == other.trajectories.shape[1:], "All axes except sequential axis must match"

        if self.store_trajectories and other.store_trajectories:
            union.trajectories = np.vstack((self.trajectories, other.trajectories))
        else:
            union.trajectories = self.trajectories + other.trajectories 
        
        union.bundle_centers = self.bundle_centers
        union.bundle_trajectories = self.bundle_trajectories

        self_trajectory_count = self.trajectories.shape[0] if self.store_trajectories else self.trajectories
        
        # for each time step, find non-intersecting bundle centers
        # give preference to self's bundle centers
        for t in range(len(union.bundle_centers)):
            for center, trajs in zip(other.bundle_centers[t], other.bundle_trajectories[t]):
                # compute the distance between this center and all union bundle centers at this time
                distances = union.dist(union.bundle_centers[t], center, axis=1)
                if np.all(distances > union.epsilon * adjust_epsilon):
                    union.bundle_centers[t] = np.vstack((union.bundle_centers[t], center))
                    union.bundle_trajectories[t] += [np.array(trajs, ndmin=1) + self_trajectory_count]
                else:
                    # map this center to the nearest existing bundle
                    nearest_bundle = np.argmin(distances)
                    union.bundle_trajectories[t][nearest_bundle] = np.hstack((union.bundle_trajectories[t][nearest_bundle], np.array(trajs, ndmin=1) + self_trajectory_count))
        
        if compute_graph:
            union.__build_graph()

        return union