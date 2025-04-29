from itertools import count
from quopri import decodestring
from networkx.classes import DiGraph
from dataclasses import dataclass
import numpy as np
from collections.abc import Iterable
from types import SimpleNamespace

"""
Sequential Reeb Graph - assumes no explicit time dimension (fixed sampling rate)
TODO: generalize axis beyond default (0)
"""
class SequentialReebGraph(DiGraph):
    @staticmethod
    def euclidean_distance(x, y, axis=0):
        # note: the distance function must handle masked trajectories (nan values) 
        return np.linalg.norm(x - y, axis=axis)


    """
    Sequential Reeb Graph constructor
    Parameters
    ----------
    dist : callable
        Distance function to use for computing distances between points.
        Default is euclidean_distance.
    epsilon : float
        Distance threshold for determining if two points are in the same bundle.
        Default is 1.
    store_trajectories : bool
        If True, store the trajectories in the graph.
        If False, store the number of trajectories.
        Default is False.
    trajectory_shape : tuple
        Shape of the trajectory data.
        If provided, initializes the graph with the specified shape. This is very important for improving graph computation speed.
    """
    def __init__(self, dist=euclidean_distance, epsilon=1, store_trajectories=False, 
                 trajectory_shape=None):
        super().__init__()
        self.dist = dist
        self.epsilon = epsilon

        if store_trajectories:
            self.trajectories = None
        else:
            self.trajectories = 0

        self.store_trajectories = store_trajectories

        self.bundles = None

        if trajectory_shape is not None:
            self.__initialize(trajectory_shape)
    
    def __initialize(self, traj_shape):
        self.bundles = [SimpleNamespace(centers=np.array([], ndmin=1), trajectories=[]) for _ in range(traj_shape[0])]
        self.trajectories = np.empty((0, *traj_shape)) if self.store_trajectories else 0

    def __append(self, traj: np.ndarray):
        if self.bundles is None:
            self.__initialize(traj.shape)
        
        trajectory_index = self.trajectories.shape[0] if self.store_trajectories else self.trajectories
        
        for t in range(traj.shape[0]):
            if np.any(np.isnan(traj[t])):
                continue
            elif self.bundles[t].centers.shape[0] > 0:
                bundle_norms = self.dist(self.bundles[t].centers, traj[t], axis=1)
                traj_bundle = np.argmin(bundle_norms)

                if bundle_norms[traj_bundle] < self.epsilon:
                    # TODO: adjust bundle centers dynamically based on epsilon
                    self.bundles[t].trajectories[traj_bundle] = np.append(self.bundles[t].trajectories[traj_bundle], trajectory_index)
                else:
                    # create a new bundle
                    self.bundles[t].centers = np.vstack((self.bundles[t].centers, np.array([traj[t]], ndmin=1)))
                    self.bundles[t].trajectories.append(np.array([trajectory_index], ndmin=1))
            else:
                self.bundles[t].centers = np.array([traj[t]], ndmin=1)
                self.bundles[t].trajectories = [np.array([trajectory_index], ndmin=1)]
            
        if self.store_trajectories:
            self.trajectories = np.vstack((self.trajectories, traj[np.newaxis, ...]))
        else:
            self.trajectories += 1

    def __build_graph(self):
        # initialize appear events
        traj_count = self.trajectories.shape[0] if self.store_trajectories else self.trajectories

        # initialize iteration variables
        concomp = self.bundles[0].trajectories

        nodes = [
            (0, *centroid) for centroid in self.bundles[0].centers
        ]
        for node_index, node in enumerate(nodes):
            self.add_node(node, trajectories=concomp[node_index])

        tb_lut = [np.nan] * traj_count
        for bidx, bundle in enumerate(concomp):
            for traj in bundle:
                tb_lut[traj] = bidx

        for t, bundles in enumerate(self.bundles[1:], start=1):
            new_concomp = bundles.trajectories
            new_nodes = []

            edges = np.empty((0, 2), dtype=int)

            new_tb_lut = [np.nan] * traj_count
            for bidx, bundle in enumerate(new_concomp):
                for traj in bundle:
                    new_tb_lut[traj] = bidx
                
                # look up which bundle this trajectory belonged to in the previous timestep
                pbidx = tb_lut[bundle[0]]

                if not np.isnan(pbidx) and len(concomp[pbidx]) == len(bundle) and all([bundle[_] == concomp[pbidx][_] for _ in range(len(bundle))]):
                    new_nodes.append(nodes[pbidx])
                else:
                    for traj in bundle:
                        edges = np.vstack((edges, (tb_lut[traj], bidx)))
                    new_nodes.append((t, *bundles.centers[bidx]))
        
            unique_edges, counts = np.unique(edges, axis=0, return_counts=True)

            origin_nodes, origin_node_counts = np.unique(edges[:, 0], return_counts=True)
            origin_nodes = np.column_stack((origin_nodes, origin_node_counts))
            origin_nodes = origin_nodes[~np.isnan(origin_nodes[:, 0])]

            origin_edgecount_vector = np.zeros(len(nodes))
            origin_edgecount_vector[origin_nodes[:, 0].astype(int)] = origin_nodes[:, 1].astype(int)

            destination_nodes = np.unique(unique_edges[:, 1])

            for node in destination_nodes:
                self.add_node(new_nodes[int(node)], trajectories=new_concomp[int(node)])  
            
            for edx, edge in enumerate(unique_edges):
                if not np.isnan(edge[0]):
                    self.add_edge(nodes[int(edge[0])], new_nodes[int(edge[1])], weight=counts[edx]/origin_edgecount_vector[int(edge[0])])
            
            concomp = new_concomp
            nodes = new_nodes
            tb_lut = new_tb_lut


    def append_trajectory(self, traj: np.ndarray, compute_graph=True):
        self.__append(traj)
        if compute_graph:
            self.__build_graph()
        
    def append_trajectories(self, trajs: Iterable, compute_graph=True): 
        for traj in trajs:
            self.__append(traj)
        if compute_graph:
            self.__build_graph()
    
    """
    Note: epsilon should be at least the sum of the two epsilon values
    """
    def append_reeb(self, other, epsilon, compute_graph=True):
        if epsilon is not None:
            self.epsilon = epsilon

        if self.bundles is None:
            self.__initialize(other.trajectories.shape[1:])

        traj_count = self.trajectories.shape[0] if self.store_trajectories else self.trajectories

        for t in range(len(other.bundles)):
            for bidx in range(len(other.bundles[t].trajectories)):
                bundle_trajs = [t + traj_count for t in other.bundles[t].trajectories[bidx]]
                bundle_center = other.bundles[t].centers[bidx]

                # for each bundle, check if it is within epsilon of any existing bundle
                if self.bundles[t].centers.shape[0] > 0:
                    bundle_norms = self.dist(self.bundles[t].centers, bundle_center, axis=1)
                    nearest_bundle = np.argmin(bundle_norms)

                    if bundle_norms[nearest_bundle] < self.epsilon:
                        self.bundles[t].trajectories[nearest_bundle] = np.append(self.bundles[t].trajectories[nearest_bundle], bundle_trajs)
                    else:
                        self.bundles[t].centers = np.vstack((self.bundles[t].centers, bundle_center))
                        self.bundles[t].trajectories.append(bundle_trajs)
                else:
                    # Highly unlikely for this to be called
                    self.bundles[t].centers = np.array([bundle_center], ndmin=1)
                    self.bundles[t].trajectories = [bundle_trajs]
        
        if self.store_trajectories and other.store_trajectories:
            self.trajectories = np.vstack((self.trajectories, other.trajectories))
        elif other.store_trajectories:
            self.trajectories += other.trajectories.shape[0]
        else:
            self.trajectories += other.trajectories
    
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