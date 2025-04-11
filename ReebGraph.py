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


    def __init__(self, dist=euclidean_distance, epsilon=1, store_trajectories=False, mask=np.nan):
        super().__init__()
        self.dist = dist
        self.epsilon = epsilon
        self.mask = mask

        if store_trajectories:
            self.trajectories = None
        else:
            self.trajectories = 0

        self.store_trajectories = store_trajectories

        self.bundles = None

    def __append(self, traj: np.ndarray):
        # TODO: move this outside the helper function for speed
        if self.bundles is None:
            self.bundles = [SimpleNamespace(centers=np.array([], ndmin=len(traj.shape) - 1), trajectories=[]) for _ in range(traj.shape[0])]

            if self.store_trajectories:
                self.trajectories = np.empty((0, *traj.shape))
        
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
        def __sort_bundles(bundles):
            if bundles.centers.shape[0] == 0:
                return

            bundle_order = np.argsort([len(trajectories) for trajectories in bundles.trajectories])
            bundles.centers = bundles.centers[bundle_order]
            bundles.trajectories = [bundles.trajectories[i] for i in bundle_order]

        nodes = [(0, *tuple(center)) for center in self.bundles[0].centers if not np.any(np.isnan(center))]
        connected_components = self.bundles[0].trajectories

        # manually map the trajectories to their respective bundles
        trajectory_count = self.trajectories.shape[0] if self.store_trajectories else self.trajectories
        bundles = [None] * trajectory_count
        for i, bundle in enumerate(self.bundles[0].trajectories):
            for traj in bundle:
                bundles[traj] = i

        for node in nodes:
            self.add_node(node)
        
        for t in range(1, len(self.bundles)):
            new_bundles = [None] * trajectory_count

            edges = np.empty((0, 2), dtype=int)
            reappear_events = []

            for i, bundle in enumerate(self.bundles[t].trajectories):
                for traj in bundle:
                    new_bundles[traj] = i
                
            for i, bundle in enumerate(self.bundles[t].trajectories):
                ip = bundles[bundle[0]] # where was this in the previous time step?

                # assumption: more trajectories than bundles
                if ip is not None and len(connected_components[ip]) == len(bundle) and all([t1 == t2 for t1, t2 in zip(connected_components[ip], bundle)]):
                    continue
                else:
                    for j, traj in enumerate(bundle):
                        if bundles[traj] is not None:
                            edges = np.vstack((edges, [bundles[traj], new_bundles[traj]]))
                        else:
                            # add this node in-place (re-appear event)
                            reappear_events.append((t, *tuple(self.bundles[t].centers[i])))

            # get unique rows of edges along with counts (for weights)
            unique_edges, counts = np.unique(edges, axis=0, return_counts=True)

            origin_nodes = np.unique(unique_edges[:, 0])
            destination_nodes = np.unique(unique_edges[:, 1])

            old_node_count = len(nodes)

            # create destination nodes
            for dest in destination_nodes:
                nodes.append((t, *tuple(self.bundles[t].centers[dest])))
                self.add_node((t, *tuple(self.bundles[t].centers[dest])))
            
            for node in reappear_events:
                nodes.append(node)
                self.add_node(node)
                print(node)
            
            new_node_count = len(nodes) - old_node_count

            print(f"Added {new_node_count} new nodes at time {t}")
            
            for edge in unique_edges:
                origin, destination = edge
                # print(f"[{nodes[origin]}] -> [{nodes[destination + old_node_count]}]")
            
            connected_components = self.bundles[t].trajectories
            bundles = new_bundles


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