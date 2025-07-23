from .ReebGraph import ReebGraph
from .clustering import NaiveClusterer
import pandas as pd
import numpy as np

class SequentialReebGraph(ReebGraph):
    def __init__(self, clusterer=None, epsilon=1e-5, dist=None, store_trajectories=False):
        super().__init__()

        # Set up the clusterer object
        if clusterer is None:
            clusterer = NaiveClusterer
        self.Clusterer = clusterer

        self.epsilon = epsilon
        self.dist = dist

        # bundles stores centroids and clusters
        self.bundles = None
        # number of trajectories 
        self.trajc = 0
        # sequence length
        self.seq_len = None

        self.store = store_trajectories
        self.trajectories = []

    
    def append_trajectory(self, traj: pd.DataFrame):
        if self.store:
            if self.trajectories is None:
                self.trajectories = [traj]
            else:
                self.trajectories.append(traj)

        if self.seq_len is None:
            self.seq_len = traj.shape[0]

            # there will be one bundle at each timestep 
            self.bundles = [
                {"clusters": [], "centroids": []} for _ in range(self.seq_len)
            ]
        
        # inject the seq_index into the trajectory (note - this is equivalent to index)
        # this will be used to create unique nodes
        traj["seq_index"] = np.arange(self.seq_len)
        
        for i in range(self.seq_len):
            point = traj.iloc[i]
            
            min_dist = np.inf
            min_index = -1

            # find the closest centroid to the current point
            for j, centroid in enumerate(self.bundles[i]["centroids"]):
                d = self.dist(point, centroid)
                if d < self.epsilon and d < min_dist:
                    min_dist = d
                    min_index = j
                    # by construction, there should only be one such j
            
            if min_index == -1:
                # create a new centroid
                self.bundles[i]["centroids"].append(point)
                self.bundles[i]["clusters"].append([self.trajc])
            else:
                self.bundles[i]["clusters"][min_index].append(self.trajc)

        self.trajc += 1
    
    def append_trajectories(self, trajs):
        for traj in trajs:
            self.append_trajectory(traj)
    
    def build(self):
        centroids = self.bundles[0]["centroids"]
        clusters = self.bundles[0]["clusters"]
        state = np.zeros((self.trajc,), dtype=int) - 1

        for i, centroid in enumerate(centroids):
            node = tuple(centroid)
            self.add_node(node)
            for traj in self.bundles[0]["clusters"][i]:
                state[traj] = i
        
        edge_dict = None

        for t in range(1, self.seq_len):
            new_centroids = self.bundles[t]["centroids"]
            new_clusters = self.bundles[t]["clusters"]
            new_state = np.zeros((self.trajc,), dtype=int) - 1

            edge_dict = {}

            for i, centroid in enumerate(new_centroids):
                for traj in new_clusters[i]:
                    new_state[traj] = i
            
            visited_vector = np.zeros_like(new_state, dtype=bool)

            for traj_i in range(self.trajc):
                if visited_vector[traj_i]:
                    # this trajectory has already been processed
                    continue
                
                # compare states
                pc = clusters[state[traj_i]]
                cc = new_clusters[new_state[traj_i]]

                if pc != cc or t == self.seq_len - 1:
                    # make cc a new node
                    node = tuple(new_centroids[new_state[traj_i]])
                    self.add_node(node)

                    for traj in cc:
                        src = tuple(centroids[state[traj]])
                        srcsize = len(clusters[state[traj]])
                        dst = tuple(new_centroids[new_state[traj]])

                        if (src, dst) not in edge_dict:
                            edge_dict[(src, dst)] = 1./srcsize
                        else:
                            edge_dict[(src, dst)] += 1./srcsize
                        visited_vector[traj] = True
                else:
                    new_centroids[new_state[traj_i]] = centroids[state[traj_i]]
                    for traj in cc:
                        visited_vector[traj] = True
                
            # flush the edge dict
            for (src, dst), weight in edge_dict.items():
                self.add_edge(src, dst, weight=weight)
            
            clusters = new_clusters
            centroids = new_centroids
            state = new_state