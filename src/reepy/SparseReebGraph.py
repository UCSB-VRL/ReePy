from .ReebGraph import ReebGraph
import pandas as pd

from .clustering import NaiveClusterer

class SparseReebGraph(ReebGraph):
    def __init__(self, clusterer=None, epsilon=1e-5, dist=None):
        super().__init__()

        if clusterer is None:
            clusterer = NaiveClusterer
        self.Clusterer = clusterer

        self.epsilon = epsilon
        self.dist = dist

    def append_trajectory(self, traj):
        raise NotImplementedError("SparseReebGraph does not support incremental updates (yet)")
    
    def append_trajectories(self, trajs):
        clusterer = self.Clusterer(trajs, self.epsilon, dist=self.dist)

        clusters, centroids = clusterer.cluster()

        self.bundles = {
            "centroids": centroids,
            "clusters": clusters
        }
    
