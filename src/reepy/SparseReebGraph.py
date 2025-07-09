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

        self.clusterer = None
        self.bundles = None
        self.trajc = 0

    def append_trajectory(self, traj):
        raise NotImplementedError("SparseReebGraph does not support incremental updates (yet)")
    
    def append_trajectories(self, trajs):
        if self.clusterer is None:
            self.clusterer = self.Clusterer(trajs, self.epsilon, dist=self.dist)
            clusters, centroids = self.clusterer.cluster()
        else:
            clusters, centroids = self.clusterer.incr_cluster(trajs, init=self.bundles, index=self.trajc)

        self.trajc += len(trajs)
        self.bundles = {
            "centroids": centroids,
            "clusters": clusters
        }