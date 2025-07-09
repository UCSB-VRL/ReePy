import heapq
import numpy as np

class KDArray:
    def __init__(self, points, dist=None):
        self.points = points

        if dist is None:
            dist = lambda a, b: sum((a[i] - b[i]) ** 2 for i in range(len(a)))
        self.dist = dist

    def get_ball(self, center, radius):
        indices = []
        points = []

        for i, point in enumerate(self.points):
            if self.dist(center, point) <= radius ** 2:
                indices.append(i)
                points.append(point)
        
        return indices, points

# note: this clusterer is extremely slow and will not lead to optimal cluster formation
#       points should be a pandas dataframe of features
class NaiveClusterer:
    def __init__(self, points, epsilon, dist=None):
        self.points = points
        self.epsilon = epsilon

        if dist is None:
            dist = lambda a, b: np.sqrt(sum((a[i] - b[i]) ** 2) for i in range(len(a)))
        self.dist = dist
    
    def cluster(self):
        # returns a list of clusters (indices) and centroids
        unclustered = set(range(len(self.points)))

        clusters = []
        centroids = []

        while len(unclustered) > 0:
            index = unclustered.pop()
            cluster = [index]
            centroid = self.points.iloc[index]

            for i in list(unclustered):
                if self.dist(centroid, self.points.iloc[i]) <= self.epsilon:
                    cluster.append(i)
                    unclustered.remove(i)
            
            clusters.append(cluster)
            centroids.append(centroid)
        return clusters, centroids

    # incremental clustering algorithm
    def incr_cluster(self, points, init=None, index=0):
        if init is not None:
            new_clusters = init["clusters"]
            new_centroids = init["centroids"]
        else:
            new_clusters = []
            new_centroids = []

        for _, point in points.iterrows():
            found = False

            for i, centroid in enumerate(new_centroids):
                if self.dist(centroid, point) <= self.epsilon:
                    new_clusters[i].append(index)
                    found = True
                    break
            
            if not found:
                new_clusters.append([index])
                new_centroids.append(point)
        
        return new_clusters, new_centroids