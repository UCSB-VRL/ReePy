from .spatial import SpatialDS
import numpy as np

class NaiveDS(SpatialDS):
    def __init__(self, dist=lambda x, y: np.linalg.norm(x - y)):
        self.points = []
        self.dist = dist

    def insert(self, point):
        self.points.append(point)

    def remove(self, point):
        self.points.remove(point)

    def nearest(self, point, k=1):
        if k > 1:
            raise ValueError("Unsupported K.")
        index = np.argmin([self.dist(point, x) for x in self.points])
        return index, self.points[index]
