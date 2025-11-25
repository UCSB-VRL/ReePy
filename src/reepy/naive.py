from .spatial import SpatialDS
import numpy as np

from pyinstrument import profile

class NaiveDS(SpatialDS):
    def __init__(self, dist=lambda x, y: np.linalg.norm(x - y), 
                 points=None,
                 **kwargs):
        self.points = set()
        self.dist = dist

        if points is not None:
            for point in points:
                self.insert(point)

    def insert(self, point):
        self.points.add(tuple(point))

    def remove(self, point):
        if tuple(point) in self.points:
            self.points.remove(tuple(point))

    def nearest(self, point, k=1):
        if k > 1:
            raise ValueError("Unsupported K.")

        return np.array(next(iter(self.points)))

        index, distance = -1, np.inf
        nn = None

        for i, candidate in enumerate(self.points):
            if self.dist(np.array(candidate), point) < distance:
                index = i
                distance = self.dist(candidate, point)
                nn = candidate

        return np.array(candidate)

    def __len__(self):
        return len(self.points)
