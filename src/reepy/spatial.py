"""
Spatial Data Structures (general use)
"""

class SpatialDS:
    def __init__(self):
        pass

    # insert point into structure 
    def insert(self, point):
        raise NotImplementedError()

    # remove point from structure
    def remove(self, point):
        raise NotImplementedError()

    # returns the nearest k points in the structure
    def nearest(self, point, k=1):
        raise NotImplementedError()

    # returns the number of points in the structure
    def __len__(self):
        pass


