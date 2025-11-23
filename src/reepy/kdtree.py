from spatial import SpatialDS

class KDTree(SpatialDS):
    class Node:
        def __init__(self, data=None):
            self.data = data
            self.left = None
            self.right = None

    def __init__(self):
        self.root = None

    def insert(self, point):
        pass

    def remove(self, point):
        pass

    def nearest(self, point, k=1):
        pass
