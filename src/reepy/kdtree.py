from .spatial import SpatialDS
from dataclasses import dataclass
import numpy as np

@dataclass
class Node:
    data: np.ndarray
    left: Node
    right: Node
    axis: int

class KDTree(SpatialDS):
    def __init__(self, points=None, axes=None, 
                 dist=lambda x, y: np.linalg.norm(x - y),
                 **kwargs):
        self.root = None
        self.shape = points.shape
        self.dist = dist

        if points is None:
            return None

        # require points be a 2D numpy array
        assert len(points.shape) == 2

        if axes is None:
            axes = np.arange(points.shape[1], dtype=int) # add all axes to tree

        def build(points, depth=0):
            if points.shape[0] == 0:
                return None

            axis = axes[depth % axes.shape[0]]

            points_sorted = points[points[:, axis].argsort()]
            median = points.shape[0] // 2

            node = Node(
                data=points_sorted[median],
                left=build(points[:median], depth + 1),
                right=build(points[median+1:], depth+1),
                axis=axis
            )

            return node

        self.root = build(points)

    def insert(self, point):
        raise ValueError("KDTree incremental construction is unsupported.")

    def remove(self, point):
        raise ValueError("KDTree incremental construction is unsupported.")

    # TODO: early stopping with epsilon
    def nearest(self, point, k=1):
        def recursive_nearest(root, point, best=None):
            if root is None:
                return best

            if best is None or self.dist(point, root.data) < self.dist(point, best):
                best = root.data

            axis = root.axis

            if point[axis] < root.data[axis]:
                next_branch = root.left
                opp_branch = root.right
            else:
                next_branch = root.right
                opp_branch = root.left

            best = recursive_nearest(next_branch, point, best)

            # TODO: generalize the case of searching the other branch
            # if best is None or (point[axis] - root.point[axis]) ** 2 < distance_squared(point, best):
            #   best = nearest_neighbor(opposite_branch, point, best)

            if best is None:
                best = recursive_nearest(opp_branch, point, best)

            return best

        if k > 1:
            raise ValueError("k >= 1 is not supported")

        return recursive_nearest(self.root, point)


    def __len__(self):
        return self.shape[0]
