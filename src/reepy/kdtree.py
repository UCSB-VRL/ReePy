import heapq

class KDTree:
    def __init__(self, dim, dist=None):
        if dist is None:
            dist = lambda a, b: sum((a[i] - b[i]) ** 2 for i in range(len(a)))
        
        self.dim = dim
        self._root = None
        self.dist = dist

    def __iter__(self):
        return self._walk(self._root)

    def _make(self, points, i=0):
        if len(points) > 1:
            points.sort(key=lambda x: x[i])
            i = (i + 1) % self.dim
            m = len(points) >> 1
            return [self._make(points[:m], i), self._make(points[m + 1:], i),
                points[m]]
        if len(points) == 1:
            return [None, None, points[0]]
    
    def _add_point(self, node, point, i=0):
        if node is not None:
            dx = node[2][i] - point[i]
            for j, c in ((0, dx >= 0), (1, dx < 0)):
                if c and node[j] is None:
                    node[j] = [None, None, point]
                elif c:
                    self._add_point(node[j], point, (i + 1) % self.dim)

    def _get_knn(self, node, point, k, return_dist_sq, heap, i=0, tiebreaker=1):
        if node is not None:
            dist = self.dist(point, node[2])
            dx = node[2][i] - point[i]
            if len(heap) < k:
                heapq.heappush(heap, (-dist, tiebreaker, node[2]))
            elif dist < -heap[0][0]:
                heapq.heappushpop(heap, (-dist, tiebreaker, node[2]))
            i = (i + 1) % self.dim
            # Goes into the left branch, then the right branch if needed
            for b in (dx < 0, dx >= 0)[:1 + (dx * dx < -heap[0][0])]:
                self._get_knn(node[b], point, k, return_dist_sq, 
                    heap, i, (tiebreaker << 1) | b)
        if tiebreaker == 1:
            return [(-h[0], h[2]) if return_dist_sq else h[2] 
                for h in sorted(heap)][::-1]

    def _walk(self, node):
        if node is not None:
            for j in 0, 1:
                for x in self._walk(node[j]):
                    yield x
            yield node[2]

        
    def add_point(self, point):
        if self._root is None:
            self._root = [None, None, point]
        else:
            self._add_point(self._root, point)

    def get_knn(self, point, k, return_dist_sq=True):
        return self._get_knn(self._root, point, k, return_dist_sq, [])

    def get_nearest(self, point, return_dist_sq=True):
        l = self._get_knn(self._root, point, 1, return_dist_sq, [])
        return l[0] if len(l) else None
