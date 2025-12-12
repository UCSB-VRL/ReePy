from .ReebGraph  import ReebGraph
import numpy as np

from rtree import index
from collections import Counter


class SequentialReebGraph(ReebGraph):
    def __init__(self, epsilon=1e-5, store_trajectories=False, dist=None):
        super().__init__()

        if dist is not None:
            raise ValueError("Custom Distance Functions are not yet supported.")
        else:
            self.dist = lambda x, y: np.linalg.norm(x - y)

        self.epsilon = epsilon

        # bundles stores centroids and clusters
        self.bundles = None
        self.bundle_dict = {}

        # number of trajectories
        self.trajc = 0

        # length of each sequence
        self.L = None
        # dimension of each point
        self.D = None

        # store original trajectories within Reeb Graph
        self.store = store_trajectories
        self.trajectories = []

    def append_trajectory(self, traj: np.ndarray):
        if self.L is None:
            (L, D) = traj.shape
            self.L = L
            self.D = D

            # initialize the bundle data structures
            self.bundles = [
                index.Index() for _ in range(L)
            ]

        assert traj.shape == (self.L, self.D), "All trajectories must be same shape"

        # compute bundle counts
        for i, point in enumerate(traj):
            bbox = tuple(point) + tuple(point)
            nn = list(self.bundles[i].nearest(bbox, 1, objects=True))

            if len(nn) == 1:
                nn_point = nn[0].bbox[:self.D]
                if self.dist(nn_point, point) > self.epsilon:
                    # create a new bundle
                    self.bundles[i].insert(
                        len(self.bundle_dict),
                        bbox
                    )
                    self.bundle_dict[len(self.bundle_dict)] = [self.trajc]
                else:
                    # add to existing bundle
                    self.bundle_dict[nn[0].id].append(self.trajc)
            else:
                # always create a new bundle
                self.bundles[i].insert(
                    len(self.bundle_dict),
                    bbox
                )
                self.bundle_dict[len(self.bundle_dict)] = [self.trajc]


        if self.store:
            self.trajectories.append(traj)

        self.trajc += 1

    def append_trajectories(self, trajs):
        for traj in trajs:
            self.append_trajectory(traj)
        self.build()

    def build(self):
        # clear existing nodes and edges -- allows us to call this repeatedly to
        # plot node count over time
        self.clear()
        
        # assert at least one bundle exists
        assert self.trajc >= 1

        # initialize the state vector
        states = [None for _ in range(self.trajc)]
        active = {}

        bindex = self.bundles[0] 

        nodec = 0
        for bundle in bindex.intersection(bindex.bounds, objects=True):
            centroid = bundle.bbox[:self.D]
            time = 0
            trajs = self.bundle_dict[bundle.id]

            self.add_node(nodec, centroid=centroid, time=time, trajs=tuple(trajs))

            active[nodec] = tuple(trajs)

            # state of each trajectory
            for traj in trajs:
                states[traj] = nodec

            nodec += 1

        for rtime, bindex in enumerate(self.bundles[1:-1]):
            # rtime is relative to the loop => time = rtime + 1
            time = rtime + 1
            new_edges = []
            inactive_nodes = set()

            for bundle in bindex.intersection(bindex.bounds, objects=True):
                centroid = bundle.bbox[:self.D]
                curr_cc = tuple(self.bundle_dict[bundle.id])

                # check if there was a change in the connected components
                prev_cc = active[states[curr_cc[0]]]

                if curr_cc != prev_cc:
                    self.add_node(nodec, centroid=centroid, time=time, trajs=curr_cc)
                    active[nodec] = curr_cc

                    # replace active nodes
                    inactive_nodes |= {
                        states[traj] for traj in curr_cc
                    }

                    new_edges += [
                        (states[traj], nodec) for traj in curr_cc
                    ]

                    # update state -- we will never get this again since each
                    # trajectory belongs to exactly one bundle, so it won't
                    # appear in curr_cc during this iteration
                    for traj in curr_cc:
                        states[traj] = nodec

                    nodec += 1

            # update edges
            edges = Counter(new_edges)
            for edge, count in edges.items():
                weight = count / len(active[edge[0]])
                self.add_edge(edge[0], edge[1], weight=weight)

            # evict inactive nodes
            for node in inactive_nodes:
                active.pop(node)

        # add a node for a disappear event
        bindex = self.bundles[-1]
        time = len(self.bundles) - 1
        new_edges = []
        for bundle in bindex.intersection(bindex.bounds, objects=True):
            centroid = bundle.bbox[:self.D]
            curr_cc = tuple(self.bundle_dict[bundle.id])

            # always add a disappear node
            self.add_node(nodec, centroid=centroid, time=time, trajs=curr_cc)

            # compute edge to previous node
            new_edges += [
                (states[traj], nodec) for traj in curr_cc
            ]

            nodec += 1

        edges = Counter(new_edges)
        for edge, count in edges.items():
            weight = count / len(active[edge[0]])
            self.add_edge(edge[0], edge[1], weight=weight)

    def __getstate__(self):
        state = self.__dict__.copy()
        # remove problematic lambda functions
        if "dist" in state:
            del state["dist"]
        state["dist"] = True
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.__dict__.get("dist", False):
            self.dist = lambda x, y: np.linalg.norm(x - y)
