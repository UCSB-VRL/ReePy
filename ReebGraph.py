import numpy as np
from networkx.classes.multigraph import MultiGraph

"""
Builds a ReebGraph given a set of trajectories, a norm, and a specified axis.
"""
class ReebGraph(MultiGraph):
    @staticmethod
    def tslicenorm(x):
        return np.linalg.norm(x[1:]) + (np.linalg.norm(x[0]) + 1)**np.inf - 1

    @staticmethod
    def euclidian(x, y):
        return np.linalg.norm(x - y)

    def __init__(self, data: np.ndarray, norm=np.linalg.norm, axis=0):
        super().__init__()
        self.data = data
        self.norm = norm
        self.axis = axis

    def compute_bundles(self):
        datapoints = np.array([]).reshape(0, self.data[0].shape[1] + 1)
        for i, traj in enumerate(self.data):
            explicit_label = np.repeat(i, traj.shape[0]).reshape(traj.shape[0], 1)
            datapoints = np.vstack((datapoints, np.hstack((explicit_label, traj))))

        # for any two points in datapoints, compute the distance between them
        points = datapoints[:, 1:]
        distances = np.apply_along_axis(self.norm, 2, points[:, np.newaxis] - points)
        
        # get list of connections on full dataset
        events = np.argwhere(np.tril((distances < 1) - np.diag(np.ones(distances.shape[0]))) == 1)

class BaselineSequentialReebGraph(ReebGraph):
    def __init__(self, data: np.ndarray, metric=np.linalg.norm, epsilon=0.1):
        self.data = data # data.reshape(data.shape[0], data.shape[1], 1) if data.ndim == 2 else data
        self.metric = np.vectorize(metric)
        self.epsilon = epsilon
    
        self.bundles = np.empty(self.data.shape[1], dtype=np.ndarray)
        self.compute_bundles()
    
    def compute_bundles(self):
        for index in range(self.data.shape[1]):
            # Note: current method greedily fixes the center of each cluster (not optimal bundles)
            slice = self.data[:, index]

            distances = np.apply_along_axis(self.metric, 1, slice[:, np.newaxis] - slice) < self.epsilon

            self.bundles[index] = {
                "points": [],
                "centers": []
            }

            for i in range(distances.shape[0]):
                bundle_indices = np.argwhere(distances[i])
                if bundle_indices.size > 0:
                    self.bundles[index]["centers"].append(slice[bundle_indices[0]])
                    # assign a bundle by going row by row
                    self.bundles[index]["points"].append(tuple(bundle_indices))
                    # remove these values from being bundled
                    distances[:, bundle_indices] = 0
    
    # builds a "conventional" reeb graph
    def build_graph(self):
        # add disappear events
        print(self.bundles[-1]["points"])
        self.add_nodes_from(self.bundles[-1]["points"])


class SequentialReebGraph(ReebGraph):
    def __init__(self, data: np.ndarray, metric=ReebGraph.euclidian, epsilon=0.1):
        super().__init__(data)

        self.data = data
        self.metric = np.vectorize(metric)
        self.epsilon = epsilon
    
        self.bundles = np.empty(self.data.shape[1], dtype=np.ndarray)
        self.compute_bundles()
        self.build_graph()
    
    def compute_bundles(self):
        for index in range(self.data.shape[1]):
            # Note: current method greedily fixes the center of each cluster (suboptimal bundles)
            slice = self.data[:, index]
            bundles = {
                "center": [],
                "points": []
            }

            for i, point in enumerate(slice):
                if len(bundles["center"]) == 0:
                    bundles["center"].append(point)
                    bundles["points"].append([i])
                else:
                    closest = np.argmin(self.metric(bundles["center"], point))
                    if self.metric(bundles["center"][closest], point) < self.epsilon:
                        bundles["points"][closest].append(i)
                    else:
                        bundles["center"].append(point)
                        bundles["points"].append([i])
            # flatten points
            for i in range(len(bundles["points"])):
                bundles["points"][i] = tuple(bundles["points"][i])

            self.bundles[index] = bundles
    
    def build_graph(self):
        pbundles = [(x,) for x in range(self.data.shape[0])]
        pbundle_LUT = [i for i in range(self.data.shape[0])]
        pedges = [[]] * len(pbundles)
        pcenters = [[]] * len(pbundles)

        # compute nodes in reverse
        for t in range(self.bundles.shape[0] - 2, 0, -1):
            cbundles = self.bundles[t]["points"]
            npedges = [[]] * len(cbundles)
            recent_nodes = []
            for j, cbundle in enumerate(cbundles):
                for traj in cbundle:
                    i = pbundle_LUT[traj]
                    pbundle_LUT[traj] = j

                    if cbundle == pbundles[i]:
                        for traj in cbundle:
                            npedges[j] = pedges[i]
                            pbundle_LUT[traj] = j
                        break
                    else:
                        new_node = (t + 1, *pbundles[i])
                        if new_node not in recent_nodes:
                            self.add_node(new_node, center=pcenters[i])
                            recent_nodes.append(new_node)

                            for edge in set(pedges[i]):
                                self.add_edge(new_node, edge)
                        npedges[j].append(new_node)
            pbundles = cbundles
            pcenters = self.bundles[t]["center"]
            pedges = npedges

import time

def test_reeb_graph_nodes():
    # data = np.array([
    #     [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2], 
    #     [1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1]
    #     ])
    # 5k points is roughly 1 day of data sampled at 15 second intervals
    data = np.array(np.random.rand(30, 5000, 1))

    starttime = time.time()
    reeb_graph = SequentialReebGraph(data)
    endtime = time.time()
    print(f"Number of bundles: {sum([len(bundle['points']) for bundle in reeb_graph.bundles])}")
    print(f"Average number of bundles: {sum([len(bundle['points'])/len(reeb_graph.bundles) for bundle in reeb_graph.bundles])}")
    print(f"Number of nodes: {len(reeb_graph.nodes())}")
    print(f"Number of edges: {len(reeb_graph.edges())}")

    print("Computation time: ", endtime - starttime)

import matplotlib.pyplot as plt

from tqdm import tqdm

def plot_performance():
    points = (np.linspace(10, 300, 10, dtype=int)**1.5).astype(int)
    seq_times = []
    base_times = []

    for n in tqdm(points, desc="SequentialReebGraph"):
        data = np.array(np.random.rand(30, n))
        starttime = time.time()
        reeb_graph = SequentialReebGraph(data)
        endtime = time.time()
        seq_times.append(endtime - starttime)

    for n in tqdm(points, desc="BaselineSequentialReebGraph"):
        data = np.array(np.random.rand(30, n))
        starttime = time.time()
        reeb_graph = BaselineSequentialReebGraph(data)
        endtime = time.time()
        base_times.append(endtime - starttime)

    plt.plot(points, seq_times, label='SequentialReebGraph')
    plt.plot(points, base_times, label='BaselineSequentialReebGraph')
    plt.xlabel('Number of Points')
    plt.ylabel('Computation Time (s)')
    plt.title('Performance Comparison')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # plot_performance()
    test_reeb_graph_nodes()

