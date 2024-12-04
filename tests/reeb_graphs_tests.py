from ReebGraph import ReebGraph

import numpy as np
from networkx import MultiGraph


def test_reeb_graph_init():
    data = np.array([[1, 1, 1], [2, 2, 2]])
    data_t = np.array([zip(range(traj.shape[0]), traj) for traj in data])
    reeb_graph = ReebGraph(data_t)
    assert isinstance(reeb_graph, ReebGraph)


def test_reeb_graph_nodes():
    data = np.array([[1, 1, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 1, 1, 1, 1, 1]])
    data_t = np.array([np.vstack((range(traj.shape[0]), traj)) for traj in data])
    reeb_graph = ReebGraph(data)
    # assert reeb_graph.nodes() == [1, 2]
    pass
