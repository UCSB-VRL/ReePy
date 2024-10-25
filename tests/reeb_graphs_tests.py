from ReebGraph import ReebGraph

import numpy as np
from networkx import MultiGraph


def test_reeb_graph_init():
    data = np.array([[1, 1, 1], [2, 2, 2]])
    reeb_graph = ReebGraph(data)
    assert isinstance(reeb_graph, ReebGraph)


def test_reeb_graph_nodes():
    data = np.array([[1, 1, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 1, 1, 1, 1, 1]])
    reeb_graph = ReebGraph(data)
    #assert reeb_graph.nodes() == [1, 2]
    pass