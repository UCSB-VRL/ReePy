import reepy
import numpy as np

def test_random():
    # 33 days with a 10 second SR for 2D points
    data = np.random.rand(33, 100, 2)

    reeb = reepy.SequentialReebGraph(epsilon=0.1)
    reeb.append_trajectories(data)

    # verify that sum of outbound edges of a node is (near) 1
    for n, w in reeb.out_degree(weight="weight"):
        if reeb.out_degree(n) > 0:             # has outgoing edges
            assert np.isclose(w, 1.0, rtol=1e-6, atol=1e-9), f"Node {n} sum={w}"

def test_1d_sequence_disconnect():
    data = np.array([
        [[3, 3, 3, 3, 3, 3, 3, 3]],
        [[3, 3, 3, 4, 4, 4, 4, 4]],
        [[3, 3, 2, 2, 2, 2, 2, 2]],
        [[3, 3, 2, 2, 1, 1, 1, 1]]
    ]).transpose(0, 2, 1) # (4, 8, 1)


    reeb = reepy.SequentialReebGraph(epsilon=1e-5)
    reeb.append_trajectories(data)

    solution = [
        (0 , [3.0], 0),
        (1 , [3.0], 2),
        (2 , [2.0], 2),
        (3 , [3.0], 3),
        (4 , [4.0], 3),
        (5 , [2.0], 4),
        (6 , [1.0], 4),
        (7 , [3.0], 7),
        (8 , [4.0], 7),
        (9 , [2.0], 7),
        (10, [1.0], 7),
    ]

    for node, data in reeb.nodes(data=True):
        assert (node, data.get("centroid"), data.get("time")) in solution

def test_1d_sequence():
    data = np.array([
        [[3, 3, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3]],
        [[1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 3, 3]],
    ]).transpose(0, 2, 1)

    reeb = reepy.SequentialReebGraph(epsilon=1e-5)
    reeb.append_trajectories(data)

    solution = [
        (0, [3.0], 0 ),
        (1, [1.0], 0 ),
        (2, [2.0], 4 ),
        (3, [2.0], 6 ),
        (4, [1.0], 6 ),
        (5, [3.0], 10),
        (6, [3.0], 11),
    ]

    for node, data in reeb.nodes(data=True):
        assert (node, data.get("centroid"), data.get("time")) in solution
