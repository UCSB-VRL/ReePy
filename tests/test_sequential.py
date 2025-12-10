import reepy
import numpy as np

def test_random():
    # 33 days with a 10 second SR for 2D points
    data = np.random.rand(33, 6 * 60 * 24, 2)

    reeb = reepy.SequentialReebGraph(epsilon=0.1)
    reeb.append_trajectories(data)
