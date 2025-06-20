from networkx import DiGraph
from typing import Iterable

class ReebGraph(DiGraph):
    def __init__(self):
        self.bundles = None

    def append_trajectory(self, traj):
        raise NotImplementedError()

    def append_trajectories(self, trajs: Iterable):
        for traj in trajs:
            self.append_trajectory(traj)
    
    def union(self, other):
        raise NotImplementedError()