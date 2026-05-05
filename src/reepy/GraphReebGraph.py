from collections import deque
from typing import Any, Iterable

import networkx as nx

from reepy.ReebGraph import ReebGraph

"""
GraphReebGraph efficiently constructs a 'Reeb Graph' given a directed graph as input
"""


class GraphReebGraph(ReebGraph):
    """
    orderer: f(x) returns a real, scalar value
    equivalence: f(x, y) returns True if two classes are equivalent,
                 False otherwise. Must be reflexive and symmetric
    """

    def __init__(self, orderer=None, equivalence=None):
        super().__init__()

        self.orderer = (lambda x: x) if orderer is None else orderer
        self.equivalence = (lambda x, y: x == y) if equivalence is None else equivalence

        self.bundles = {}
        self._trajectory_count = 0

    """
    traj: an iterable of points (input into orderer/equivalence). Conceptually,
          we consider these points to be "continuous" with respect to each other
    """

    def append_trajectory(self, traj, build=True):
        raise NotImplementedError("Unsupported behavior for GraphReebGraphs")

    def append_trajectories(self, trajs: nx.DiGraph):
        self.clear()
        self.bundles = {}

        source_graph = trajs.copy(as_view=False)
        visited: set[Any] = set()
        node_to_super: dict[Any, int] = {}
        queue: deque[int] = deque()
        supernode_count = 0

        def group_equivalent(nodes: list[Any]) -> list[list[Any]]:
            groups: list[list[Any]] = []
            for node in nodes:
                placed = False
                for group in groups:
                    if self.equivalence(group[0], node):
                        group.append(node)
                        placed = True
                        break
                if not placed:
                    groups.append([node])
            return groups

        def create_supernode(nodes: list[Any]) -> int:
            nonlocal supernode_count
            sid = supernode_count
            supernode_count += 1
            members = tuple(nodes)
            rep = members[0]
            self.add_node(
                sid, members=members, representative=rep, level=self.orderer(rep)
            )
            for node in members:
                visited.add(node)
                node_to_super[node] = sid
            queue.append(sid)
            return sid

        sources = [n for n in source_graph.nodes if source_graph.in_degree(n) == 0]
        for group in group_equivalent(sources):
            create_supernode(group)

        while queue:
            parent_sid = queue.popleft()
            members = self.nodes[parent_sid]["members"]
            next_nodes: list[Any] = []

            for node in members:
                for succ in source_graph.successors(node):
                    if succ in visited:
                        self.add_edge(parent_sid, node_to_super[succ])
                    else:
                        next_nodes.append(succ)

            if not next_nodes:
                continue

            seen: set[Any] = set()
            deduped = []
            for node in next_nodes:
                if node not in seen:
                    seen.add(node)
                    deduped.append(node)

            for group in group_equivalent(deduped):
                child_sid = create_supernode(group)
                self.add_edge(parent_sid, child_sid)

        remaining = [n for n in source_graph.nodes if n not in visited]
        for group in group_equivalent(remaining):
            create_supernode(group)

        # TODO: correctly define the trajectory count
        # self._trajectory_count += 1
