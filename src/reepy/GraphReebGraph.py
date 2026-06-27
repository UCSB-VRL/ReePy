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
                    if self.orderer(group[0]) == self.orderer(
                        node
                    ) and self.equivalence(group[0], node):
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
            rep_attrs = dict(source_graph.nodes[rep])
            rep_attrs.update(
                members=members,
                representative=rep,
                level=self.orderer(rep),
            )
            self.add_node(sid, **rep_attrs)
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

        # Reeb Graph pass
        def add_or_merge_edge(u: int, v: int, attrs: dict[str, Any]) -> None:
            if u == v:
                return
            if self.has_edge(u, v):
                if "weight" in attrs:
                    self[u][v]["weight"] = (
                        self[u][v].get("weight", 1.0) + attrs["weight"]
                    )
                return
            self.add_edge(u, v, **attrs)

        active: list[int] = [n for n in self.nodes if self.in_degree(n) == 0]
        for sid in list(nx.topological_sort(self)):
            if sid not in self:
                continue

            succs = list(self.successors(sid))
            preds = list(self.predecessors(sid))
            is_sink = not succs
            rep = self.nodes[sid]["representative"]

            equivalent_active = None
            for a in active:
                if (
                    a != sid
                    and a in self
                    and self.equivalence(self.nodes[a]["representative"], rep)
                ):
                    equivalent_active = a
                    break

            if equivalent_active is not None and not is_sink:
                for child in succs:
                    add_or_merge_edge(
                        equivalent_active,
                        child,
                        dict(self.get_edge_data(sid, child) or {}),
                    )
                if sid in active:
                    active.remove(sid)
                self.remove_node(sid)
                continue

            for p in preds:
                if p in active:
                    active.remove(p)
                    break
            if sid not in active:
                active.append(sid)

        # TODO: correctly define the trajectory count
        # self._trajectory_count += 1
