"""
graph.py - adjacency matrices for the two ST-GCN graphs (hand graph, body graph).

The edge lists live in skeleton_def.py (global joint numbers 0..49). Here we
turn them into the matrices an ST-GCN layer needs.

    hand graph : joints 0-41   -> 42 nodes (left hand 0-20, right hand 21-41)
    body graph : joints 42-49  ->  8 nodes (eyes, mouth, shoulders, elbows)

Inside each graph the nodes are renumbered from 0 ("local" numbering), because
the hand tensor only holds 42 joints and the body tensor only holds 8:

    hand local index = global index          (0..41)
    body local index = global index - 42     (0..7)

What you get from build_graphs():

    g = build_graphs()
    g["hand"]["A"]        (3, 42, 42)  ST-GCN "spatial configuration" partition
    g["hand"]["A_plain"]  (42, 42)     one normalized matrix (self-loops included)
    g["hand"]["num_nodes"], g["hand"]["edges"] (local numbering)
    g["body"][...]        same, 8 nodes

Convention: A[k][src, dst] is the weight with which node `src` feeds node `dst`.
Layer usage (as in the original ST-GCN):
    out[n, c, t, dst] = sum_k sum_src  X_k[n, c, t, src] * A[k][src, dst]

The 3 partitions (original ST-GCN paper, "spatial configuration"):
    k=0  the node itself + neighbours at the same distance from the root joint
    k=1  neighbours that are CLOSER to the root joint (wrist for hands,
         shoulders for the body)
    k=2  neighbours that are FARTHER from the root joint
Each column is normalized so that the weights feeding any node add up to 1.
"""

from collections import deque

import numpy as np

from skeleton_def import (
    HAND_GRAPH_EDGES, HAND_GRAPH_JOINTS,
    BODY_GRAPH_EDGES, BODY_GRAPH_JOINTS,
)

# Root joints in LOCAL numbering.
HAND_ROOTS = [0, 21]    # left wrist, right wrist
BODY_ROOTS = [4, 5]     # left shoulder, right shoulder (global 46, 47)


def to_local(edges, joints):
    """Global joint numbers -> 0-based numbers inside this graph."""
    start = joints[0]
    return [(a - start, b - start) for a, b in edges]


def adjacency_matrix(num_nodes, edges, self_loops=True):
    """Symmetric 0/1 adjacency matrix (bones are undirected)."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for a, b in edges:
        A[a, b] = 1.0
        A[b, a] = 1.0
    if self_loops:
        A += np.eye(num_nodes, dtype=np.float32)
    return A


def normalize_columns(A):
    """Divide every column by its sum, so the weights into a node add up to 1."""
    deg = A.sum(axis=0)
    deg[deg == 0] = 1.0
    return (A / deg).astype(np.float32)


def hop_distance(num_nodes, edges, roots):
    """Number of bones from the nearest root joint to every node (BFS)."""
    nbrs = [[] for _ in range(num_nodes)]
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    dist = [None] * num_nodes
    queue = deque()
    for r in roots:
        dist[r] = 0
        queue.append(r)
    while queue:
        u = queue.popleft()
        for v in nbrs[u]:
            if dist[v] is None:
                dist[v] = dist[u] + 1
                queue.append(v)
    if any(d is None for d in dist):
        raise ValueError("graph has nodes that cannot be reached from the roots")
    return np.array(dist)


def spatial_partition(num_nodes, edges, roots):
    """Return A with shape (3, V, V): self/same-distance, closer, farther."""
    A_total = adjacency_matrix(num_nodes, edges, self_loops=True)
    A_norm = normalize_columns(A_total)
    hop = hop_distance(num_nodes, edges, roots)

    A = np.zeros((3, num_nodes, num_nodes), dtype=np.float32)
    for src in range(num_nodes):
        for dst in range(num_nodes):
            if A_total[src, dst] == 0:
                continue
            if hop[src] == hop[dst]:
                k = 0
            elif hop[src] < hop[dst]:
                k = 1          # source is closer to the root than the target
            else:
                k = 2          # source is farther from the root
            A[k, src, dst] = A_norm[src, dst]
    return A


def _make(edges_global, joints, roots):
    edges = to_local(edges_global, joints)
    n = len(joints)
    return {
        "num_nodes": n,
        "edges": edges,
        "roots": roots,
        "A": spatial_partition(n, edges, roots),
        "A_plain": normalize_columns(adjacency_matrix(n, edges, self_loops=True)),
    }


def build_graphs():
    """Both graphs, ready to hand to the ST-GCN."""
    return {
        "hand": _make(HAND_GRAPH_EDGES, HAND_GRAPH_JOINTS, HAND_ROOTS),
        "body": _make(BODY_GRAPH_EDGES, BODY_GRAPH_JOINTS, BODY_ROOTS),
    }


if __name__ == "__main__":
    g = build_graphs()
    for name in ("hand", "body"):
        print(f"{name:5s}: nodes={g[name]['num_nodes']:2d}  "
              f"edges={len(g[name]['edges']):2d}  "
              f"A={g[name]['A'].shape}  A_plain={g[name]['A_plain'].shape}")
