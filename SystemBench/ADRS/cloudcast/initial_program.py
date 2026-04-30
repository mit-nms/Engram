# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    for dst in dsts:
        path = nx.dijkstra_path(h, src, dst, weight="cost")
        for i in range(0, len(path) - 1):
            s, t = path[i], path[i + 1]
            for j in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])

    return bc_topology

# EVOLVE-BLOCK-END

class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, SingleDstPath] = None):
        self.src = src  # single str
        self.dsts = dsts  # list of strs
        self.num_partitions = num_partitions

        # dict(dst) --> dict(partition) --> list(nx.edges)
        # example: {dst1: {partition1: [src->node1, node1->dst1], partition 2: [src->dst1]}}
        if paths is not None:
            self.paths = paths
            self.set_graph()
        else:
            self.paths = {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

    def get_paths(self):
        print(f"now the set path is: {self.paths}")
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        """
        Set paths for partition = partition to reach dst
        """
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        """
        Append path for partition = partition to reach dst
        """
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)

# Helper functions that won't be evolved
def cluster_regions(G, num_clusters=20):
    """Cluster nodes using (avg in/out cost, in/out throughput) as features."""
    nodes = list(G.nodes)
    if len(nodes) <= num_clusters:
        return nodes

    features = []
    for n in nodes:
        out_edges = G.out_edges(n, data=True)
        in_edges = G.in_edges(n, data=True)
        out_cost = np.mean([d["cost"] for _, _, d in out_edges]) if out_edges else 0
        in_cost = np.mean([d["cost"] for _, _, d in in_edges]) if in_edges else 0
        out_bw = np.mean([d["throughput"] for _, _, d in out_edges]) if out_edges else 0
        in_bw = np.mean([d["throughput"] for _, _, d in in_edges]) if in_edges else 0
        features.append([out_cost, in_cost, out_bw, in_bw])

    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    labels = kmeans.fit_predict(features)
    representatives = []
    for i in range(num_clusters):
        cluster_nodes = [nodes[j] for j in range(len(nodes)) if labels[j] == i]
        representatives.append(cluster_nodes[0])
    return representatives

def build_subgraph(G, clustered_nodes, hop_limit=2):
    """Keep only edges among clustered nodes and limit hop distances."""
    H = G.subgraph(clustered_nodes).copy()
    to_remove = []
    for u, v in H.edges:
        try:
            if nx.shortest_path_length(H, u, v) > hop_limit:
                to_remove.append((u, v))
        except nx.NetworkXNoPath:
            to_remove.append((u, v))
    H.remove_edges_from(to_remove)
    return H

def update_capacities(G, P_sol, stripe_size, TIME):
    """Reduce edge and node capacities after each stripe iteration."""
    for (u, v), val in P_sol.items():
        if val > 0.5:
            # Only update if edge exists in graph
            if u in G.nodes and v in G.nodes and G.has_edge(u, v):
                if "throughput" in G[u][v]:
                    G[u][v]["throughput"] = max(0, G[u][v]["throughput"] - stripe_size / TIME)
    return G

def create_broadcast_topology(src: str, dsts: List[str], num_partitions: int = 4):
    """Create a broadcast topology instance"""
    return BroadCastTopology(src, dsts, num_partitions)

def run_search_algorithm(src: str, dsts: List[str], G, num_partitions: int):
    """Run the search algorithm and return the topology"""
    return search_algorithm(src, dsts, G, num_partitions)
