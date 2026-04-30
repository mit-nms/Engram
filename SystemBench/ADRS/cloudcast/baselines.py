import networkx as nx
import subprocess
import os
import inspect
import re
from broadcast import BroadCastTopology
from networkx.algorithms.tree.branchings import Edmonds
from utils import append_src_dst_paths
import json
from typing import Dict, List
import numpy as np


# EVOLVE-BLOCK-START

def Cloudcast_OPT(src, dsts, G, num_partitions, TIME=60, TRANSFER_SIZE=300):
    """
    Cloudcast MILP optimizer with approximations.
    - src: source node
    - dsts: list of destination nodes
    - G: networkx.DiGraph with edges (cost, throughput)
    - num_partitions: number of stripes
    - TIME: time budget (seconds)
    - TRANSFER_SIZE: total data to transfer (GB)
    """
    from pulp import LpVariable, LpProblem, LpMinimize, lpSum, LpBinary, LpInteger, PulpSolverError
    from sklearn.cluster import KMeans
    import random
    num_clusters_default = 20
    hop_limit_default = 2

    def cluster_regions(G, num_clusters=num_clusters_default):
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


    def build_subgraph(G, clustered_nodes, hop_limit=hop_limit_default):
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

    STRIPES = num_partitions
    stripe_size = TRANSFER_SIZE / STRIPES

    # === Step 1. Node Clustering ===
    clustered_nodes = cluster_regions(G)
    # clustered_nodes = random_cluster_regions(G)
    # Ensure source and all destinations are in the clustered nodes
    # Only include nodes that actually exist in G
    clustered_nodes = [n for n in clustered_nodes if n in G.nodes]
    clustered_nodes = list(set(clustered_nodes + [src] + [d for d in dsts if d in G.nodes]))
    # Final check: ensure all clustered nodes exist in G
    clustered_nodes = [n for n in clustered_nodes if n in G.nodes]

    # CRITICAL: Verify all clustered nodes exist in G before building subgraph
    # This prevents any phantom nodes from appearing
    clustered_nodes = [n for n in clustered_nodes if n in G.nodes]

    G_sub = build_subgraph(G, clustered_nodes, hop_limit=hop_limit_default)

    # CRITICAL: Verify all nodes in G_sub exist in G
    # Remove any nodes that somehow don't exist in G
    invalid_nodes = [n for n in G_sub.nodes if n not in G.nodes]
    if invalid_nodes:
        G_sub.remove_nodes_from(invalid_nodes)

    # Assign default VM parameters if missing
    # vm_cost: cost per second (0.54 per hour / 3600 = 0.00015 per second, matching BCSimulator)
    # vm_limit: default to 2 VMs per region (matching typical num_vms in evaluator)
    providers = ["aws", "gcp", "azure"]
    provider_ingress = [10, 16, 16]
    provider_egress = [5, 7, 16]
    ingress_limits = {providers[i]: provider_ingress[i] for i in range(len(providers))}
    egress_limits = {providers[i]: provider_egress[i] for i in range(len(providers))}

    for n in G_sub.nodes:
        provider = str(n).split(":")[0]
        G_sub.nodes[n]["egress_limit"] = egress_limits[provider]
        G_sub.nodes[n]["ingress_limit"] = ingress_limits[provider]
        G_sub.nodes[n]["vm_cost"] = 0.00015  # Cost per second (matching BCSimulator), 0.54 per hour / 3600 = 0.00015 per second
        G_sub.nodes[n]["vm_limit"] = 2  # Default to 2 VMs per region (matching typical usage)

    bc_topology = BroadCastTopology(src, dsts, STRIPES)

    for s in range(STRIPES):
        try:
            prob = LpProblem(f"CloudcastStripe{s}", LpMinimize)

            # === Decision Variables ===
            P = {(u, v): LpVariable(f"P_{s}_{u}_{v}", 0, 1, LpBinary) for u, v in G_sub.edges()}
            N = {v: LpVariable(f"N_{s}_{v}", 0, G_sub.nodes[v]["vm_limit"], LpInteger) for v in G_sub.nodes}
            
            # Flow variables for each destination to ensure path connectivity
            # F[d, u, v] = 1 if edge (u,v) is used in path to destination d
            from pulp import LpContinuous
            F = {
                (d, u, v): LpVariable(f"F_{s}_{d}_{u}_{v}", 0, 1, LpContinuous)
                for d in dsts if d in G_sub.nodes
                for (u, v) in G_sub.edges()
            }

            # === Objective ===
            inst_cost = lpSum(G_sub.nodes[v]["vm_cost"] * N[v] * TIME for v in G_sub.nodes)
            egress_cost = lpSum(G_sub[u][v]["cost"] * P[(u, v)] * stripe_size for u, v in G_sub.edges)
            prob += inst_cost + egress_cost

            # === Constraints ===
            for u, v in G_sub.edges:
                bw = G_sub[u][v]["throughput"]
                prob += stripe_size * P[(u, v)] <= bw * N[u] * TIME, f"throughput_{u}_{v}"

            for v in G_sub.nodes:
                egress_lim = G_sub.nodes[v]["egress_limit"] * N[v] * TIME
                ingress_lim = G_sub.nodes[v]["ingress_limit"] * N[v] * TIME
                prob += lpSum(P[(v, w)] * stripe_size for _, w in G_sub.out_edges(v)) <= egress_lim
                prob += lpSum(P[(u, v)] * stripe_size for u, _ in G_sub.in_edges(v)) <= ingress_lim

            # === Flow conservation constraints for path connectivity ===
            # For each destination, ensure a connected path from src to dst
            for d in dsts:
                if d not in G_sub.nodes:
                    continue
                if src not in G_sub.nodes:
                    continue
                    
                # Flow can only use selected edges
                for (u, v) in G_sub.edges():
                    prob += F[(d, u, v)] <= P[(u, v)], f"flow_on_edge_{d}_{u}_{v}"
                
                # Flow conservation: for each node v
                for v in G_sub.nodes:
                    inflow = lpSum(F[(d, u, v)] for (u, _) in G_sub.in_edges(v))
                    outflow = lpSum(F[(d, v, w)] for (_, w) in G_sub.out_edges(v))
                    
                    if v == src:
                        # Source: net outflow = 1
                        prob += outflow - inflow == 1, f"flow_src_{d}"
                    elif v == d:
                        # Destination: net inflow = 1
                        prob += inflow - outflow == 1, f"flow_dst_{d}"
                    else:
                        # Intermediate: flow conservation (inflow = outflow)
                        prob += inflow - outflow == 0, f"flow_cons_{d}_{v}"

            status = prob.solve()

            # Check if solver found a solution
            if status != 1:  # 1 = Optimal, -1 = Infeasible, -2 = Unbounded, etc.
                print(f"Solver status for stripe {s}: {status} (1=Optimal)")
                continue

            # === Collect results ===
            P_sol = {k: v.value() for k, v in P.items() if v.value() is not None and v.value() > 0.5}
            G_sub = update_capacities(G_sub, P_sol, stripe_size, TIME)

            # Build a graph from selected edges and find paths to all destinations
            # Only use edges that exist in the original graph G
            # CRITICAL: Filter P_sol to only include edges where both nodes exist in G
            # P_sol contains edges from G_sub, but we must verify they exist in original G
            valid_P_sol = {}
            for (u, v) in P_sol.keys():
                # CRITICAL: Verify nodes exist in G (not G_sub) and edge exists in G
                # G_sub is a subgraph, but edges might have been filtered/modified
                if u not in G.nodes or v not in G.nodes:
                    continue
                if not G.has_edge(u, v):
                    continue
                # Additional check: verify we can actually access the edge
                try:
                    _ = G[u][v]
                    valid_P_sol[(u, v)] = P_sol[(u, v)]
                except (KeyError, TypeError):
                    continue

            selected_graph = nx.DiGraph()
            # CRITICAL: Only add edges where BOTH nodes definitely exist in G
            # This prevents any phantom nodes from appearing in paths
            for (u, v) in valid_P_sol.keys():
                # Verify nodes exist in G (not just G_sub)
                if u not in G.nodes or v not in G.nodes:
                    continue
                # Verify edge exists in G
                if not G.has_edge(u, v):
                    continue
                try:
                    edge_data = G[u][v].copy()
                    # Ensure we're not copying any invalid references
                    if "cost" in edge_data and "throughput" in edge_data:
                        selected_graph.add_edge(u, v, **edge_data)
                except (KeyError, TypeError):
                    # Edge access failed, skip this edge
                    continue

            # CRITICAL: Remove any nodes from selected_graph that don't exist in G
            # This is a safety net in case something slipped through
            nodes_to_remove = [n for n in selected_graph.nodes if n not in G.nodes]
            if nodes_to_remove:
                selected_graph.remove_nodes_from(nodes_to_remove)

            # CRITICAL: Also remove any edges where nodes don't exist in G
            # This ensures selected_graph only contains valid edges
            edges_to_remove = []
            for (u, v) in list(selected_graph.edges()):
                if u not in G.nodes or v not in G.nodes or not G.has_edge(u, v):
                    edges_to_remove.append((u, v))
            if edges_to_remove:
                selected_graph.remove_edges_from(edges_to_remove)

            # For each destination, find paths from src and assign to this partition
            for dst in dsts:
                # Ensure both src and dst are in the selected graph and original graph
                if src not in G.nodes or dst not in G.nodes:
                    continue
                # CRITICAL: Verify src and dst are in selected_graph AND all their neighbors exist in G
                if dst not in selected_graph.nodes or src not in selected_graph.nodes:
                    continue
                # Additional check: ensure src and dst are valid strings
                if not isinstance(src, str) or not isinstance(dst, str):
                    continue
                try:
                    # Try to find shortest path from src to dst in selected graph
                    if nx.has_path(selected_graph, src, dst):
                        path = nx.shortest_path(selected_graph, src, dst)
                        # CRITICAL: Validate every single node in path exists in G
                        # This is the root cause - paths can contain nodes not in G
                        valid_path = True
                        for node in path:
                            if node not in G.nodes:
                                valid_path = False
                                break

                        if valid_path:
                            # Add each edge in the path to the broadcast topology
                            for i in range(len(path) - 1):
                                u, v = path[i], path[i + 1]
                                # Triple-check: nodes exist AND edge exists
                                if u not in G.nodes or v not in G.nodes:
                                    valid_path = False
                                    break
                                if not G.has_edge(u, v):
                                    valid_path = False
                                    break
                                try:
                                    # ABSOLUTE FINAL CHECK: Verify nodes exist in G before accessing edge
                                    if u not in G.nodes or v not in G.nodes:
                                        valid_path = False
                                        break
                                    if not G.has_edge(u, v):
                                        valid_path = False
                                        break
                                    edge = G[u][v]
                                    # Final safety check: ensure edge data is valid and is a dict
                                    if not isinstance(edge, dict):
                                        valid_path = False
                                        break
                                    if "cost" not in edge or "throughput" not in edge:
                                        valid_path = False
                                        break
                                    # Verify u and v are strings (not None or other types)
                                    if not isinstance(u, str) or not isinstance(v, str) or u is None or v is None:
                                        valid_path = False
                                        break
                                    # Only add if we've passed all checks - create a clean copy of edge data
                                    # Final validation: ensure u and v are non-empty strings
                                    u_str = str(u).strip()
                                    v_str = str(v).strip()
                                    if not u_str or not v_str or u_str == "None" or v_str == "None":
                                        valid_path = False
                                        break
                                    try:
                                        cost_val = float(edge["cost"]) if edge["cost"] is not None else float('inf')
                                        throughput_val = float(edge["throughput"]) if edge["throughput"] is not None else 0.0
                                        edge_copy = {"cost": cost_val, "throughput": throughput_val}
                                        bc_topology.append_dst_partition_path(dst, s, [u_str, v_str, edge_copy])
                                    except (ValueError, TypeError, KeyError):
                                        valid_path = False
                                        break
                                except (KeyError, TypeError):
                                    # Edge access failed, skip this path
                                    valid_path = False
                                    break

                        if not valid_path:
                            # Path contains invalid nodes/edges, skip it
                            pass
                    else:
                        # If no path exists, try to use edges that might help reach the destination
                        # This is a fallback - ideally the MILP should ensure connectivity
                        # Use valid_P_sol instead of P_sol to ensure all edges are valid
                        for (u, v) in valid_P_sol.keys():
                            # ABSOLUTE FINAL CHECK before adding
                            if u not in G.nodes or v not in G.nodes:
                                continue
                            if not G.has_edge(u, v):
                                continue
                            if not (v == dst or u == src):
                                continue
                            try:
                                if u not in G.nodes or v not in G.nodes:
                                    continue
                                if not G.has_edge(u, v):
                                    continue
                                edge = G[u][v]
                                # Final safety check: ensure edge data is valid and is a dict
                                if not isinstance(edge, dict):
                                    continue
                                if "cost" not in edge or "throughput" not in edge:
                                    continue
                                # Verify u and v are strings
                                if not isinstance(u, str) or not isinstance(v, str):
                                    continue
                                # Create a clean copy of edge data
                                edge_copy = {"cost": edge["cost"], "throughput": edge["throughput"]}
                                bc_topology.append_dst_partition_path(dst, s, [u, v, edge_copy])
                            except (KeyError, TypeError):
                                # Edge access failed, skip this edge
                                continue
                except (nx.NodeNotFound, nx.NetworkXNoPath, KeyError) as e:
                    # Fallback: add any edges that might be useful, but only if they exist in G
                    # Use valid_P_sol instead of P_sol to ensure all edges are valid
                    for (u, v) in valid_P_sol.keys():
                        # ABSOLUTE FINAL CHECK before adding
                        if u not in G.nodes or v not in G.nodes:
                            continue
                        if not G.has_edge(u, v):
                            continue
                        if not (v == dst or u == src):
                            continue
                        try:
                            edge = G[u][v]
                            # Final safety check: ensure edge data is valid
                            if "cost" in edge and "throughput" in edge:
                                u_str = str(u).strip()
                                v_str = str(v).strip()
                                if u_str and v_str and u_str != "None" and v_str != "None":
                                    try:
                                        cost_val = float(edge["cost"]) if edge["cost"] is not None else float('inf')
                                        throughput_val = float(edge["throughput"]) if edge["throughput"] is not None else 0.0
                                        edge_copy = {"cost": cost_val, "throughput": throughput_val}
                                        bc_topology.append_dst_partition_path(dst, s, [u_str, v_str, edge_copy])
                                    except (ValueError, TypeError, KeyError):
                                        pass
                        except (KeyError, TypeError):
                            # Edge access failed, skip this edge
                            continue

        except PulpSolverError as e:
            print(f"Solver failed for stripe {s}: {e}")
            continue

    return bc_topology

def N_dijkstra(src, dsts, G, num_partitions):
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


def N_direct(src, dsts, G, num_partitions):
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    for dst in dsts:
        edge = G[src][dst]
        for j in range(bc_topology.num_partitions):
            bc_topology.set_dst_partition_paths(dst, j, [[src, dst, edge]])

    return bc_topology


def MDST(src, dsts, G, num_partitions):
    # Construct MDST path
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    DST_graph = Edmonds(h.subgraph([src] + dsts))
    opt_DST = DST_graph.find_optimum(attr="cost", kind="min", preserve_attrs=True, style="arborescence")
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Construct MDST graph
    MDST_graph = nx.DiGraph()
    for edge in list(opt_DST.edges()):
        s, d = edge[0], edge[1]
        MDST_graph.add_edge(s, d, **G[s][d])

    return append_src_dst_paths(src, dsts, MDST_graph, bc_topology), MDST_graph


def MULTI_MDST(src, dsts, G, num_partitions):
    # Construct MDST path based on original graph
    h = G.copy()
    MDST_graphs = []
    while len(list(h.edges())) > 0:
        _, MDST_graph = MDST(src, dsts, h, 1)
        print("MDST graph: ", MDST_graph.edges.data())
        MDST_graphs.append(MDST_graph)
        h.remove_edges_from(list(MDST_graph.edges()))

    print("Number of MDSTs: ", len(MDST_graphs))


def Min_Steiner_Tree(src, dsts, G, num_partitions, hop_limit=3000):
    # TODO: need to download the dependencies first
    source_v, dest_v = src, dsts

    h = G.copy()
    h.remove_edges_from(list(h.in_edges(source_v)) + list(nx.selfloop_edges(h)))

    nodes, edges = list(h.nodes), list(h.edges)
    num_nodes, num_edges = len(nodes), len(edges)
    id_to_name = {nodes.index(n) + 1: n for n in nodes}

    config_loc = "write.set"
    write_loc = "test.stplog"
    param_loc = "test.stp"

    with open(config_loc, "w") as f:
        f.write('stp/logfile = "use_probname"')
        f.close()

    _default_scipstp = os.path.join(os.path.dirname(__file__), "scipoptsuite-9.2.4", "build", "bin", "applications", "scipstp")
    scipstp_bin = os.environ.get("SCIPSTP_PATH", _default_scipstp)
    command = scipstp_bin + " "
    command += f"-f {param_loc} -s {config_loc} -l {write_loc}"

    def construct_stp():
        Edge_info = []
        actual_edge_count = 0
        for edge in edges:
            s, d = nodes.index(edge[0]) + 1, nodes.index(edge[1]) + 1
            cost = h[edge[0]][edge[1]]["cost"]
            if cost is None:
                # Skip edges without cost
                continue
            Edge_info.append(f"A {s} {d} {cost}\n")
            actual_edge_count += 1

        section_begin = '33D32945 STP File, STP Format Version 1.0\n\nSECTION Comment\nName "Relay: cloud regions"\nCreator "S. Liu"\n'
        section_begin += f'Remark "Cloud region problem adapted from relay"\nEND\n\nSECTION Graph\n'
        section_begin += f"Nodes {num_nodes}\nEdges {actual_edge_count}\nHopLimit {hop_limit}\n"
        Edge_info.append("END\n")

        # Ensure source and destinations are in nodes list
        if source_v not in nodes:
            raise ValueError(f"Source node {source_v} not found in graph after removing in-edges")
        for d in dest_v:
            if d not in nodes:
                raise ValueError(f"Destination node {d} not found in graph")

        s = nodes.index(source_v) + 1
        v = [nodes.index(i) + 1 for i in dest_v]
        terminal_info = [f"T {i}\n" for i in v]
        terminal_info.append("END\n\nEOF")
        section_terminal = f"""\nSECTION Terminals\nRoot {s}\nTerminals {len(dest_v)}\n"""

        with open(param_loc, "w") as f:
            f.write(section_begin)
            for edge in Edge_info:
                f.write(edge)
            f.write(section_terminal)
            for t in terminal_info:
                f.write(t)
            f.close()
        return

    def read_result(loc):
        di_stree_graph = nx.DiGraph()
        if not os.path.exists(loc):
            return di_stree_graph

        with open(loc, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("E") and len(line.split()) == 3:
                    l = line.split()
                    try:
                        src_id, dst_id = int(l[1]), int(l[2])
                        if src_id in id_to_name and dst_id in id_to_name:
                            src_r, dst_r = id_to_name[src_id], id_to_name[dst_id]
                            if src_r in G and dst_r in G[src_r]:
                                di_stree_graph.add_edge(src_r, dst_r, **G[src_r][dst_r])
                    except (ValueError, KeyError, IndexError) as e:
                        # Skip invalid lines
                        continue

        # Ensure source node is in the graph (add it even if isolated)
        if src not in di_stree_graph:
            di_stree_graph.add_node(src)

        # overlays = [node for node in di_stree_graph.nodes if node not in [source_v]+dest_v]
        return di_stree_graph

    construct_stp()  # construct problem to a file
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )  # run the steiner tree solver
    process.wait()

    # Check if SCIP ran successfully
    if process.returncode != 0:
        stderr_output = process.stderr.read().decode() if process.stderr else ""
        print(f"Warning: SCIP solver returned non-zero exit code {process.returncode}")
        print(f"SCIP stderr: {stderr_output[:500]}")  # Print first 500 chars of error

    solution_graph = read_result(loc=write_loc)

    # If solution graph is empty, has no edges, missing source, or missing destinations, fall back to MDST
    has_edges = len(solution_graph.edges()) > 0
    has_all_dests = all(dst in solution_graph for dst in dsts)
    needs_fallback = len(solution_graph.nodes) == 0 or src not in solution_graph or not has_edges or not has_all_dests

    if needs_fallback:
        print(f"Warning: SCIP solver produced empty solution or missing source node. Falling back to MDST.")
        # Use MDST as fallback - use the same logic as the MDST function
        h_fallback = G.copy()
        h_fallback.remove_edges_from(list(h_fallback.in_edges(src)) + list(nx.selfloop_edges(h_fallback)))

        # Check if all destinations are reachable
        missing_dsts = [d for d in dsts if d not in h_fallback]
        if missing_dsts:
            print(f"Warning: Some destinations not in graph: {missing_dsts}")

        # Use Edmonds algorithm - same approach as MDST function
        # Note: Edmonds on subgraph should work, but if it fails, fall back to Dijkstra
        try:
            subgraph_nodes = [src] + [d for d in dsts if d in h_fallback]
            DST_graph = Edmonds(h_fallback.subgraph(subgraph_nodes))
            opt_DST = DST_graph.find_optimum(attr="cost", kind="min", preserve_attrs=True, style="arborescence")
            MDST_graph = nx.DiGraph()

            # Ensure source node is in the graph first
            MDST_graph.add_node(src)

            # Add edges from the optimal DST, including intermediate nodes
            for edge in list(opt_DST.edges()):
                s, d = edge[0], edge[1]
                if s in G and d in G[s]:
                    MDST_graph.add_edge(s, d, **G[s][d])

            # Verify MDST graph has edges and paths to all destinations
            if len(MDST_graph.edges()) == 0:
                raise ValueError("MDST produced no edges")

            # Check if all destinations are reachable
            unreachable = []
            for dst in dsts:
                if dst in h_fallback:
                    # Ensure destination is in graph
                    if dst not in MDST_graph:
                        MDST_graph.add_node(dst)
                    # Check if path exists
                    if not nx.has_path(MDST_graph, src, dst):
                        unreachable.append(dst)

            if unreachable:
                print(f"Warning: MDST graph missing paths to: {unreachable}. Adding Dijkstra paths.")
                # Add Dijkstra paths for unreachable destinations
                for dst in unreachable:
                    try:
                        path = nx.dijkstra_path(h_fallback, src, dst, weight="cost")
                        for i in range(len(path) - 1):
                            s, t = path[i], path[i + 1]
                            if s in G and t in G[s]:
                                MDST_graph.add_edge(s, t, **G[s][t])
                    except (nx.NodeNotFound, nx.NetworkXNoPath):
                        print(f"Warning: No path found to {dst}")

            solution_graph = MDST_graph
        except Exception as e:
            print(f"Warning: MDST fallback failed with error: {e}. Falling back to N_dijkstra.")
            # Final fallback: use Dijkstra for each destination
            MDST_graph = nx.DiGraph()
            for dst in dsts:
                if dst in h_fallback:
                    try:
                        path = nx.dijkstra_path(h_fallback, src, dst, weight="cost")
                        for i in range(len(path) - 1):
                            s, t = path[i], path[i + 1]
                            if s in G and t in G[s]:
                                MDST_graph.add_edge(s, t, **G[s][t])
                    except (nx.NodeNotFound, nx.NetworkXNoPath):
                        print(f"Warning: No path found to {dst}")
            solution_graph = MDST_graph

    print(
        f"Number of overlays added: {len(solution_graph.nodes) - (1 + len(dsts))}, {[node for node in solution_graph.nodes if node not in [src]+dsts]}"
    )
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Clean up files
    if os.path.exists(config_loc):
        os.remove(config_loc)
    if os.path.exists(write_loc):
        os.remove(write_loc)
    if os.path.exists(param_loc):
        os.remove(param_loc)

    return append_src_dst_paths(src, dsts, solution_graph, bc_topology)


_baselines_list = [
    ("N_dijkstra", N_dijkstra),
    ("N_direct", N_direct),
    # ("MDST", MDST),
    # ("MULTI_MDST", MULTI_MDST),
    ("Min_Steiner_Tree", Min_Steiner_Tree),
    ("Expert", Cloudcast_OPT),
]

baselines_codes = [(lbl, inspect.getsource(foo)) for lbl, foo in _baselines_list]
# make them all search_algorithm
baselines_codes = [
    (lbl, re.sub(r"^def \w+\(", "def search_algorithm(", code, flags=re.MULTILINE)) for lbl, code in baselines_codes
]


def get_baselines():
    return baselines_codes
