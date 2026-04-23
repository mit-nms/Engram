# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
from typing import Dict, List
import numpy as np
import pandas as pd
from pulp import LpVariable, LpProblem, LpMinimize, lpSum, LpBinary, LpInteger, LpContinuous, PulpSolverError
from broadcast import BroadCastTopology


def search_algorithm(src, dsts, G, num_partitions, TIME=60, TRANSFER_SIZE=300):
    STRIPES = num_partitions
    stripe_size = TRANSFER_SIZE / STRIPES

    # Defaults (same as your code)
    providers = ["aws", "gcp", "azure"]
    provider_ingress = [10, 16, 16]
    provider_egress = [5, 7, 16]
    ingress_limits = {providers[i]: provider_ingress[i] for i in range(len(providers))}
    egress_limits = {providers[i]: provider_egress[i] for i in range(len(providers))}

    for n in G.nodes:
        provider = str(n).split(":")[0]
        G.nodes[n]["egress_limit"] = egress_limits[provider]
        G.nodes[n]["ingress_limit"] = ingress_limits[provider]
        G.nodes[n]["vm_cost"] = 0.00015
        G.nodes[n]["vm_limit"] = 2

    bc_topology = BroadCastTopology(src, dsts, STRIPES)

    if src not in G.nodes:
        return bc_topology
    dsts = [d for d in dsts if d in G.nodes]
    if not dsts:
        return bc_topology

    try:
        prob = LpProblem("Cloudcast_AllStripes_MulticastTree", LpMinimize)

        # Shared VMs across all stripes
        N = {
            v: LpVariable(f"N_{v}", 0, G.nodes[v]["vm_limit"], LpInteger)
            for v in G.nodes
        }

        # Edge selected in tree for stripe s
        X = {
            (s, u, v): LpVariable(f"X_{s}_{u}_{v}", 0, 1, LpBinary)
            for s in range(STRIPES)
            for (u, v) in G.edges()
        }

        # Commodity flow to ensure reachability: one unit from src to each dst for each stripe
        # Continuous is fine because X is binary and we only need existence of a route.
        F = {
            (s, d, u, v): LpVariable(f"F_{s}_{d}_{u}_{v}", 0, 1, LpContinuous)
            for s in range(STRIPES)
            for d in dsts
            for (u, v) in G.edges()
        }

        # Objective: instance cost + egress cost of sending each stripe along selected edges once
        inst_cost = lpSum(G.nodes[v]["vm_cost"] * N[v] * TIME for v in G.nodes)
        egress_cost = lpSum(
            G[u][v]["cost"] * stripe_size * X[(s, u, v)]
            for s in range(STRIPES)
            for (u, v) in G.edges()
        )
        prob += inst_cost + egress_cost

        # Flow can only go on selected edges
        for s in range(STRIPES):
            for d in dsts:
                for (u, v) in G.edges():
                    prob += F[(s, d, u, v)] <= X[(s, u, v)], f"flow_on_tree_s{s}_d{d}_{u}_{v}"

        nodes = list(G.nodes())

        # Flow conservation for each (s,d): send 1 from src to d
        for s in range(STRIPES):
            for d in dsts:
                for v in nodes:
                    inflow = lpSum(F[(s, d, u, v)] for (u, _) in G.in_edges(v))
                    outflow = lpSum(F[(s, d, v, w)] for (_, w) in G.out_edges(v))

                    if v == src:
                        # net outflow = 1
                        prob += outflow - inflow == 1, f"flow_src_s{s}_d{d}"
                    elif v == d:
                        # net inflow = 1
                        prob += inflow - outflow == 1, f"flow_dst_s{s}_d{d}"
                    else:
                        prob += inflow - outflow == 0, f"flow_cons_s{s}_d{d}_v{v}"

        # ===== Shared capacity constraints across all stripes (tree edges counted once) =====
        for (u, v) in G.edges():
            bw = G[u][v]["throughput"]
            prob += (
                lpSum(stripe_size * X[(s, u, v)] for s in range(STRIPES))
                <= bw * N[u] * TIME
            ), f"link_cap_{u}_{v}"

        for v in G.nodes():
            egress_lim = G.nodes[v]["egress_limit"] * N[v] * TIME
            ingress_lim = G.nodes[v]["ingress_limit"] * N[v] * TIME

            prob += (
                lpSum(
                    stripe_size * X[(s, v, w)]
                    for s in range(STRIPES)
                    for (_, w) in G.out_edges(v)
                )
                <= egress_lim
            ), f"node_egress_{v}"

            prob += (
                lpSum(
                    stripe_size * X[(s, u, v)]
                    for s in range(STRIPES)
                    for (u, _) in G.in_edges(v)
                )
                <= ingress_lim
            ), f"node_ingress_{v}"

        status = prob.solve()
        if status != 1:
            print(f"Solver status: {status} (1=Optimal)")
            return bc_topology

        # ===== Build per-dst paths per stripe from flow edges =====
        # For each (s,d), we can trace a path by following edges with F ~ 1 out of src.
        for s in range(STRIPES):
            for d in dsts:
                # adjacency from flow
                next_hop = {}
                for (u, v) in G.edges():
                    val = F[(s, d, u, v)].value()
                    if val is not None and val > 0.5:
                        next_hop[u] = v  # should work as a simple path in typical solutions

                cur = src
                seen = set([src])
                path_nodes = [src]
                ok = True
                while cur != d:
                    nxt = next_hop.get(cur, None)
                    if nxt is None or nxt in seen:
                        ok = False
                        break
                    seen.add(nxt)
                    path_nodes.append(nxt)
                    cur = nxt

                if not ok:
                    continue

                for i in range(len(path_nodes) - 1):
                    u, v = path_nodes[i], path_nodes[i + 1]
                    if not G.has_edge(u, v):
                        ok = False
                        break
                    edge = G[u][v]
                    if "cost" not in edge or "throughput" not in edge:
                        ok = False
                        break
                    edge_copy = {
                        "cost": float(edge["cost"]) if edge["cost"] is not None else float("inf"),
                        "throughput": float(edge["throughput"]) if edge["throughput"] is not None else 0.0,
                    }
                    bc_topology.append_dst_partition_path(d, s, [str(u), str(v), edge_copy])

    except PulpSolverError as e:
        print(f"Solver failed: {e}")

    return bc_topology
