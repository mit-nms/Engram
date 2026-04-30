import os
import sys
import itertools
import importlib

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from utils import *
from simulator import *
from broadcast import *
from evaluator import *


def path_to_edges(path, G):
    """
    Convert a path (list of nodes) to edge format [u, v, G[u][v]].

    Args:
        path: List of nodes [n1, n2, n3, ...]
        G: networkx DiGraph with edge attributes

    Returns:
        List of edges in format [[u, v, edge_data], ...]
    """
    edges = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if G.has_edge(u, v):
            edges.append([u, v, G[u][v]])
        else:
            raise ValueError(f"Edge {u}->{v} does not exist in graph")
    return edges


def brute_force_search_simulator_edge_lists(
    src,
    dsts,
    G,
    num_partitions,
    TIME=60,
    TRANSFER_SIZE=300,
    max_vms_per_node=2,
    fixed_default_vms=None,
    verbose=True,
):
    """
    Brute force search over a *different* solution space: arbitrary per-(dst,partition) edge lists.

    Why this exists:
      - BCSimulator does NOT require that bc_topology.paths[dst][partition] is a single ordered src->dst path.
      - It builds a global graph from the *union* of listed edges, and transfer time is computed only over
        the edges present in each (dst,partition) list.

    This brute force enumerates the smallest non-empty edge-list that satisfies the validator:
      - for each (dst,partition), pick exactly ONE incoming edge (u -> dst).

    This makes the brute-force comparable to what the simulator actually scores, and can outperform the
    "simple-path brute force" above because it is strictly more general.
    """
    H = G.copy()
    H.remove_edges_from(list(H.in_edges(src)) + list(nx.selfloop_edges(H)))

    # VM allocations (uniform across all nodes in the simulator)
    max_vm_limit = max([H.nodes[n].get("vm_limit", max_vms_per_node) for n in H.nodes()])
    max_vm_limit = min(max_vm_limit, max_vms_per_node)
    if fixed_default_vms is not None:
        vm_allocations = [int(fixed_default_vms)]
    else:
        vm_allocations = list(range(1, max_vm_limit + 1))

    # Candidate edge choices per destination: incoming edges only
    incoming_edges = {}
    for dst in dsts:
        preds = list(H.predecessors(dst)) if dst in H.nodes else []
        incoming_edges[dst] = [(u, dst) for u in preds if H.has_edge(u, dst)]
        if verbose:
            print(f"  Incoming-edge choices for {dst}: {len(incoming_edges[dst])}")

    if any(len(incoming_edges[dst]) == 0 for dst in dsts):
        raise ValueError(f"Some destinations have no incoming edges in H: {[d for d in dsts if len(incoming_edges[d]) == 0]}")

    # Total combinations per partition: product over dst of indeg(dst)
    per_partition_choices = 1
    for dst in dsts:
        per_partition_choices *= len(incoming_edges[dst])
    total_path_assignments = per_partition_choices ** int(num_partitions)
    total_solutions = len(vm_allocations) * total_path_assignments

    if verbose:
        print("\nBrute force over simulator edge-lists (1 incoming edge per dst per partition)")
        print(f"  Per-partition combinations: {per_partition_choices:,}")
        print(f"  Total path assignments: {total_path_assignments:,}")
        print(f"  VM allocations tried: {len(vm_allocations)}")
        print(f"  Total solutions to evaluate: {total_solutions:,}")

    # Simulator config
    config = {
        "num_partitions": num_partitions,
        "data_vol": TRANSFER_SIZE,
        "source_node": src,
        "dest_nodes": dsts,
    }

    best_cost = float("inf")
    best_transfer_time = float("inf")
    best_solution = None
    best_vm_alloc = None
    evaluated = 0
    valid_solutions = 0

    # Precompute shortest paths from src to any node in H (for reachability prefix edges).
    # This prevents "cheating" solutions where data appears at u without src->u edges listed.
    try:
        shortest_from_src = nx.single_source_shortest_path(H, src)
    except Exception:
        shortest_from_src = {}

    # Build all per-partition incoming-edge assignments:
    # For a single partition, an assignment is a tuple of one (u,dst) per destination in dsts order.
    per_partition_assignments = list(itertools.product(*[incoming_edges[dst] for dst in dsts]))

    for default_vms in vm_allocations:
        simulator = BCSimulator(2)
        # IMPORTANT: recreate the cartesian product iterator per VM allocation (iterators are one-shot)
        for assignment in itertools.product(per_partition_assignments, repeat=num_partitions):
            evaluated += 1

            bc_topology = BroadCastTopology(src, dsts, num_partitions)
            # assignment[part_id] is a tuple aligned with dsts order: ((u0,dst0),(u1,dst1),...)
            for part_id in range(num_partitions):
                part_edges = assignment[part_id]
                for dst_idx, dst in enumerate(dsts):
                    u, v = part_edges[dst_idx]
                    # Add a src->u prefix path (if needed) so that dst is reachable from src under validation.
                    if u != src:
                        if u not in shortest_from_src:
                            # unreachable prefix; leave invalid (validator will reject)
                            pass
                        else:
                            prefix_nodes = shortest_from_src[u]
                            for i in range(len(prefix_nodes) - 1):
                                a, b = prefix_nodes[i], prefix_nodes[i + 1]
                                if H.has_edge(a, b):
                                    bc_topology.append_dst_partition_path(dst, part_id, [a, b, H[a][b]])
                    if H.has_edge(u, v):
                        bc_topology.append_dst_partition_path(dst, part_id, [u, v, H[u][v]])

            is_valid, errors, warnings, stats, validity_percentage = validate_broadcast_topology(
                bc_topology, H, src, dsts, num_partitions
            )
            if not is_valid:
                continue

            valid_solutions += 1

            # Evaluate using simulator (in-memory object path is supported by simulator.initialization)
            try:
                transfer_time, cost, _ = simulator.evaluate_path(bc_topology, config, write_to_file=False)
            except Exception:
                continue

            if cost < best_cost:
                best_cost = cost
                best_transfer_time = transfer_time
                best_solution = bc_topology
                best_vm_alloc = {"default_vms_per_region": default_vms}

    if best_solution is None:
        raise ValueError("No valid solutions found in simulator-edge-list brute force")

    stats = {"total_solutions": total_solutions, "valid_solutions": valid_solutions, "evaluated": evaluated}
    return best_solution, best_cost, best_transfer_time, best_vm_alloc, stats


def test_simple_5node_graph(code_file_name):
    """
    Simple test function that creates a minimal directed graph and tests
    the search_algorithm function from {code_file_name} to verify it works correctly.

    Returns:
        bool: True if test passes, False otherwise
    """
    print("=" * 70)
    print(f"Testing 5-node Graph with search_algorithm from {code_file_name}")
    print("=" * 70)

    # Create 5-node directed graph
    G = nx.DiGraph()

    # Add nodes with provider format
    nodes = {
        "A": "aws:us-east-1",  # source
        "B": "aws:us-west-2",
        "C": "gcp:us-central-1",  # destination
        "D": "gcp:us-west-1",
        "E": "azure:eastus",  # destination
        "F": "azure:eastus2",  # destination
    }

    for short_name, full_name in nodes.items():
        G.add_node(full_name)

    # Add edges with cost and throughput attributes
    # Structure: A -> B -> C, A -> D -> E, B -> E, C -> E
    edges = [
        ("A", "B", {"cost": 0.02, "throughput": 5.0}),  # aws:us-east-1 -> aws:us-west-2
        ("B", "C", {"cost": 0.03, "throughput": 4.0}),  # aws:us-west-2 -> gcp:us-central-1
        ("A", "D", {"cost": 0.025, "throughput": 6.0}),  # aws:us-east-1 -> gcp:us-west-1
        ("D", "E", {"cost": 0.03, "throughput": 5.0}),  # gcp:us-west-1 -> azure:eastus
        ("B", "E", {"cost": 0.035, "throughput": 3.5}),  # aws:us-west-2 -> azure:eastus
        ("C", "E", {"cost": 0.04, "throughput": 4.5}),  # gcp:us-central-1 -> azure:eastus
        # ("C", "F", {"cost": 0.05, "throughput": 5.0}),  # gcp:us-central-1 -> aws:us-east-2
        # ("D", "F", {"cost": 0.06, "throughput": 6.0}),  # gcp:us-west-1 -> aws:us-east-2
        # ("F", "E", {"cost": 0.07, "throughput": 7.0}),  # aws:us-east-2 -> azure:eastus
    ]

    for src_short, dst_short, attrs in edges:
        src = nodes[src_short]
        dst = nodes[dst_short]
        G.add_edge(src, dst, **attrs)

    # Set node attributes (defaults as in optimal.py)
    providers = ["aws", "gcp", "azure"]
    provider_ingress = [10, 16, 16]
    provider_egress = [5, 7, 16]
    ingress_limits = {providers[i]: provider_ingress[i] for i in range(len(providers))}
    egress_limits = {providers[i]: provider_egress[i] for i in range(len(providers))}

    for v in G.nodes():
        provider = str(v).split(":")[0]
        G.nodes[v]["egress_limit"] = egress_limits.get(provider, 10)
        G.nodes[v]["ingress_limit"] = ingress_limits.get(provider, 10)
        G.nodes[v]["vm_cost"] = 0.00015  # Cost per second
        G.nodes[v]["vm_limit"] = 2  # Max 2 VMs per region

    # Test configuration
    src = nodes["A"]  # aws:us-east-1
    dsts = [nodes["C"], nodes["E"]]  # [gcp:us-central-1, azure:eastus]
    num_partitions = 2
    TIME = 60
    TRANSFER_SIZE = 300

    print(f"\nGraph structure:")
    print(f"  Nodes: {list(G.nodes())}")
    print(f"  Edges: {list(G.edges())}")
    print(f"\nTest configuration:")
    print(f"  Source: {src}")
    print(f"  Destinations: {dsts}")
    print(f"  Partitions: {num_partitions}")
    print(f"  TIME: {TIME}s")
    print(f"  TRANSFER_SIZE: {TRANSFER_SIZE} GB")

    # Run search_algorithm from {code_file_name}
    try:
        print(f"\n{'=' * 70}")
        print(f"Running search_algorithm from {code_file_name}...")
        print(f"{'=' * 70}")
        bc_topology = search_algorithm(src, dsts, G, num_partitions, TIME=TIME, TRANSFER_SIZE=TRANSFER_SIZE)
        print("✓ search_algorithm completed successfully")
    except Exception as e:
        print(f"✗ search_algorithm from {code_file_name} failed: {e}")
        traceback.print_exc()
        return False

    # Validate the solution
    print(f"\n{'=' * 70}")
    print("Validating solution...")
    print(f"{'=' * 70}")

    is_valid, errors, warnings, stats, validity_percentage = validate_broadcast_topology(
        bc_topology, G, src, dsts, num_partitions
    )

    print(f"\nValidation Results:")
    print(f"  Total partitions: {stats['total_partitions']}")
    print(f"  Valid partitions: {stats['valid_partitions']}")
    print(f"  Invalid partitions: {stats['invalid_partitions']}")
    print(f"  Validity: {validity_percentage:.1f}%")

    if stats["self_loop_partitions"] > 0:
        print(f"  ⚠️  Self-loop partitions: {stats['self_loop_partitions']}")
    if stats["empty_path_partitions"] > 0:
        print(f"  ⚠️  Empty path partitions: {stats['empty_path_partitions']}")
    if stats["missing_destination_partitions"] > 0:
        print(f"  ⚠️  Partitions not reaching destination: {stats['missing_destination_partitions']}")
    if stats["invalid_edge_partitions"] > 0:
        print(f"  ⚠️  Partitions with invalid edges: {stats['invalid_edge_partitions']}")

    if warnings:
        print(f"\n  Warnings:")
        for warning in warnings[:5]:  # Show first 5 warnings
            print(f"    - {warning}")
        if len(warnings) > 5:
            print(f"    ... and {len(warnings) - 5} more warnings")

    if errors:
        print(f"\n  Errors:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"    - {error}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more errors")

    # Final result
    print(f"\n{'=' * 70}")
    if is_valid and stats["invalid_partitions"] == 0:
        print("✅ TEST PASSED: All partitions have valid paths")
        print(f"{'=' * 70}\n")
        return True
    else:
        print("❌ TEST FAILED: Solution has invalid partitions")
        print(f"{'=' * 70}\n")
        return False


def compare_code_file_name_vs_bruteforce(code_file_name):
    """
    Compare {code_file_name}'s search_algorithm with brute force search.
    Creates a simple 5-node graph and evaluates both methods.

    Returns:
        bool: True if comparison completed successfully
    """
    print("=" * 70)
    print(f"Comparing {code_file_name} vs Brute Force Search")
    print("=" * 70)

    # Create the same 5-node graph as in test_simple_5node_graph
    G = nx.DiGraph()

    # Add nodes with provider format
    nodes = {
        "A": "aws:us-east-1",  # source
        "B": "aws:us-west-2",
        "C": "gcp:us-central-1",  # destination
        "D": "gcp:us-west-1",
        "E": "azure:eastus",  # destination
        "F": "azure:eastus2",  # destination
    }

    for short_name, full_name in nodes.items():
        G.add_node(full_name)

    # Add edges with cost and throughput attributes
    edges = [
        ("A", "B", {"cost": 0.02, "throughput": 5.0}),
        ("B", "C", {"cost": 0.03, "throughput": 4.0}),
        ("A", "D", {"cost": 0.025, "throughput": 6.0}),
        ("D", "E", {"cost": 0.03, "throughput": 5.0}),
        ("B", "E", {"cost": 0.035, "throughput": 3.5}),
        ("C", "E", {"cost": 0.04, "throughput": 4.5}),
        ("F", "E", {"cost": 0.05, "throughput": 5.0}),
        ("C", "F", {"cost": 0.06, "throughput": 6.0}),
        ("D", "F", {"cost": 0.07, "throughput": 7.0}),
    ]

    for src_short, dst_short, attrs in edges:
        src = nodes[src_short]
        dst = nodes[dst_short]
        G.add_edge(src, dst, **attrs)

    # Set node attributes
    providers = ["aws", "gcp", "azure"]
    provider_ingress = [10, 16, 16]
    provider_egress = [5, 7, 16]
    ingress_limits = {providers[i]: provider_ingress[i] for i in range(len(providers))}
    egress_limits = {providers[i]: provider_egress[i] for i in range(len(providers))}

    for v in G.nodes():
        provider = str(v).split(":")[0]
        G.nodes[v]["egress_limit"] = egress_limits.get(provider, 10)
        G.nodes[v]["ingress_limit"] = ingress_limits.get(provider, 10)
        G.nodes[v]["vm_cost"] = 0.00015
        G.nodes[v]["vm_limit"] = 2

    # Test configuration
    src = nodes["A"]
    dsts = [nodes["C"], nodes["E"], nodes["F"]]
    num_partitions = 2
    TIME = 60
    TRANSFER_SIZE = 300

    print(f"\nTest Configuration:")
    print(f"  Source: {src}")
    print(f"  Destinations: {dsts}")
    print(f"  Partitions: {num_partitions}")
    print(f"  TIME: {TIME}s")
    print(f"  TRANSFER_SIZE: {TRANSFER_SIZE} GB")

    # Create config for simulator
    config = {
        "num_partitions": num_partitions,
        "data_vol": TRANSFER_SIZE,
        "source_node": src,
        "dest_nodes": dsts,
    }

    print(f"\n{'=' * 70}")
    print(f"Step 1: Running {code_file_name}...")
    print(f"{'=' * 70}")

    # Run {code_file_name}
    try:
        optimal_start = time.time()
        optimal_solution = search_algorithm(src, dsts, G, num_partitions, TIME=TIME, TRANSFER_SIZE=TRANSFER_SIZE)
        optimal_time = time.time() - optimal_start
        print(f"✓ {code_file_name} completed in {optimal_time:.4f}s")
    except Exception as e:
        print(f"✗ {code_file_name} failed: {e}")
        traceback.print_exc()
        return False

    # Evaluate {code_file_name} solution using simulator
    print(f"\nEvaluating {code_file_name} solution...")
    try:
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            setting = {
                "algo": code_file_name,
                "source_node": src,
                "terminal_nodes": dsts,
                "num_partitions": num_partitions,
                "generated_path": optimal_solution.paths,
            }
            json.dump(setting, f)
            temp_path = f.name

        # Use default VMs (2) for evaluation - same as brute force
        # Keep evaluation consistent with brute-force by using a fixed VM count here.
        fixed_num_vms = 2
        simulator_result = BCSimulator(int(fixed_num_vms))
        optimal_transfer_time, optimal_cost, optimal_output = simulator_result.evaluate_path(
            temp_path, config, write_to_file=False
        )
        os.unlink(temp_path)

        print(f"  {code_file_name} cost: ${optimal_cost:.6f}")
        print(f"  {code_file_name} transfer time: {optimal_transfer_time:.4f}s")
    except Exception as e:
        print(f"✗ Failed to evaluate {code_file_name} solution: {e}")
        traceback.print_exc()
        return False

    print(f"\n{'=' * 70}")
    print(f"Step 2: Running brute force search (simulator scoring) from {code_file_name}...")
    print(f"{'=' * 70}")

    # Run simulator-edge-list brute force (brute force for what BCSimulator scores)
    try:
        bf_start = time.time()
        bf_solution, bf_cost, bf_transfer_time, bf_vm_alloc, bf_stats = brute_force_search_simulator_edge_lists(
            src, dsts, G, num_partitions, TIME=TIME, TRANSFER_SIZE=TRANSFER_SIZE, max_vms_per_node=2,
            fixed_default_vms=fixed_num_vms, verbose=True
        )
        bf_time = time.time() - bf_start
        print(f"\n✓ Brute force (simulator scoring) completed in {bf_time:.4f}s")
    except Exception as e:
        print(f"✗ Brute force (simulator scoring) failed: {e}")
        traceback.print_exc()
        return False

    # Calculate combined_score (cost_score as per evaluator)
    # combined_score = 1.0 / (1.0 + cost)  # Lower cost = higher score
    optimal_combined_score = 1.0 / (1.0 + optimal_cost)
    bf_combined_score = 1.0 / (1.0 + bf_cost)

    # Print comparison
    print(f"\n{'=' * 70}")
    print("COMPARISON RESULTS")
    print(f"{'=' * 70}")
    print(f"\n{code_file_name} (MILP):")
    print(f"  Cost: ${optimal_cost:.6f}")
    print(f"  Transfer time: {optimal_transfer_time:.4f}s")
    print(f"  Combined score: {optimal_combined_score:.6f}")
    print(f"  Search time: {optimal_time:.4f}s")

    print(f"\nBrute Force (simulator scoring):")
    print(f"  Cost: ${bf_cost:.6f}")
    print(f"  Transfer time: {bf_transfer_time:.4f}s")
    print(f"  Combined score: {bf_combined_score:.6f}")
    print(f"  Search time: {bf_time:.4f}s")
    print(f"  Solutions evaluated: {bf_stats['evaluated']:,}")
    print(f"  Valid solutions: {bf_stats['valid_solutions']:,}")
    print(f"  Best VM allocation: {bf_vm_alloc}")

    print(f"\n{'=' * 70}")
    print("DIFFERENCE")
    print(f"{'=' * 70}")
    cost_diff = optimal_cost - bf_cost
    cost_diff_pct = (cost_diff / bf_cost * 100) if bf_cost > 0 else 0
    score_diff = optimal_combined_score - bf_combined_score
    print(f"\n{code_file_name} vs Brute Force (simulator scoring):")
    print(f"  Cost difference ({code_file_name} - BF): ${cost_diff:.6f} ({cost_diff_pct:.2f}%)")
    print(f"  Score difference ({code_file_name} - BF): {score_diff:.6f}")

    best_name, best_cost_val = min(
        [(f"{code_file_name} (MILP)", optimal_cost), ("Brute Force (simulator scoring)", bf_cost)],
        key=lambda x: x[1],
    )
    print(f"\nBest by cost: {best_name} (${best_cost_val:.6f})")

    # Validation check
    print(f"\n{'=' * 70}")
    print("VALIDATION")
    print(f"{'=' * 70}")

    optimal_is_valid, _, _, optimal_stats, optimal_validity = validate_broadcast_topology(
        optimal_solution, G, src, dsts, num_partitions
    )

    bf_is_valid, _, _, bf_validation_stats, bf_validity = validate_broadcast_topology(
        bf_solution, G, src, dsts, num_partitions
    )

    print(f"\n{code_file_name} validation:")
    print(f"  Valid: {optimal_is_valid}")
    print(f"  Validity: {optimal_validity:.1f}%")
    print(f"  Valid partitions: {optimal_stats['valid_partitions']}/{optimal_stats['total_partitions']}")

    print(f"\nBrute Force (simulator scoring) validation:")
    print(f"  Valid: {bf_is_valid}")
    print(f"  Validity: {bf_validity:.1f}%")
    print(f"  Valid partitions: {bf_validation_stats['valid_partitions']}/{bf_validation_stats['total_partitions']}")

    print(f"\n{'=' * 70}\n")

    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python verify_brute_force.py <code_file_name> [compare]")
        exit(1)
    code_file_name = sys.argv[1]
    try:
        # Dynamically import the module using importlib
        module = importlib.import_module(code_file_name)
        search_algorithm = module.search_algorithm
    except ImportError as e:
        print(f"Error: Could not import module '{code_file_name}': {e}")
        exit(1)
    # Check command line arguments
    if len(sys.argv) > 2 and sys.argv[2] == "compare":
        # Run comparison
        success = compare_code_file_name_vs_bruteforce(code_file_name)
        exit(0 if success else 1)
    else:
        # Run the 5-node graph test
        success = test_simple_5node_graph(code_file_name)
        exit(0 if success else 1)
