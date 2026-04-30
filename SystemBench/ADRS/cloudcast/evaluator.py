import hashlib
import importlib.util
import numpy as np
import time
import concurrent.futures
import traceback
import signal
import json
import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
import numpy as np
import pulp
from pulp import LpVariable, LpProblem, LpMinimize, lpSum, LpBinary, LpInteger, PulpSolverError
from sklearn.cluster import KMeans
from colorama import Fore, Style

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from utils import *
from simulator import *
from broadcast import *


def validate_broadcast_topology(bc_t, G, source_node, terminal_nodes, num_partitions):
    """
    Validate that the broadcast topology is correct:
    - All partitions have valid paths (not None, not empty)
    - Paths are not just self-loops
    - Paths actually reach the destination
    - All edges in paths exist in the graph
    - Edge attributes (cost, throughput, etc.) match the original graph

    Returns: (is_valid, error_messages, validation_stats)
    """
    errors = []
    warnings = []
    stats = {
        'total_partitions': 0,
        'valid_partitions': 0,
        'invalid_partitions': 0,
        'self_loop_partitions': 0,
        'empty_path_partitions': 0,
        'missing_destination_partitions': 0,
        'invalid_edge_partitions': 0,
        'disconnected_path_partitions': 0,
        'missing_source_partitions': 0,
        'modified_cost_edges': 0,
        'modified_throughput_edges': 0,
        'modified_attribute_edges': 0
    }

    # Check all destinations exist
    for dst in terminal_nodes:
        if dst not in bc_t.paths:
            errors.append(f"Destination {dst} not found in paths dictionary")
            continue

        # Check all partitions for this destination
        for partition_id in range(num_partitions):
            partition_str = str(partition_id)
            stats['total_partitions'] += 1

            if partition_str not in bc_t.paths[dst]:
                errors.append(f"Partition {partition_id} missing for destination {dst}")
                stats['invalid_partitions'] += 1
                continue

            path_list = bc_t.paths[dst][partition_str]

            # Check if path is None or empty
            if path_list is None:
                errors.append(f"Partition {partition_id} for destination {dst} is None")
                stats['invalid_partitions'] += 1
                stats['empty_path_partitions'] += 1
                continue

            if len(path_list) == 0:
                errors.append(f"Partition {partition_id} for destination {dst} has empty path")
                stats['invalid_partitions'] += 1
                stats['empty_path_partitions'] += 1
                continue

            # Check if path is just self-loops
            is_self_loop_only = True
            path_reaches_dst = False
            path_starts_from_src = False
            path_is_connected = True  # Track if edges form a connected path
            has_invalid_edge = False
            has_modified_attributes = False
            
            # Track the "current" node as we traverse the path
            # Path should start from source and each edge should connect to the next
            prev_edge_dst = source_node  # Path should start from source
            
            for idx, edge in enumerate(path_list):
                if len(edge) < 2:
                    errors.append(f"Partition {partition_id} for destination {dst} has malformed edge: {edge}")
                    has_invalid_edge = True
                    break

                edge_src, edge_dst = edge[0], edge[1]

                # Check if it's a self-loop
                if edge_src != edge_dst:
                    is_self_loop_only = False

                # Check if path reaches destination
                if edge_dst == dst:
                    path_reaches_dst = True
                
                # Check if first edge starts from source
                if idx == 0:
                    if edge_src == source_node:
                        path_starts_from_src = True
                
                # Check path connectivity: current edge should start where previous edge ended
                if edge_src != prev_edge_dst:
                    path_is_connected = False
                    errors.append(
                        f"Partition {partition_id} for destination {dst} has disconnected path: "
                        f"edge {idx} starts at {edge_src} but previous edge ended at {prev_edge_dst}"
                    )
                
                # Update previous edge destination for next iteration
                prev_edge_dst = edge_dst

                # Check if edge exists in graph
                if edge_src not in G.nodes or edge_dst not in G.nodes:
                    errors.append(f"Partition {partition_id} for destination {dst} uses node not in graph: {edge_src} -> {edge_dst}")
                    has_invalid_edge = True
                    break

                # Determine the correct edge direction in the graph
                graph_edge_src = edge_src
                graph_edge_dst = edge_dst
                if not G.has_edge(edge_src, edge_dst):
                    if G.has_edge(edge_dst, edge_src):
                        # Edge exists in reverse direction
                        graph_edge_src = edge_dst
                        graph_edge_dst = edge_src
                    else:
                        errors.append(f"Partition {partition_id} for destination {dst} uses edge not in graph: {edge_src} -> {edge_dst}")
                        has_invalid_edge = True
                        break

                # Validate edge attributes match the original graph
                # Edge format should be [src, dst, data_dict]
                if len(edge) >= 3:
                    edge_data = edge[2]
                    if isinstance(edge_data, dict):
                        # Get original edge data from graph
                        original_edge_data = G[graph_edge_src][graph_edge_dst]
                        
                        # Check cost attribute
                        if 'cost' in edge_data:
                            original_cost = original_edge_data.get('cost', None)
                            path_cost = edge_data.get('cost', None)
                            if original_cost is not None and path_cost is not None:
                                # Use small epsilon for floating point comparison
                                if abs(original_cost - path_cost) > 1e-9:
                                    errors.append(
                                        f"Partition {partition_id} for destination {dst} modifies edge cost: "
                                        f"edge {edge_src}->{edge_dst} has cost {path_cost} but original graph has {original_cost}"
                                    )
                                    stats['modified_cost_edges'] += 1
                                    has_modified_attributes = True
                        
                        # Check throughput attribute
                        if 'throughput' in edge_data:
                            original_throughput = original_edge_data.get('throughput', None)
                            path_throughput = edge_data.get('throughput', None)
                            if original_throughput is not None and path_throughput is not None:
                                if abs(original_throughput - path_throughput) > 1e-9:
                                    errors.append(
                                        f"Partition {partition_id} for destination {dst} modifies edge throughput: "
                                        f"edge {edge_src}->{edge_dst} has throughput {path_throughput} but original graph has {original_throughput}"
                                    )
                                    stats['modified_throughput_edges'] += 1
                                    has_modified_attributes = True
                        
                        # Check all other attributes match
                        for attr_key in edge_data:
                            if attr_key not in ['cost', 'throughput']:  # Already checked above
                                original_value = original_edge_data.get(attr_key, None)
                                path_value = edge_data.get(attr_key, None)
                                if original_value != path_value:
                                    errors.append(
                                        f"Partition {partition_id} for destination {dst} modifies edge attribute '{attr_key}': "
                                        f"edge {edge_src}->{edge_dst} has {attr_key}={path_value} but original graph has {attr_key}={original_value}"
                                    )
                                    stats['modified_attribute_edges'] += 1
                                    has_modified_attributes = True

            if has_invalid_edge:
                stats['invalid_partitions'] += 1
                stats['invalid_edge_partitions'] += 1
                continue

            if has_modified_attributes:
                stats['invalid_partitions'] += 1
                continue

            # Check for self-loops only
            if is_self_loop_only:
                warnings.append(f"Partition {partition_id} for destination {dst} uses only self-loops (does not deliver data)")
                stats['invalid_partitions'] += 1
                stats['self_loop_partitions'] += 1
                continue

            # Check if path reaches destination
            if not path_reaches_dst:
                errors.append(f"Partition {partition_id} for destination {dst} does not reach destination")
                stats['invalid_partitions'] += 1
                stats['missing_destination_partitions'] += 1
                continue

            # Check if path starts from source
            if not path_starts_from_src:
                errors.append(f"Partition {partition_id} for destination {dst} does not start from source {source_node}")
                stats['invalid_partitions'] += 1
                stats['missing_source_partitions'] += 1
                continue

            # Check if path is connected (each edge connects to the next)
            if not path_is_connected:
                stats['invalid_partitions'] += 1
                stats['disconnected_path_partitions'] += 1
                continue

            # Path is valid
            stats['valid_partitions'] += 1

    # Calculate validity percentage
    if stats['total_partitions'] > 0:
        validity_percentage = (stats['valid_partitions'] / stats['total_partitions']) * 100
    else:
        validity_percentage = 0

    is_valid = len(errors) == 0 and stats['invalid_partitions'] == 0

    return is_valid, errors, warnings, stats, validity_percentage


def evaluate(program_path, small_test=False):
    """
    Evaluate the evolved broadcast optimization program across multiple configurations.
    
    Args:
        program_path: Path to the evolved program file
        
    Returns:
        Dictionary with evaluation metrics including required 'combined_score'
    """
    try:
        # Load the evolved program
        spec = importlib.util.spec_from_file_location("program", program_path)
        # Add missib
        program = importlib.util.module_from_spec(spec)

        # Inject commonly used imports into the program namespace BEFORE execution
        # Typing imports
        program.Dict = Dict
        program.List = List
        program.Tuple = Tuple
        program.Optional = Optional
        program.Any = Any
        # Standard library imports
        program.os = os
        program.sys = sys
        program.json = json
        # Third-party imports
        program.nx = nx  # networkx
        program.subprocess = subprocess
        program.np = np
        # import these
        # from pulp import LpVariable, LpProblem, LpMinimize, lpSum, LpBinary, LpInteger, PulpSolverError
        # from sklearn.cluster import KMeans
        program.pulp = pulp
        program.LpVariable = LpVariable
        program.LpProblem = LpProblem
        program.LpMinimize = LpMinimize
        program.lpSum = lpSum
        program.LpBinary = LpBinary
        program.LpInteger = LpInteger
        program.PulpSolverError = PulpSolverError
        program.KMeans = KMeans

        # Inject commonly used classes and functions from star imports
        # These are available in the evaluator namespace via: from utils import *, from broadcast import *
        if 'BroadCastTopology' in globals():
            program.BroadCastTopology = BroadCastTopology
        if 'make_nx_graph' in globals():
            program.make_nx_graph = make_nx_graph
        if 'append_src_dst_paths' in globals():
            program.append_src_dst_paths = append_src_dst_paths

        # Now execute the module
        spec.loader.exec_module(program)
        random_hash = hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]

        # Check if the required function exists
        if not hasattr(program, "search_algorithm"):
            return {
                "combined_score": 0.0,
                "runs_successfully": 0.0,
                "error": "Missing search_algorithm function"
            }
        # print(f"Program path: {program_path}")
        # print(f"search_algorithm: {program.search_algorithm}")
        
        # # Print the function source code
        # import inspect
        # try:
        #     source_code = inspect.getsource(program.search_algorithm)
        #     print("Function source code:")
        #     print("=" * 50)
        #     print(source_code)
        #     print("=" * 50)
        # except Exception as e:
        #     print(f"Could not get source code: {e}")
        # Get the directory of this evaluator file to resolve relative paths
        evaluator_dir = os.path.dirname(os.path.abspath(__file__))
        simulator_output = os.path.join(evaluator_dir, "simulator_output", random_hash)
        os.makedirs(simulator_output, exist_ok=True)
        # print(f"Simulator output: {simulator_output}")
        # Configuration - individual JSON file paths (using absolute paths)
        config_files = [
            os.path.join(evaluator_dir, "examples/config/intra_aws.json"),
            os.path.join(evaluator_dir, "examples/config/intra_azure.json"), 
            os.path.join(evaluator_dir, "examples/config/intra_gcp.json"),
            os.path.join(evaluator_dir, "examples/config/inter_agz.json"),
            os.path.join(evaluator_dir, "examples/config/inter_gaz2.json"),
            # os.path.join(evaluator_dir, "examples/config/inter_hard.json"),
            # os.path.join(evaluator_dir, "examples/config/inter_extreme.json"),
            # os.path.join(evaluator_dir, "examples/config/inter_bottleneck.json")
        ]

        if small_test:
            config_files = [
                os.path.join(evaluator_dir, "examples/config/intra_aws.json"),
                os.path.join(evaluator_dir, "examples/config/intra_azure.json")
            ]
        
        # Filter to only include files that exist
        existing_configs = [f for f in config_files if os.path.exists(f)]
        
        if not existing_configs:
            return {
                "combined_score": 0.0,
                "runs_successfully": 0.0,
                "error": f"No configuration files found. Checked: {config_files}"
            }
        
        num_vms = 2
        total_cost = 0.0
        total_transfer_time = 0.0
        total_search_time = 0.0
        successful_configs = 0
        failed_configs = 0
        last_error_message = ""
        aggregated_output = []
        total_validity_percentage = 0.0  # Track average validity across configs
        # Process each configuration file
        for jsonfile in existing_configs:
            try:
                # print(f"Processing config: {os.path.basename(jsonfile)}")
                
                # Load configuration
                with open(jsonfile, "r") as f:
                    config_name = os.path.basename(jsonfile).split(".")[0]
                    config = json.loads(f.read())

                config_output = {"config_name": config_name, **config}

                # Create graph
                G = make_nx_graph(num_vms=int(num_vms), small_test=small_test)
                    

                # Source and destination nodes
                source_node = config["source_node"]
                terminal_nodes = config["dest_nodes"]

                # Create output directory
                directory = f"{simulator_output}/paths/{config_name}"
                if not os.path.exists(directory):
                    Path(directory).mkdir(parents=True, exist_ok=True)

                # Run the evolved algorithm
                num_partitions = config["num_partitions"]
                start_time = time.time()
                bc_t = program.search_algorithm(source_node, terminal_nodes, G, num_partitions)
                end_time = time.time()
                search_time = end_time - start_time
                # print(f"Time taken: {search_time:.2f} seconds")
                bc_t.set_num_partitions(config["num_partitions"])
                
                # Validate the solution
                # print(f"\n{'='*70}")
                # print(f"Validating solution...")
                # print(f"{'='*70}")
                is_valid, errors, warnings, stats, validity_percentage = validate_broadcast_topology(
                    bc_t, G, source_node, terminal_nodes, num_partitions
                )

                print(f"Validation Results:")
                print(f"  Total partitions: {stats['total_partitions']}")
                print(f"  Valid partitions: {stats['valid_partitions']}")
                print(f"  Invalid partitions: {stats['invalid_partitions']}")
                print(f"  Validity: {validity_percentage:.1f}%")

                if stats['self_loop_partitions'] > 0:
                    print(f"  {Fore.YELLOW}⚠️  Self-loop partitions: {stats['self_loop_partitions']} (these don't deliver data!){Style.RESET_ALL}")
                if stats['empty_path_partitions'] > 0:
                    print(f"  {Fore.YELLOW}⚠️  Empty path partitions: {stats['empty_path_partitions']}{Style.RESET_ALL}")
                if stats['missing_destination_partitions'] > 0:
                    print(f"  {Fore.YELLOW}⚠️  Partitions not reaching destination: {stats['missing_destination_partitions']}{Style.RESET_ALL}")
                if stats['invalid_edge_partitions'] > 0:
                    print(f"  {Fore.YELLOW}⚠️  Partitions with invalid edges: {stats['invalid_edge_partitions']}{Style.RESET_ALL}")
                if stats['disconnected_path_partitions'] > 0:
                    print(f"  {Fore.RED}❌ Partitions with disconnected paths: {stats['disconnected_path_partitions']} (BROKEN PATHS!){Style.RESET_ALL}")
                if stats['missing_source_partitions'] > 0:
                    print(f"  {Fore.RED}❌ Partitions not starting from source: {stats['missing_source_partitions']}{Style.RESET_ALL}")
                if stats['modified_cost_edges'] > 0:
                    print(f"  {Fore.RED}❌ Edges with modified cost: {stats['modified_cost_edges']} (CHEATING DETECTED!){Style.RESET_ALL}")
                if stats['modified_throughput_edges'] > 0:
                    print(f"  {Fore.RED}❌ Edges with modified throughput: {stats['modified_throughput_edges']} (CHEATING DETECTED!){Style.RESET_ALL}")
                if stats['modified_attribute_edges'] > 0:
                    print(f"  {Fore.RED}❌ Edges with modified attributes: {stats['modified_attribute_edges']} (CHEATING DETECTED!){Style.RESET_ALL}")

                if warnings:
                    print(f"\n  {Fore.YELLOW}Warnings:{Style.RESET_ALL}")
                    for warning in warnings[:5]:  # Show first 5 warnings
                        print(f"    - {warning}")
                    if len(warnings) > 5:
                        print(f"    ... and {len(warnings) - 5} more warnings")

                if errors:
                    print(f"\n  {Fore.RED}Errors:{Style.RESET_ALL}")
                    for error in errors[:5]:  # Show first 5 errors
                        print(f"    - {error}")
                    if len(errors) > 5:
                        print(f"    ... and {len(errors) - 5} more errors")

                if not is_valid:
                    print(f"\n  {Fore.RED}❌ VALIDATION FAILED: Solution has invalid solutions!{Style.RESET_ALL}")
                    # Send a detailed raise exception
                    raise Exception(f"Validation failed: {errors}")
                else:
                    print(f"\n  {Fore.GREEN}✅ VALIDATION PASSED: All partitions have valid paths{Style.RESET_ALL}")

                # print(f"{'='*70}\n")

                # Store validity for penalty calculation
                total_validity_percentage += validity_percentage
                config_output["validity_percentage"] = validity_percentage
                config_output["is_valid"] = is_valid
                config_output["validation_stats"] = stats

                # Save the generated paths
                outf = f"{directory}/search_algorithm.json"
                setting = {
                    "algo": "search_algorithm",
                    "source_node": bc_t.src,
                    "terminal_nodes": bc_t.dsts,
                    "num_partitions": bc_t.num_partitions,
                    "generated_path": bc_t.paths,
                }
                config_output.update(setting)
                with open(outf, "w") as outfile:
                    outfile.write(json.dumps(setting))

                # Evaluate the generated paths
                input_dir = f"{directory}/paths/{config_name}"
                output_dir = f"{simulator_output}/evals/{config_name}"

                if not os.path.exists(output_dir):
                    Path(output_dir).mkdir(parents=True, exist_ok=True)

                # Run simulation
                simulator = BCSimulator(int(num_vms), output_dir)
                transfer_time, cost, output = simulator.evaluate_path(outf, config, write_to_file=True)
                config_output.update(output)
                aggregated_output.append(config_output)
                # Accumulate results
                total_cost += cost
                total_transfer_time += transfer_time
                successful_configs += 1
                total_search_time += search_time
                print(f"Config {config_name}: cost={cost:.2f}, time={transfer_time:.2f}, search_time={search_time:.2f}")
                
            except Exception as e:
                error_msg = f"Failed to process {os.path.basename(jsonfile)}: {str(e)}"
                print(error_msg)
                last_error_message = error_msg
                failed_configs += 1
                break
        
        # Check if we have any successful evaluations
        if failed_configs != 0:
            return {
                "combined_score": 0.0,
                "runs_successfully": 0.0,
                "error": last_error_message if last_error_message else "1 or more configuration files failed to process"
            }

        # save aggregated_output as csv file
        df = pd.DataFrame(aggregated_output)
        df.to_csv(f"{simulator_output}/aggregated_output.csv", index=False)
        # print(f"Aggregated output saved to {simulator_output}/aggregated_output.json")
        # Calculate aggregate metrics
        avg_cost = total_cost / successful_configs
        success_rate = successful_configs / (successful_configs + failed_configs)
        
        print(f"Summary: {successful_configs} successful, {failed_configs} failed")
        print(f"Total cost: {total_cost:.2f}, Max transfer time: {total_transfer_time:.2f}")
        
        # Calculate metrics for OpenEvolve
        # Normalize scores (higher is better)
        time_score = 1.0 / (1.0 + total_transfer_time)  # Lower time = higher score
        cost_score = 1.0 / (1.0 + total_cost)  # Lower cost = higher score
        search_score = 1.0 / (1.0 + total_search_time)  # Lower search time = higher score

        # Apply validation penalty
        avg_validity = total_validity_percentage / successful_configs if successful_configs > 0 else 0.0
        validity_penalty = avg_validity / 100.0  # 0.0 to 1.0, where 1.0 = 100% valid

        # Combined score considering total cost, max time, and success rate
        # Apply validity penalty: invalid solutions get heavily penalized
        if validity_penalty < 1.0:
            raise Exception(f"Validation failed: {errors}")
            combined_score = 0.0
        else:
            combined_score = cost_score
        
        return {
            "sim_dir": simulator_output,
            "combined_score": combined_score,  # Required by OpenEvolve
            "runs_successfully": success_rate,
            "total_cost": total_cost,
            "avg_cost": avg_cost,
            "max_transfer_time": total_transfer_time,
            "successful_configs": successful_configs,
            "failed_configs": failed_configs,
            "time_score": time_score,
            "cost_score": cost_score,
            "search_score": search_score,
            "success_rate": success_rate
        }

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        print(traceback.format_exc())
        return {
            "sim_dir": None,
            "combined_score": 0.0,  # Required by OpenEvolve
            "runs_successfully": 0.0,
            "error": str(e)
        }