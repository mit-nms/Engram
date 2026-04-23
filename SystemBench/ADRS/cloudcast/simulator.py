from typing import List
from pprint import pprint
import networkx as nx
import json
import colorama
from colorama import Fore, Style
from utils import networkx_to_graphviz
from broadcast import *
from utils import *
ENABLE_PRINT_DETAILS = False

class BCSimulator:
    # Default variables
    data_vol: float = 4.0  # size of data to be sent to multiple dsts
    num_partitions: int = 1
    partition_data_vol: int = data_vol / num_partitions
    default_vms_per_region: int = 1
    cost_per_instance_hr: float = 0.54  # based on m5.8xlarge spot
    src: str
    dsts: List[str]
    algo: str
    g = nx.DiGraph

    def __init__(self, num_vms, output_dir=None):
        # write output to file
        self.output_dir = output_dir
        self.default_vms_per_region = num_vms

    def initialization(self, path, config):
        # check if path is dict
        if isinstance(path, str):
            # Read from json
            with open(path, "r") as f:
                data = json.loads(f.read())
        else:
            data = {
                "algo": "none",
                "source_node": path.src,
                "terminal_nodes": path.dsts,
                "num_partitions": path.num_partitions,
                "generated_path": path.paths,
            }

        self.src = data["source_node"]
        self.dsts = data["terminal_nodes"]
        self.algo = data["algo"]
        self.paths = data["generated_path"]

        self.num_partitions = config["num_partitions"]
        self.data_vol = config["data_vol"]
        self.partition_data_vol = self.data_vol / self.num_partitions

        # Default in/egress limit if not set
        providers = ["aws", "gcp", "azure"]
        provider_ingress = [10, 16, 16]
        provider_egress = [5, 7, 16]
        self.ingress_limits = {providers[i]: provider_ingress[i] for i in range(len(providers))}
        self.egress_limits = {providers[i]: provider_egress[i] for i in range(len(providers))}

        if "ingress_limit" in config:
            for p, limit in config["ingress_limit"].items():
                self.ingress_limits[p] = self.default_vms_per_region * limit

        if "egress_limit" in config:
            for p, limit in config["egress_limit"].items():
                self.egress_limits[p] = self.default_vms_per_region * limit
        # print("Data vol (Gbit): ", self.data_vol * 8)
        # print("Ingress limits: ", self.ingress_limits)
        # print("Egress limits: ", self.egress_limits)

    def evaluate_path(self, path, config, write_to_file=False):
        # print(f"\n==============> Evaluation")
        self.initialization(path, config)

        # construct graph
        # print(f"\n--------- Algo: {self.algo}")
        self.g = self.__construct_g()
        # print("\n=> Data path to dests")
        # for path in self.__get_path():
        #     print("--")
        #     print(path)
        #     # NOTE: check
        #     for i in range(len(path) - 1):
        #         print(f"Flow: {self.g[path[i]][path[i+1]]['flow']}")
        #         print(f"Actual throughput: {round(self.g[path[i]][path[i+1]]['throughput'], 4)}")
        #         print(f"Cost: {self.g[path[i]][path[i+1]]['cost']}\n")

        # Print detailed per-partition, per-destination breakdown
        if ENABLE_PRINT_DETAILS:
            self.__print_detailed_paths()

        # evaluate transfer time and total cost
        max_t, avg_t, last_dst = self.__transfer_time()
        self.cost = self.__total_cost()

        # Print actual data received at each destination
        if ENABLE_PRINT_DETAILS:
            self.__print_data_received_summary()

        output = {
            "path": path,
            "max_transfer_time": max_t,
            "avg_transfer_time": avg_t,
            "last_dst": last_dst,
            "tot_cost": self.cost,
        }
        # output to json file
        if write_to_file:
            open(f"{self.output_dir}/{self.algo}_eval.json", "w").write(json.dumps(output))
        return max_t, self.cost, output

    def __construct_g(self):
        # construct a graph based on the given topology
        g = nx.DiGraph()
        # Ensure source and destination nodes are always in the graph
        g.add_node(self.src)
        for dst in self.dsts:
            g.add_node(dst)

        # Process paths for each destination
        for actual_dst in self.dsts:
            # Defensive check: ensure destination exists in paths dictionary
            if actual_dst not in self.paths:
                print(f"ERROR: Destination {actual_dst} not found in paths dictionary. Skipping.")
                raise ValueError(f"Destination {actual_dst} not found in paths dictionary! Check the search algorithm.")
            for partition_id in range(self.num_partitions):
                # print(self.paths)
                # print("Num of partitions: ", self.num_partitions)
                # Defensive check: ensure partition exists in paths dictionary
                if str(partition_id) not in self.paths[actual_dst]:
                    print(f"ERROR: Partition {partition_id} not found for destination {actual_dst}. Skipping.")
                    raise ValueError(f"Partition {partition_id} not found for destination {actual_dst}! Check the search algorithm.")
                path_list = self.paths[actual_dst][str(partition_id)]
                if path_list is None:
                    # Path not initialized, skip this partition
                    raise ValueError(f"Path not initialized for partition {partition_id} for destination {actual_dst}! Check the search algorithm.")
                for edge in path_list:
                    # CRITICAL: Use different variable names to avoid shadowing the outer loop 'dst'
                    edge_src, edge_dst, edge_data = edge[0], edge[1], edge[2]
                    if not g.has_edge(edge_src, edge_dst):
                        cost = edge_data["cost"]
                        throughput = edge_data["throughput"]  # * self.default_vms_per_region
                        g.add_edge(edge_src, edge_dst, throughput=throughput, cost=edge_data["cost"], flow=throughput)
                        g[edge_src][edge_dst]["partitions"] = set()
                    g[edge_src][edge_dst]["partitions"].add(partition_id)

        # h = networkx_to_graphviz(g, self.src, self.dsts, label="throughput")
        # h.render(view=True)

        # print(f"Default vms: {self.default_vms_per_region}")
        # Proportionally share if exceed in/egress limit of any node
        for node in g.nodes:
            # Defensive check: ensure node has valid provider format
            if ":" not in str(node):
                print(f"Warning: Node {node} does not have valid provider format. Skipping.")
                raise ValueError(f"Node {node} does not have valid provider format! Check the search algorithm.")
            provider = str(node).split(":")[0]

            # Defensive check: ensure provider exists in limits dictionaries
            if provider not in self.ingress_limits:
                print(f"Warning: Provider {provider} not found in ingress_limits. Using default.")
                raise ValueError(f"Provider {provider} not found in ingress_limits! Check the config.")
            if provider not in self.egress_limits:
                print(f"Warning: Provider {provider} not found in egress_limits. Using default.")
                raise ValueError(f"Provider {provider} not found in egress_limits! Check the config.")

            in_edges, out_edges = g.in_edges(node), g.out_edges(node)
            in_flow_sum = sum([g[i[0]][i[1]]["flow"] for i in in_edges])
            out_flow_sum = sum([g[o[0]][o[1]]["flow"] for o in out_edges])

            if in_flow_sum > self.ingress_limits[provider]:
                # print("\nExceed ingress limit")
                for edge in in_edges:
                    src, dst = edge[0], edge[1]
                    # assign based on flow proportion
                    # flow_proportion = g[src][dst]['throughput'] / in_flow_sum

                    # or assign based on num of incoming flows
                    flow_proportion = 1 / len(list(in_edges))

                    g[src][dst]["flow"] = min(g[src][dst]["flow"], self.ingress_limits[provider] * flow_proportion)

            if out_flow_sum > self.egress_limits[provider]:
                # print("\nExceed egress limit")
                for edge in out_edges:
                    src, dst = edge[0], edge[1]

                    # assign based on flow proportion
                    # flow_proportion = g[src][dst]['throughput'] / out_flow_sum

                    # or assign based on num of incoming flows
                    flow_proportion = 1 / len(list(out_edges))

                    # print(f"src: {src}, dst: {dst}, flow proportion: {flow_proportion}")
                    g[src][dst]["flow"] = min(g[src][dst]["flow"], self.egress_limits[provider] * flow_proportion)

        return g

    def __print_detailed_paths(self):
        """Print detailed information for each source-destination-partition combination."""
        print(f"\n{'='*80}")
        print(f"{Fore.BLUE}DETAILED PATH BREAKDOWN: Source -> Destination -> Partition{Style.RESET_ALL}")
        print(f"{'='*80}\n")

        for dst in self.dsts:
            print(f"{Fore.GREEN}Destination: {dst}{Style.RESET_ALL}")
            print(f"{'-'*80}")

            for partition_id in range(self.num_partitions):
                partition_str = str(partition_id)
                if partition_str not in self.paths[dst] or self.paths[dst][partition_str] is None:
                    print(f"  {Fore.YELLOW}Partition {partition_id}: No path defined{Style.RESET_ALL}\n")
                    continue

                path_list = self.paths[dst][partition_str]
                if len(path_list) == 0:
                    print(f"  {Fore.YELLOW}Partition {partition_id}: Empty path{Style.RESET_ALL}\n")
                    continue

                print(f"  {Fore.CYAN}Partition {partition_id}:{Style.RESET_ALL}")
                print(f"    Data Volume: {self.partition_data_vol:.4f} GB ({self.partition_data_vol * 8:.4f} Gbit)")
                print(f"    Path ({len(path_list)} edges):")

                total_path_cost = 0
                max_edge_time = 0
                path_nodes = []

                # Build path nodes list first
                if len(path_list) > 0:
                    path_nodes.append(path_list[0][0])  # Start with first edge's source
                    for edge in path_list:
                        path_nodes.append(edge[1])  # Add each edge's destination

                for edge_idx, edge in enumerate(path_list):
                    edge_src, edge_dst, edge_data_original = edge[0], edge[1], edge[2]

                    # Get actual edge data from graph (after flow limiting)
                    if edge_src in self.g and edge_dst in self.g[edge_src]:
                        edge_data = self.g[edge_src][edge_dst]
                        flow = edge_data["flow"]
                        cost_per_gb = edge_data["cost"]
                        throughput = edge_data["throughput"]
                        num_partitions_on_edge = len(edge_data["partitions"])

                        # Calculate metrics
                        edge_cost = self.partition_data_vol * cost_per_gb
                        # Transfer time = data_volume_in_gbit / flow_in_gbps
                        edge_time = (self.partition_data_vol * 8) / flow if flow > 1e-10 else float('inf')

                        total_path_cost += edge_cost
                        max_edge_time = max(max_edge_time, edge_time)

                        print(f"      Edge {edge_idx + 1}: {edge_src} -> {edge_dst}")
                        print(f"        Data: {self.partition_data_vol:.4f} GB")
                        print(f"        Cost: ${cost_per_gb:.6f}/GB × {self.partition_data_vol:.4f} GB = ${edge_cost:.6f}")
                        print(f"        Throughput: {throughput:.4f} Gbps (initial), {flow:.4f} Gbps (actual flow)")
                        print(f"        Partitions on edge: {num_partitions_on_edge}")
                        print(f"        Transfer Time: {edge_time:.6f} seconds")
                        print()
                    else:
                        # Edge not in graph - use original edge data if available
                        cost_per_gb_orig = edge_data_original.get("cost", 0.0) if isinstance(edge_data_original, dict) else 0.0
                        edge_cost = self.partition_data_vol * cost_per_gb_orig
                        total_path_cost += edge_cost

                        print(f"      {Fore.RED}Edge {edge_idx + 1}: {edge_src} -> {edge_dst} (NOT FOUND IN GRAPH!){Style.RESET_ALL}")
                        print(f"        Data: {self.partition_data_vol:.4f} GB")
                        if isinstance(edge_data_original, dict):
                            print(f"        Cost: ${cost_per_gb_orig:.6f}/GB × {self.partition_data_vol:.4f} GB = ${edge_cost:.6f} (from original data)")
                        print()

                print(f"    {Fore.YELLOW}Partition Summary:{Style.RESET_ALL}")
                print(f"      Path: {' -> '.join(path_nodes) if path_nodes else 'N/A'}")
                print(f"      Total Path Cost: ${total_path_cost:.6f}")
                if max_edge_time == float('inf'):
                    print(f"      Max Edge Transfer Time: ∞ (infinite - no valid path or zero flow)")
                    print(f"      Effective Throughput: 0 Gbps")
                elif max_edge_time > 0:
                    print(f"      Max Edge Transfer Time: {max_edge_time:.6f} seconds")
                    print(f"      Effective Throughput: {(self.partition_data_vol * 8) / max_edge_time:.4f} Gbps")
                else:
                    print(f"      Max Edge Transfer Time: N/A")
                    print(f"      Effective Throughput: N/A")
                print()

            print()

    def __get_path(self):
        all_paths = []
        # Check if source is in graph before trying to find paths
        if self.src not in self.g:
            return all_paths
        for node in self.dsts:
            if node not in self.g:
                raise ValueError(f"Destination {node} not found in graph! Check the search algorithm.")
            try:
                for path in nx.all_simple_paths(self.g, self.src, node):
                    all_paths.append(path)
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                # No path exists, skip this destination
                raise ValueError(f"No path exists for destination {node}! Check the search algorithm.")
        return all_paths

    def __slowest_capacity_link(self):
        min_tput = min([edge[-1]["throughput"] for edge in self.g.edges().data()])
        return min_tput

    def __transfer_time(self, log=True):
        # time for each (src, dst) pair
        t_dict = dict()
        for dst in self.dsts:
            # Defensive check: ensure destination exists in paths dictionary
            if dst not in self.paths:
                print(f"Warning: Destination {dst} not found in paths dictionary for transfer time calculation. Skipping.")
                t_dict[dst] = float("inf")  # Mark as unreachable
                raise ValueError(f"Destination {dst} not found in paths dictionary for transfer time calculation! Check the search algorithm.")
            partition_time = float("-inf")
            for i in range(self.num_partitions):
                # NOTE: how to calculate this? is it correct for both baseline and brute-force?
                # Defensive check: ensure partition exists in paths dictionary
                if str(i) not in self.paths[dst]:
                    raise ValueError(f"Partition {i} not found in paths dictionary for destination {dst}! Check the search algorithm.")
                path_list = self.paths[dst][str(i)]
                if path_list is None or len(path_list) == 0:
                    # No path for this partition, skip
                    raise ValueError(f"No path for partition {i} for destination {dst}! Check the search algorithm.")
                for edge in path_list:
                    # Use explicit variable names to avoid any confusion
                    edge_src, edge_dst = edge[0], edge[1]
                    if edge_src in self.g and edge_dst in self.g[edge_src]:
                        edge_data = self.g[edge_src][edge_dst]
                        partition_time = max(partition_time, len(edge_data["partitions"]) * self.partition_data_vol * 8 / edge_data["flow"])
            if partition_time == float("-inf"):
                # No valid paths found for this destination
                partition_time = float("inf")  # Mark as unreachable
            t_dict[dst] = partition_time

        max_t = max(t_dict.values())
        last_dst = [k for k, v in t_dict.items() if v == max_t]  # last dst receiving obj
        avg_t = sum(t_dict.values()) / len(t_dict.values())
        # assert(max_t == self.data_vol / self.__slowest_capacity_link()) # checking for single data copy case
        # if log:
        #     print(f"\n{Fore.BLUE}Algo: {Fore.YELLOW}{self.algo}{Style.RESET_ALL}")
        #     print(
        #         f"{Fore.BLUE}Data vol = {Fore.YELLOW}{self.data_vol} GB {Fore.BLUE}or {Fore.YELLOW}{self.data_vol * 8} Gbit{Style.RESET_ALL}"
        #     )
        #     print(f"\n{Fore.BLUE}Transfer time (s) for each destination: {Style.RESET_ALL}")
        #     pprint({key: round(value, 5) for key, value in t_dict.items()})
        #     print(f"{Fore.BLUE}Throughput (Gbps) for each destination: {Style.RESET_ALL}")
        #     pprint({key: round(self.data_vol * 8 / value, 5) for key, value in t_dict.items()})
        #     print(f"\n{Fore.BLUE}Max transfer time = {Fore.YELLOW}{round(max_t, 4)} s {Style.RESET_ALL}")
        #     print(
        #         f"{Fore.BLUE}Overall throughput = {Fore.YELLOW}{round(self.data_vol * 8 / max_t, 4)} Gbps{Style.RESET_ALL}"
        #     )  # data size / max transfer time
        #     print(f"{Fore.BLUE}Last dst receiving data = {Fore.YELLOW}{last_dst}{Style.RESET_ALL}")
        #     # print(f"The avg transfer time is: {round(avg_t, 3)}")
        return max_t, avg_t, last_dst

    def __total_cost(self):
        sum_egress_cost = 0
        total_data_transferred = 0  # Sum of all data transferred across all edges
        for edge in self.g.edges.data():
            edge_data = edge[-1]
            # Calculate data volume for this edge: number of partitions * partition size
            edge_data_volume = len(edge_data["partitions"]) * self.partition_data_vol
            total_data_transferred += edge_data_volume
            sum_egress_cost += (
                edge_data_volume * edge_data["cost"]
            )  ## TODO: is this calculation correct?

        runtime_s, _, _ = self.__transfer_time(log=False)
        runtime_s = round(runtime_s, 2)
        sum_instance_cost = 0
        for node in self.g.nodes():
            # print("Default vm per region: ", self.default_vms_per_region)
            # print("Cost per instance hr: ", (self.cost_per_instance_hr / 3600) * runtime_s)
            sum_instance_cost += self.default_vms_per_region * (self.cost_per_instance_hr / 3600) * runtime_s

        sum_cost = sum_egress_cost + sum_instance_cost
        # print(
        #     f"{Fore.BLUE}Total data transferred = {Fore.YELLOW}{round(total_data_transferred, 4)} GB{Style.RESET_ALL}"
        # )
        # print(
        #     f"{Fore.BLUE}Sum of total cost = egress cost {Fore.YELLOW}(${round(sum_egress_cost, 4)}) {Fore.BLUE}+ instance cost {Fore.YELLOW}(${round(sum_instance_cost, 4)}) {Fore.BLUE}= {Fore.YELLOW}${round(sum_cost, 3)}{Style.RESET_ALL}"
        # )
        return sum_cost

    def __print_data_received_summary(self):
        """Print summary of actual data received at each destination."""
        print(f"\n{'='*80}")
        print(f"{Fore.BLUE}DATA RECEIVED SUMMARY: Actual Data Transferred to Each Destination{Style.RESET_ALL}")
        print(f"{'='*80}\n")

        total_expected = self.data_vol * len(self.dsts)
        total_actual = 0

        for dst in self.dsts:
            valid_partitions = 0
            invalid_partitions = 0

            if dst not in self.paths:
                print(f"{Fore.RED}Destination {dst}: NOT FOUND in paths dictionary{Style.RESET_ALL}\n")
                continue

            for partition_id in range(self.num_partitions):
                partition_str = str(partition_id)
                if partition_str not in self.paths[dst] or self.paths[dst][partition_str] is None:
                    invalid_partitions += 1
                    continue

                path_list = self.paths[dst][partition_str]
                if len(path_list) == 0:
                    invalid_partitions += 1
                    continue

                # Check if path is valid (not a self-loop and actually reaches destination)
                is_valid = False
                path_ends_at_dst = False
                has_self_loop_only = True

                for edge in path_list:
                    edge_src, edge_dst = edge[0], edge[1]
                    # Check if it's not a self-loop
                    if edge_src != edge_dst:
                        has_self_loop_only = False
                    # Check if path ends at destination
                    if edge_dst == dst:
                        path_ends_at_dst = True
                        is_valid = True

                # Path is valid if it has non-self-loop edges and reaches destination
                if is_valid and not has_self_loop_only:
                    valid_partitions += 1
                else:
                    invalid_partitions += 1

            actual_data_gb = valid_partitions * self.partition_data_vol
            expected_data_gb = self.num_partitions * self.partition_data_vol
            total_actual += actual_data_gb

            percentage = (actual_data_gb / expected_data_gb * 100) if expected_data_gb > 0 else 0

            print(f"{Fore.GREEN}Destination: {dst}{Style.RESET_ALL}")
            print(f"  Expected data: {expected_data_gb:.4f} GB ({self.num_partitions} partitions × {self.partition_data_vol:.4f} GB)")
            print(f"  Actual data received: {Fore.YELLOW}{actual_data_gb:.4f} GB{Style.RESET_ALL} ({valid_partitions} valid partitions)")
            print(f"  Missing data: {expected_data_gb - actual_data_gb:.4f} GB ({invalid_partitions} invalid partitions)")
            print(f"  Completion: {Fore.YELLOW}{percentage:.1f}%{Style.RESET_ALL}")

            if invalid_partitions > 0:
                print(f"  {Fore.RED}⚠️  Warning: {invalid_partitions} partition(s) have invalid paths (self-loops or empty paths){Style.RESET_ALL}")
            print()

        print(f"{Fore.BLUE}Total Summary:{Style.RESET_ALL}")
        print(f"  Total expected across all destinations: {total_expected:.4f} GB")
        print(f"  Total actual received: {Fore.YELLOW}{total_actual:.4f} GB{Style.RESET_ALL}")
        print(f"  Total missing: {total_expected - total_actual:.4f} GB")
        overall_percentage = (total_actual / total_expected * 100) if total_expected > 0 else 0
        print(f"  Overall completion: {Fore.YELLOW}{overall_percentage:.1f}%{Style.RESET_ALL}")
        print(f"{'='*80}\n")