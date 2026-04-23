#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import hashlib
import os
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


@dataclass
class Code:
    code: str
    code_hash: str
    generation: int
    parent1_code: Optional[str] = None
    parent1_hash: Optional[str] = None
    parent2_code: Optional[str] = None
    parent2_hash: Optional[str] = None
    score: Optional[float] = None
    reasoning: Optional[str] = None
    parent_info: Optional[Dict] = None
    operation_name: Optional[str] = None

def process_code(code: str) -> str:
    """Process the code string to remove any extra whitespace or formatting."""
    code = code.strip()
    code = code.replace("\n", "")
    return code


def load_json_log(log_path: str) -> List[Dict]:
    """Load and parse the JSON log file."""
    with open(log_path, "r") as f:
        data = json.load(f)
        # We want all iterations to trace evolution history
        if isinstance(data, dict) and "all_iterations" in data:
            return data["all_iterations"]
        else:
            raise ValueError(f"Invalid log file format: {log_path} - missing all_iterations field")


def get_code_hash(code: str) -> str:
    """Generate a short hash for a code string."""
    if not code:
        return "none"
    code = process_code(code)
    return hashlib.md5(code.encode()).hexdigest()[:8]


def list_seeds(log_data: List[Dict]):
    """List all seeds that have no parent and their codes with scores."""
    print("\nAvailable Seeds:")
    print("=" * 80)
    hashes = set()
    for idx, individual in enumerate(log_data):
        code_hash = get_code_hash(individual.get("code", ""))
        parent_info = individual.get("metadata", {}).get("parent_info", {})
        generation = individual.get("metadata", {}).get("generation", 0)
        if not parent_info and not parent_info.get("parent1_code") and not parent_info.get("parent2_code"):
            hashes.add(code_hash)
            print(f"\nSeed {idx}:")
            print(f"  Hash: {code_hash}")
            print(f"  Generation: {generation}")
            print(f"  Code: {individual.get('code', 'N/A')}")
            print()
    print(f"Total seeds: {len(hashes)}")
    print(f"Seeds Hashes: {hashes}")


def list_all_generations(log_data: List[Dict]):
    """List all generations and their codes with scores."""
    print("\nAvailable Codes:")
    print("=" * 80)

    for idx, individual in enumerate(log_data):
        code_hash = get_code_hash(individual.get("code", ""))
        score = individual.get("score", "N/A")
        parent_info = individual.get("metadata", {}).get("parent_info", {})
        parent_code = parent_info.get("parent1_code", "")
        parent_hash = get_code_hash(parent_code) if parent_code else "None"

        print(f"\nCode {idx}:")
        print(f"  Hash: {code_hash}")
        print(f"  Parent Hash: {parent_hash}")
        print()


def build_population_map(log_data: List[Dict], remove_diversify: bool = False) -> Dict[str, Code]:
    """Build a map of all individuals using code hash as key."""
    population_map = {}
    for individual in log_data:
        code = individual.get("code", "")
        if not code:
            continue

        code_hash = get_code_hash(code)
        metadata = individual.get("metadata", {})
        generation = metadata.get("generation", 0)
        parent_info = metadata.get("parent_info", {})
        parent1_key = "parent_code" if "parent_code" in parent_info else "parent1_code"
        parent1_code = parent_info.get(parent1_key, None)
        parent1_hash = get_code_hash(parent1_code) if parent1_code else None
        parent2_code = parent_info.get("parent2_code", None)
        parent2_hash = get_code_hash(parent2_code) if parent2_code else None
        operation_name = metadata.get("operation_name", None)

        # Validate that non-seed nodes must have parents
        if generation > 0 and (parent1_hash is None and parent2_hash is None):
            print(f"Parent info: {parent_info}")
            print(f"\nError: Non-generation-0 node {code_hash} has no parents!")
            print(f"Generation: {generation}")
            raise ValueError(f"Non-generation-0 node {code_hash} has no parents!")

        population_map[code_hash] = Code(
            code=code,
            code_hash=code_hash,
            generation=generation,
            parent1_code=parent1_code,
            parent1_hash=parent1_hash,
            parent2_code=parent2_code,
            parent2_hash=parent2_hash,
            score=individual.get("score"),
            reasoning=individual.get("reasoning"),
            parent_info=parent_info,
            operation_name=operation_name,
        )
    if remove_diversify:
        population_map = remove_diversify_function(population_map)
    return population_map

def remove_diversify_function(population_map: Dict[str, Code]) -> Dict[str, Code]:
    """Remove all nodes that have operation_name 'diversify', and all their descendants."""
    diversify_hash_list = [code_hash for code_hash, code in population_map.items() if code.operation_name == "diversify"]
    for code_hash in diversify_hash_list:
        print(f"Removing diversify node: {code_hash}")
        population_map = remove_node(population_map, code_hash)
    return population_map

def remove_node(population_map: Dict[str, Code], code_hash: str) -> Dict[str, Code]:
    """Remove a node and all its descendants."""
    descendants = find_descendants(code_hash, population_map)
    for descendant_hash in descendants:
        print(f"Removing node: {descendant_hash}")
        del population_map[descendant_hash]
    return population_map

def find_descendants(code_hash: str, population_map: Dict[str, Code]) -> Set[str]:
    """Find all descendants of a given code."""
    descendants = set()
    to_check = {code_hash}

    while to_check:
        current = to_check.pop()
        for ind_hash, individual in population_map.items():
            # Check if either parent hash matches the current code
            if (
                individual.parent1_hash == current or individual.parent2_hash == current
            ) and ind_hash not in descendants:
                descendants.add(ind_hash)
                to_check.add(ind_hash)

    return descendants


def find_ancestors(code_hash: str, population_map: Dict[str, Code]) -> Set[str]:
    """Find all ancestors of a given code by following both parent paths."""
    ancestors = set()
    to_check = {code_hash}

    while to_check:
        current = to_check.pop()
        if current not in population_map:
            continue

        # Check both parents
        if population_map[current].parent1_hash:
            parent1 = population_map[current].parent1_hash
            if parent1 not in ancestors:
                ancestors.add(parent1)
                to_check.add(parent1)

        if population_map[current].parent2_hash:
            parent2 = population_map[current].parent2_hash
            if parent2 not in ancestors:
                ancestors.add(parent2)
                to_check.add(parent2)

    return ancestors


def print_tree(code_hash: str, population_map: Dict[str, Code], prefix: str = "", is_last: bool = True):
    """Print a tree representation of an individual and its details."""
    if code_hash not in population_map:
        print(f"{prefix}{'└── ' if is_last else '├── '}[Hash: {code_hash}] - Not in current population")
        return

    individual = population_map[code_hash]

    # Prepare the branch symbol
    branch = "└── " if is_last else "├── "

    # Print the current node with all available information
    print(f"{prefix}{branch}[Hash: {code_hash}]")
    print(f"{prefix}{'    ' if is_last else '│   '}Generation: {individual.generation}")
    # Prepare the prefix for children
    new_prefix = prefix + ("    " if is_last else "│   ")


def print_lineage_tree(log_data: List[Dict], target_code: Optional[str] = None, target_hash: Optional[str] = None):
    """Print the complete lineage tree including ancestors and descendants."""
    # Build population map
    population_map = build_population_map(log_data)
    if target_code is not None:
        target_code = process_code(target_code)
        target_hash = get_code_hash(target_code)
    elif target_hash is not None:
        # retrive target code from population map
        target_code = population_map[target_hash].code

    if target_hash is None and target_code is None:
        raise ValueError("Either target_code or target_hash must be provided")

    # Find ancestors and descendants
    ancestors = find_ancestors(target_hash, population_map)
    descendants = find_descendants(target_hash, population_map)

    print("\nLineage Tree")
    print("=" * 80)

    # Print ancestors (if any)
    print("\nAncestors:")
    print("-" * 40)
    if ancestors:
        sorted_ancestors = sorted(
            [a for a in ancestors if a in population_map],
            key=lambda x: population_map[x].score if population_map[x].score is not None else -float("inf"),
        )
        for idx, ancestor_hash in enumerate(sorted_ancestors):
            print_tree(ancestor_hash, population_map, is_last=(idx == len(sorted_ancestors) - 1))

    # Print the target individual
    print("\nTarget Code:")
    print("-" * 40)
    print_tree(target_hash, population_map)
    print(f"Target Code: {target_code}")

    # Print descendants (if any)
    print("\nDescendants:")
    print("-" * 40)
    if descendants:
        sorted_descendants = sorted(
            [d for d in descendants if d in population_map],
            key=lambda x: population_map[x].score if population_map[x].score is not None else -float("inf"),
        )
        for idx, descendant_hash in enumerate(sorted_descendants):
            print_tree(descendant_hash, population_map, is_last=(idx == len(sorted_descendants) - 1))


def find_seed_ancestry(
    code_hash: str,
    population_map: Dict[str, Code],
    memo: Dict[str, Dict[str, float]] = None,
    debug_chain: Set[str] = None,
) -> Dict[str, float]:
    """
    Recursively trace back to find all seed ancestors and their proportional influence.
    Returns a dictionary mapping seed hashes to their proportional influence (0-1).
    """
    if memo is None:
        memo = {}
    if debug_chain is None:
        debug_chain = set()

    # Add current node to debug chain
    debug_chain.add(code_hash)

    # If we've already computed this, return memoized result
    if code_hash in memo:
        return memo[code_hash]

    code = population_map[code_hash]

    # If this is a generation 0 node, it's a seed
    if code.generation == 0:
        result = {code_hash: 1.0}
        memo[code_hash] = result
        return result

    # If this is not generation 0 but has no parents, it's an error
    if not code.parent1_hash and not code.parent2_hash:
        print(f"\nError: Non-generation-0 node {code_hash} has no parents!")
        print(f"Generation: {code.generation}")
        raise ValueError(f"Non-generation-0 node {code_hash} has no parents!")

    # Verify both parents exist in population_map
    if code.parent1_hash and code.parent1_hash not in population_map:
        print(f"\nError: Parent1 {code.parent1_hash} not found for node {code_hash}!")
        print(f"Generation: {code.generation}")
        raise ValueError(f"Parent1 {code.parent1_hash} not found!")

    if code.parent2_hash and code.parent2_hash not in population_map:
        print(f"\nError: Parent2 {code.parent2_hash} not found for node {code_hash}!")
        print(f"Generation: {code.generation}")
        raise ValueError(f"Parent2 {code.parent2_hash} not found!")

    # Initialize empty result
    result = {}

    # Check if parents are the same code
    has_duplicate_parents = (code.parent1_hash == code.parent2_hash) and code.parent1_hash is not None

    # Process parents and collect ONLY generation 0 seed proportions
    if has_duplicate_parents:
        # For duplicate parents, verify both exist and get ancestry from parent1
        if code.parent1_hash not in population_map or code.parent2_hash not in population_map:
            print(f"\nError: Missing parent for node {code_hash} with duplicate parents!")
            print(f"Parent hash: {code.parent1_hash}")
            print(f"Generation: {code.generation}")
            raise ValueError(f"Missing parent {code.parent1_hash}!")

        parent_seeds = find_seed_ancestry(code.parent1_hash, population_map, memo, debug_chain)
        # Only copy proportions from generation 0 seeds
        for seed, proportion in parent_seeds.items():
            if seed in population_map and population_map[seed].generation == 0:
                result[seed] = proportion
            else:
                print(f"\nWarning: Dropping non-generation-0 ancestor {seed} from node {code_hash}'s ancestry")
    else:
        # Process parents separately and average their contributions
        num_parents = 0

        # Process parent1
        if code.parent1_hash:
            num_parents += 1
            parent1_seeds = find_seed_ancestry(code.parent1_hash, population_map, memo, debug_chain)
            # Only copy proportions from generation 0 seeds
            for seed, proportion in parent1_seeds.items():
                if seed in population_map and population_map[seed].generation == 0:
                    result[seed] = result.get(seed, 0.0) + proportion
                else:
                    print(
                        f"\nWarning: Dropping non-generation-0 ancestor {seed} from node {code_hash}'s parent1 ancestry"
                    )

        # Process parent2
        if code.parent2_hash:
            num_parents += 1
            parent2_seeds = find_seed_ancestry(code.parent2_hash, population_map, memo, debug_chain)
            # Only copy proportions from generation 0 seeds
            for seed, proportion in parent2_seeds.items():
                if seed in population_map and population_map[seed].generation == 0:
                    result[seed] = result.get(seed, 0.0) + proportion
                else:
                    print(
                        f"\nWarning: Dropping non-generation-0 ancestor {seed} from node {code_hash}'s parent2 ancestry"
                    )

        # Normalize proportions
        if num_parents > 0:  # Safety check
            for seed in result:
                result[seed] /= num_parents

    # If we ended up with no valid seeds, this is an error
    if not result:
        print(f"\nError: Code {code_hash} has no generation-0 seed ancestors!")
        print(f"Generation: {code.generation}")
        print(f"Parents: {code.parent1_hash}, {code.parent2_hash}")
        print(f"Ancestry chain: {' -> '.join(debug_chain)}")
        raise ValueError(f"Code {code_hash} has no valid seed ancestors!")

    # Verify all proportions sum to 1.0 (within floating point error)
    total = sum(result.values())
    if abs(total - 1.0) > 1e-10:
        print(f"\nWarning: Node {code_hash} proportions sum to {total}, normalizing...")
        for seed in result:
            result[seed] /= total

    memo[code_hash] = result
    debug_chain.remove(code_hash)  # Remove this node from chain before returning
    return result


def get_seed_colors(population_map: Dict[str, Code]) -> Dict[str, List[float]]:
    """Assign unique colors to seed codes and calculate mixed colors for descendants."""
    # First identify all seeds (codes with generation 0)
    seeds = [
        code_hash
        for code_hash, code in population_map.items()
        if code.generation == 0 and code.parent1_hash is None and code.parent2_hash is None
    ]
    print(f"Seeds: {seeds}")
    if not seeds:
        raise ValueError("No generation 0 seeds found in the population!")

    # Create maximally distant colors for seeds
    base_colors = []
    for i in range(len(seeds)):
        hue = i / len(seeds)  # Evenly spaced hues
        color = plt.cm.hsv(hue)  # This gives RGBA
        # Make the color more vivid
        color = [color[0], color[1], color[2], 1.0]
        base_colors.append(color)

    # Reorder colors to maximize distance between adjacent colors
    if len(seeds) > 2:
        reordered_colors = []
        indices = list(range(len(seeds)))
        current_idx = 0
        while indices:
            reordered_colors.append(base_colors[current_idx])
            indices.remove(current_idx)
            if indices:
                distances = [abs((idx - current_idx + len(seeds) / 2) % len(seeds) - len(seeds) / 2) for idx in indices]
                current_idx = indices[distances.index(max(distances))]
        base_colors = reordered_colors

    # Create a mapping from seed hash to its color
    seed_to_color = {seed: base_colors[i] for i, seed in enumerate(seeds)}

    # For each code, compute its seed ancestry and corresponding colors
    color_map = {}
    memo = {}  # Memoization dictionary for find_seed_ancestry

    for code_hash in population_map:
        try:
            # For seeds, just use their assigned color
            if code_hash in seed_to_color:
                color_map[code_hash] = {"colors": [seed_to_color[code_hash]], "proportions": [1.0]}
                continue

            # Validate that non-seed nodes have parents
            code = population_map[code_hash]
            if code.generation > 0 and not code.parent1_hash and not code.parent2_hash:
                print(f"\nError: Non-generation-0 node {code_hash} has no parents!")
                print(f"Generation: {code.generation}")
                raise ValueError(f"Non-generation-0 node {code_hash} has no parents!")

            # For all other nodes, get seed proportions
            try:
                seed_proportions = find_seed_ancestry(code_hash, population_map, memo)
            except ValueError as e:
                print(f"Generation: {code.generation}")
                print(f"Parents: {code.parent1_hash}, {code.parent2_hash}")
                raise  # Re-raise the error instead of continuing

            # Convert seed proportions to colors
            colors = []
            proportions = []
            for seed, proportion in seed_proportions.items():
                if proportion > 0.001:  # Filter out very small proportions
                    if seed not in seed_to_color:
                        print(f"\nWarning: Node {code_hash} has proportion from non-seed {seed}")
                        print(f"Generation: {code.generation}")
                        print(f"Parents: {code.parent1_hash}, {code.parent2_hash}")
                        continue
                    colors.append(seed_to_color[seed])
                    proportions.append(proportion)

            # Skip nodes that ended up with no valid colors
            if not colors:
                print(f"\nWarning: Node {code_hash} has no valid seed colors")
                print(f"Generation: {code.generation}")
                print(f"Parents: {code.parent1_hash}, {code.parent2_hash}")
                continue

            # Normalize proportions to ensure they sum to 1.0
            total = sum(proportions)
            proportions = [p / total for p in proportions]

            color_map[code_hash] = {"colors": colors, "proportions": proportions}

        except Exception as e:
            print(f"\nError processing node {code_hash}: {str(e)}")
            print(f"Generation: {population_map[code_hash].generation}")
            print(f"Parents: {population_map[code_hash].parent1_hash}, {population_map[code_hash].parent2_hash}")
            raise  # Re-raise the error instead of continuing

    return color_map


def create_population_graph(population_map: Dict[str, Code]) -> nx.DiGraph:
    """Create a NetworkX directed graph from the population map."""
    G = nx.DiGraph()

    # Get color information that traces back to seeds
    color_map = get_seed_colors(population_map)

    # Group codes by generation for layered layout
    by_generation = defaultdict(list)
    for code_hash, code in population_map.items():
        # Only include nodes that have valid colors
        if code_hash in color_map:
            by_generation[code.generation].append(code_hash)

    # print the hash codes of each node in by_generation
    for gen, codes in by_generation.items():
        print(f"Generation {gen}: {codes}")

    # First add all nodes with valid colors
    for code_hash in color_map:
        code = population_map[code_hash]
        # Store node information
        G.add_node(
            code_hash,
            generation=code.generation,
            colors=color_map[code_hash]["colors"],
            proportions=color_map[code_hash]["proportions"],
            score=code.score  # Add score to node attributes
        )

    # Add edges only between nodes that have valid colors
    for code_hash in color_map:
        code = population_map[code_hash]
        if code.parent1_hash and code.parent1_hash in color_map:
            G.add_edge(code.parent1_hash, code_hash)
        if code.parent2_hash and code.parent2_hash in color_map:
            G.add_edge(code.parent2_hash, code_hash)

    # Use hierarchical layout
    pos = nx.spring_layout(G)

    # Adjust positions to ensure generations are properly layered
    max_gen = max(by_generation.keys())
    layer_height = 2.0  # Height between layers

    # Calculate x positions for each generation
    for gen in range(max_gen + 1):
        nodes_in_gen = by_generation[gen]
        num_nodes = len(nodes_in_gen)
        for i, node in enumerate(sorted(nodes_in_gen)):
            x = (i - (num_nodes - 1) / 2) * 2.0  # Horizontal spacing
            y = (max_gen - gen) * layer_height  # Top to bottom
            pos[node] = np.array([x, y])

    # Store positions in the graph
    nx.set_node_attributes(G, pos, "pos")

    return G


def plot_population_graph(G: nx.DiGraph, output_path: str):
    """Plot the population graph using matplotlib."""
    plt.figure(figsize=(20, 12))

    # Get node positions
    pos = nx.get_node_attributes(G, "pos")

    # Draw edges with straight arrows
    # Only draw edges that go from a higher y-value to a lower y-value (top to bottom)
    for edge in G.edges():
        start_pos = pos[edge[0]]
        end_pos = pos[edge[1]]
        if start_pos[1] > end_pos[1]:  # Only draw if parent is above child
            plt.arrow(
                start_pos[0],
                start_pos[1],
                end_pos[0] - start_pos[0],
                end_pos[1] - start_pos[1],
                head_width=0.15,
                head_length=0.3,
                fc="gray",
                ec="gray",
                alpha=0.3,
                length_includes_head=True,
            )

    # Draw nodes as pie charts
    for node in G.nodes():
        x, y = pos[node]
        colors = G.nodes[node]["colors"]
        sizes = G.nodes[node]["proportions"]
        gen = G.nodes[node]["generation"]

        # Create pie chart without labels
        plt.pie(
            sizes,
            colors=colors,
            center=(x, y),
            radius=0.4,
            wedgeprops={"alpha": 0.8, "linewidth": 0.5, "edgecolor": "white"},
        )

        # Add hash label inside the node
        # Use a small font size and wrap text to fit
        if len(G.nodes()) < 100:
            plt.text(
                x, y,  # Position at center of node
                node,  # The hash string
                ha='center',  # Horizontal alignment
                va='center',  # Vertical alignment
                fontsize=6,   # Very small font
                color='black',  # Black text
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.1)  # White background for readability
            )

    # Group nodes by generation and find max score for each generation
    gen_scores = defaultdict(list)
    for node in G.nodes():
        gen = G.nodes[node]["generation"]
        score = G.nodes[node].get("score", None)
        if score is not None:
            gen_scores[gen].append(score)

    # Add max score labels for each generation
    max_x = max(pos[node][0] for node in G.nodes()) + 2.0  # Add some padding
    for gen, scores in gen_scores.items():
        if scores:  # Only add label if there are scores
            y = (max(G.nodes[node]["generation"] for node in G.nodes()) - gen) * 2.0  # Match layer height
            max_score = max(scores)
            plt.text(
                max_x,
                y,
                f"Gen {gen}\nMax: {max_score:.3f}",
                ha='left',
                va='center',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
            )

    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True, pad_inches=0.2)
    plt.close()
    print(f"\nPopulation graph saved to: {output_path}")


def plot_population_map(log_data: List[Dict], output_path: Optional[str] = None, remove_diversify: bool = True):
    """Create and save a visualization of the population map."""
    population_map = build_population_map(log_data, remove_diversify=True)
    G = create_population_graph(population_map)

    if output_path:
        plot_population_graph(G, output_path)
    else:
        plot_population_graph(G, "population_evolution.png")


def main():
    """
    first, run --list_all or --list_seeds to get the code hashes
    then, run --code_hash with the code hash to get the lineage
    or run --code with the code to get the lineage
    """
    parser = argparse.ArgumentParser(description="Build and display a lineage tree for a specific code.")
    parser.add_argument("--log_file", type=str, help="Path to the JSON log file")
    parser.add_argument("--code", type=str, help="A unique substring of the code to trace")
    parser.add_argument("--code_hash", type=str, help="The hash of the code to trace")
    parser.add_argument("--list_all", action="store_true", help="List all available codes")
    parser.add_argument("--list_seeds", action="store_true", help="List all available seeds")
    parser.add_argument("--plot", action="store_true", help="Plot the population evolution diagram")
    parser.add_argument(
        "--output", type=str, default="population_evolution.png", help="Save the diagram to this PNG file path"
    )

    args = parser.parse_args()

    if not args.log_file:
        print("Please provide the log file path using --log_file")
        return

    # Verify log file exists
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {args.log_file}")
        return

    try:
        # Load the log data
        log_data = load_json_log(args.log_file)

        if args.plot:
            plot_population_map(log_data, args.output)
        elif args.list_all:
            list_all_generations(log_data)
        elif args.list_seeds:
            list_seeds(log_data)
        elif args.code:
            if os.path.isfile(args.code):
                # read the file and get the code
                with open(args.code, "r") as f:
                    code = f.read()
            else:
                code = args.code
            print(f"Tracing code: {code[:100]}...")
            print_lineage_tree(log_data, code)
        elif args.code_hash:
            print(f"Tracing code hash: {args.code_hash}")
            print_lineage_tree(log_data, target_hash=args.code_hash)
        else:
            print("Please specify either --list to see all codes or --code to trace a specific code")

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in log file: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
