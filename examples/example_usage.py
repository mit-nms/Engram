#!/usr/bin/env python3
"""
Example usage of the Evolution Optimizer with seed algorithms.

This script demonstrates how to run the evolution optimizer using 
predefined seed algorithms from a JSON file.
"""

import subprocess
import sys
import os


def run_evolution_with_seeds():
    """Run evolution optimization with seed algorithms."""
    
    # Get the Glia root directory (go up one level from examples/)
    glia_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Example command showing how to use the seeds
    cmd = [
        "python",
        "-m",
        "Architect.main",
        "--method",
        "evolution",
        "--model",
        "o3",  # or your preferred model
        "--task_prompt_path",
        os.path.join(glia_root, "SystemBench", "vidur", "deepagents_files", "task_prompt.txt"),
        "--evaluator_path",
        os.path.join(glia_root, "SystemBench", "vidur"),
        "--seeds_path",
        os.path.join(glia_root, "SystemBench", "vidur", "seeds.json"),  # Path to the seeds file we created
        "--population_size",
        "7", # number of seeds or will be updated with length seed codes of seeds.json
        "--num_generations",
        "200",
        "--results_dir",
        os.path.join(glia_root, "SystemBench", "vidur", "vidur_evolution_results_seed_population_sarathi_qps7.5"),
        "--debug",
    ]

    print("Running evolution with seeds:")
    print(" ".join(cmd))

    # Uncomment the line below to actually run the command
    subprocess.run(cmd)


if __name__ == "__main__":
    run_evolution_with_seeds()
