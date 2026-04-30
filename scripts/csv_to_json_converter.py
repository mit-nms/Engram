#!/usr/bin/env python3
"""
CSV to JSON Converter

Converts a CSV file with simulation results to JSON format similar to 
o3-vidur-agentic_evolution_1gen.json structure.

Usage:
    python csv_to_json_converter.py input.csv output.json

The CSV should have columns: simulation_number, request_e2e_time
The JSON output will have all_iterations ordered by simulation_number
with score = 20/request_e2e_time (if not nan)
"""

import csv
import json
import sys
import math
from typing import List, Dict, Any


def convert_csv_to_json(csv_file_path: str, json_file_path: str) -> None:
    """
    Convert CSV file to JSON format.
    
    Args:
        csv_file_path: Path to input CSV file
        json_file_path: Path to output JSON file
    """
    
    # Read CSV data
    iterations = []
    # simulation_number,avg_request_e2e_time,avg_request_e2e_slowdown,least_request_e2e_time_sofar,least_request_e2e_slowdown_sofar

    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            sim_number = int(row['simulation_number'])
            e2e_time = row['request_e2e_time_avg']
            slowdown = row.get('avg_request_e2e_slowdown', float('nan'))
            
            # Handle nan values
            if e2e_time.lower() == 'nan' or e2e_time == '':
                score = float("nan")  # or you could skip this iteration
            else:
                score = 20.0 / float(e2e_time)
            
            # Create iteration entry
            iteration = {
                "code": f"# Simulation {sim_number} code placeholder",
                "score": score,
                "reasoning": f"Simulation {sim_number} with E2E time: {e2e_time}",
                "metadata": {
                    "generation": 0,
                    "operation": "simulation",
                    "simulation_number": sim_number,
                    "request_e2e_time": e2e_time,
                    "request_e2e_slowdown": slowdown,
                    "role": "simulation_result"
                }
            }
            
            iterations.append(iteration)
    
    # Sort iterations by simulation number
    iterations.sort(key=lambda x: x['metadata']['simulation_number'])
    
    # Find best solution (highest score)
    best_solution = max(iterations, key=lambda x: x['score'])
    
    # Create final JSON structure
    result = {
        "best_solution": {
            "code": best_solution["code"],
            "score": best_solution["score"]
        },
        "all_iterations": iterations
    }
    
    # Write JSON file
    with open(json_file_path, 'w') as jsonfile:
        json.dump(result, jsonfile, indent=2)
    
    print(f"Successfully converted {csv_file_path} to {json_file_path}")
    print(f"Processed {len(iterations)} iterations")
    print(f"Best solution: Simulation {best_solution['metadata']['simulation_number']} with score {best_solution['score']:.6f}")


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 3:
        print("Usage: python csv_to_json_converter.py <input.csv> <output.json>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_json = sys.argv[2]
    
    try:
        convert_csv_to_json(input_csv, output_json)
    except FileNotFoundError:
        print(f"Error: File '{input_csv}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()