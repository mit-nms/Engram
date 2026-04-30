#!/usr/bin/env python3
"""
Aggregate OpenEvolve results into a single JSON file similar to Vidur format.

Usage:
    python aggregate_openevolve_logs.py <openevolve_dir> [output_file]
"""

import json
import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional


def read_program_json(json_path: Path) -> Optional[Dict[str, Any]]:
    """Read a program JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to read {json_path}: {e}", file=sys.stderr)
        return None


def read_program_code(program_path: Path) -> Optional[str]:
    """Read program code from program.py file."""
    try:
        with open(program_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Failed to read {program_path}: {e}", file=sys.stderr)
        return None


def get_code_from_json(program_json: Dict[str, Any]) -> Optional[str]:
    """Extract code from JSON if available."""
    return program_json.get('code')


def scan_generated_programs(openevolve_dir: Path) -> List[Dict[str, Any]]:
    """Scan generated_programs directory and collect all programs."""
    generated_dir = openevolve_dir / "generated_programs"
    
    if not generated_dir.exists():
        print(f"Error: {generated_dir} does not exist", file=sys.stderr)
        return []
    
    programs = []
    
    # Find all generated_N directories
    generated_dirs = sorted(
        generated_dir.glob("generated_*"),
        key=lambda x: int(x.name.split("_")[1]) if x.name.split("_")[1].isdigit() else 0
    )
    
    for gen_dir in generated_dirs:
        # Find JSON files in this directory
        json_files = list(gen_dir.glob("*.json"))
        program_py = gen_dir / "program.py"
        
        # Extract iteration number from directory name
        try:
            dir_iteration = int(gen_dir.name.split("_")[1])
        except (ValueError, IndexError):
            dir_iteration = None
        
        for json_file in json_files:
            program_json = read_program_json(json_file)
            if not program_json:
                continue
            
            # Try to get code from program.py first, then from JSON
            code = read_program_code(program_py)
            if not code:
                code = get_code_from_json(program_json)
            
            if not code:
                print(f"Warning: No code found for {json_file}", file=sys.stderr)
                continue
            
            # Determine iteration number (prefer from JSON, fallback to directory name)
            iteration = (
                program_json.get('iteration_found') or
                program_json.get('generation') or
                dir_iteration or
                0
            )
            
            programs.append({
                'json_data': program_json,
                'code': code,
                'iteration': iteration,
                'json_path': json_file,
                'program_path': program_py
            })
    
    return programs


def map_to_vidur_format(programs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert OpenEvolve programs to Vidur-like format."""
    # Sort by iteration
    programs.sort(key=lambda x: x['iteration'])
    
    all_iterations = []
    best_score = float('-inf')
    best_solution = None
    
    for idx, prog in enumerate(programs):
        json_data = prog['json_data']
        metrics = json_data.get('metrics', {})
        combined_score = metrics.get('combined_score', float('-inf'))

        # Track best solution
        if combined_score > best_score:
            best_score = combined_score
            best_solution = {
                'code': prog['code'],
                'score': combined_score
            }
        
        # Map to Vidur format
        iteration_data = {
            'iteration': idx + 1,  # 1-indexed like Vidur
            'code': prog['code'],
            'score': combined_score,
            'analysis': '',  # OpenEvolve doesn't have analysis field
            'success': metrics.get('runs_successfully', 1.0) > 0.5,
            'sim_dirs': [],  # OpenEvolve doesn't track sim dirs
            'error': '',
            'retry_needed': False,
            'usage_stats': {
                'total_cost': metrics.get('total_cost', 0.0),
                'total_prompt_tokens': 0,  # Not tracked in OpenEvolve
                'total_completion_tokens': 0,
                'total_tokens': 0,
                'model': 'unknown',
                'pricing_info': {
                    'input': 0.0,
                    'output': 0.0
                },
                'component': 'openevolve'
            },
            # Additional OpenEvolve-specific fields
            'openevolve_metadata': {
                'id': json_data.get('id'),
                'generation': json_data.get('generation'),
                'iteration_found': json_data.get('iteration_found'),
                'parent_id': json_data.get('parent_id'),
                'timestamp': json_data.get('timestamp'),
                'metrics': metrics,
                'complexity': json_data.get('complexity', 0.0),
                'diversity': json_data.get('diversity', 0.0)
            }
        }
        
        all_iterations.append(iteration_data)
    
    # If no best solution found, use the last one
    if not best_solution and all_iterations:
        best_solution = {
            'code': all_iterations[-1]['code'],
            'score': all_iterations[-1]['score']
        }
    
    # Ensure we have a best solution
    if not best_solution:
        best_solution = {'code': '', 'score': 0.0}
    
    result = {
        'best_solution': best_solution,
        'all_iterations': all_iterations,
        'final_code': best_solution['code'],
        'total_iterations': len(all_iterations),
        'convergence_reason': 'completed',  # OpenEvolve doesn't track this
        'baselines': {}
    }
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python aggregate_openevolve_logs.py <openevolve_dir> [output_file]")
        sys.exit(1)
    
    openevolve_dir = Path(sys.argv[1])
    if not openevolve_dir.exists():
        print(f"Error: Directory {openevolve_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Determine output file
    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])
    else:
        output_file = openevolve_dir / "aggregated_results.json"
    
    print(f"Scanning OpenEvolve directory: {openevolve_dir}")
    programs = scan_generated_programs(openevolve_dir)
    print(f"Found {len(programs)} programs")
    
    if not programs:
        print("Error: No programs found", file=sys.stderr)
        sys.exit(1)
    
    print("Converting to Vidur format...")
    result = map_to_vidur_format(programs)
    
    print(f"Writing results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Done! Best score: {result['best_solution']['score']}")
    print(f"Total iterations: {result['total_iterations']}")


if __name__ == "__main__":
    main()

