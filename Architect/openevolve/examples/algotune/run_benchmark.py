#!/usr/bin/env python3
"""
AlgoTune Benchmark Script for OpenEvolve

This script runs all AlgoTune tasks in the examples/algotune folder sequentially,
measures speedup achieved by OpenEvolve, and outputs detailed JSON results.

Usage:
    python examples/algotune/run_benchmark.py
    python examples/algotune/run_benchmark.py --iterations 50 --timeout 3600
    python examples/algotune/run_benchmark.py --tasks eigenvectors_complex,psd_cone_projection
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import statistics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark all AlgoTune tasks with OpenEvolve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/algotune/run_benchmark.py
  python examples/algotune/run_benchmark.py --iterations 50 --timeout 3600
  python examples/algotune/run_benchmark.py --tasks eigenvectors_complex,psd_cone_projection
  python examples/algotune/run_benchmark.py --output my_results.json
        """
    )
    
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=100,
        help="Number of iterations to run for each task (default: 100)"
    )
    
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=7200,  # 2 hours
        help="Timeout in seconds for each task (default: 7200)"
    )
    
    parser.add_argument(
        "--tasks",
        type=str,
        help="Comma-separated list of specific tasks to run (default: all tasks)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file path (default: algotune_benchmark_TIMESTAMP.json)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Custom config file to use instead of task default configs"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def discover_algotune_tasks(algotune_dir: Path) -> List[str]:
    """
    Discover all AlgoTune task directories.
    
    Args:
        algotune_dir: Path to examples/algotune directory
        
    Returns:
        List of task names (directory names)
    """
    tasks = []
    
    # Look for directories that contain the required files
    for item in algotune_dir.iterdir():
        if item.is_dir() and item.name not in ['__pycache__']:
            # Check if it has the required files
            required_files = ['initial_program.py', 'evaluator.py', 'config.yaml']
            if all((item / file).exists() for file in required_files):
                tasks.append(item.name)
    
    return sorted(tasks)


def run_openevolve_task(
    task_name: str,
    task_dir: Path,
    iterations: int,
    timeout: int,
    custom_config: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run OpenEvolve on a single AlgoTune task.
    
    Args:
        task_name: Name of the task
        task_dir: Path to the task directory
        iterations: Number of iterations to run
        timeout: Timeout in seconds
        custom_config: Optional custom config file path
        verbose: Enable verbose output
        
    Returns:
        Dictionary containing task results
    """
    print(f"Starting task: {task_name}")
    start_time = time.time()
    
    # Prepare command
    initial_program = task_dir / "initial_program.py"
    evaluator = task_dir / "evaluator.py"
    config = Path(custom_config) if custom_config else (task_dir / "config.yaml")
    
    # Get the project root (parent of examples/algotune)
    project_root = Path(__file__).parent.parent.parent
    openevolve_script = project_root / "openevolve-run.py"
    
    cmd = [
        sys.executable, str(openevolve_script),
        str(initial_program),
        str(evaluator),
        "--config", str(config),
        "--iterations", str(iterations)
    ]
    
    if verbose:
        print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run OpenEvolve
        result = subprocess.run(
            cmd,
            cwd=task_dir,  # Run from task directory
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        end_time = time.time()
        runtime_seconds = end_time - start_time
        
        if verbose:
            print(f"stdout: {result.stdout}")
            if result.stderr:
                print(f"stderr: {result.stderr}")
        
        # Parse results from output directory
        task_results = parse_task_results(task_dir, task_name, runtime_seconds)
        task_results["command_output"] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
        if result.returncode == 0:
            print(f"✅ {task_name} completed successfully in {runtime_seconds:.1f}s")
        else:
            print(f"❌ {task_name} failed with return code {result.returncode}")
            task_results["status"] = "failed"
            task_results["error"] = f"Process failed with return code {result.returncode}"
            
    except subprocess.TimeoutExpired:
        end_time = time.time()
        runtime_seconds = end_time - start_time
        print(f"⏰ {task_name} timed out after {timeout}s")
        task_results = {
            "status": "timeout",
            "error": f"Task timed out after {timeout} seconds",
            "runtime_seconds": runtime_seconds,
            "speedup": None
        }
        
    except Exception as e:
        end_time = time.time()
        runtime_seconds = end_time - start_time
        print(f"💥 {task_name} crashed: {str(e)}")
        task_results = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "runtime_seconds": runtime_seconds,
            "speedup": None
        }
    
    return task_results


def parse_task_results(task_dir: Path, task_name: str, runtime_seconds: float) -> Dict[str, Any]:
    """
    Parse OpenEvolve results from the output directory.
    
    Args:
        task_dir: Path to the task directory
        task_name: Name of the task
        runtime_seconds: Runtime in seconds
        
    Returns:
        Dictionary containing parsed results
    """
    # Look for openevolve_output directory
    output_dir = task_dir / "openevolve_output"
    best_dir = output_dir / "best"
    
    if not best_dir.exists():
        return {
            "status": "failed",
            "error": "No openevolve_output/best directory found",
            "runtime_seconds": runtime_seconds,
            "speedup": None
        }
    
    # Try to read best_program_info.json
    info_file = best_dir / "best_program_info.json"
    if info_file.exists():
        try:
            with open(info_file, 'r') as f:
                info = json.load(f)
            
            # Extract key metrics
            speedup = info.get("speedup_score", info.get("score"))
            if speedup is None:
                # Try to find speedup in evaluation_result
                eval_result = info.get("evaluation_result", {})
                speedup = eval_result.get("speedup_score", eval_result.get("score"))
            
            return {
                "status": "success",
                "iterations_run": info.get("iteration", "unknown"),
                "best_iteration": info.get("iteration", "unknown"),
                "speedup": speedup,
                "score": info.get("score"),
                "evaluation_result": info.get("evaluation_result", {}),
                "runtime_seconds": runtime_seconds,
                "best_program_path": str(best_dir / "best_program.py"),
                "best_program_info": info
            }
            
        except Exception as e:
            return {
                "status": "failed", 
                "error": f"Failed to parse best_program_info.json: {str(e)}",
                "runtime_seconds": runtime_seconds,
                "speedup": None
            }
    else:
        return {
            "status": "failed",
            "error": "No best_program_info.json found",
            "runtime_seconds": runtime_seconds,
            "speedup": None
        }


def calculate_algotune_score(speedups: List[float]) -> float:
    """
    Calculate AlgoTune Score as harmonic mean of speedups.
    
    Args:
        speedups: List of speedup values
        
    Returns:
        Harmonic mean of speedups
    """
    if not speedups:
        return 0.0
    
    # Filter out None and invalid values
    valid_speedups = [s for s in speedups if s is not None and s > 0]
    
    if not valid_speedups:
        return 0.0
    
    return statistics.harmonic_mean(valid_speedups)


def generate_summary(task_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics from task results.
    
    Args:
        task_results: Dictionary of task results
        
    Returns:
        Summary statistics
    """
    total_tasks = len(task_results)
    successful_tasks = sum(1 for r in task_results.values() if r.get("status") == "success")
    failed_tasks = total_tasks - successful_tasks
    
    # Collect speedups for successful tasks
    speedups = []
    for result in task_results.values():
        if result.get("status") == "success" and result.get("speedup") is not None:
            speedups.append(result["speedup"])
    
    # Calculate metrics
    avg_speedup = statistics.mean(speedups) if speedups else 0.0
    algotune_score = calculate_algotune_score(speedups)
    total_runtime = sum(r.get("runtime_seconds", 0) for r in task_results.values())
    
    return {
        "total_tasks": total_tasks,
        "successful_tasks": successful_tasks,
        "failed_tasks": failed_tasks,
        "average_speedup": round(avg_speedup, 3),
        "algotune_score": round(algotune_score, 3),
        "total_runtime_seconds": round(total_runtime, 1),
        "total_runtime_minutes": round(total_runtime / 60, 1),
        "speedups": speedups
    }


def main():
    """Main function."""
    args = parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    algotune_dir = script_dir
    
    print("🚀 AlgoTune Benchmark for OpenEvolve")
    print(f"📁 AlgoTune directory: {algotune_dir}")
    print(f"🔢 Iterations per task: {args.iterations}")
    print(f"⏱️  Timeout per task: {args.timeout}s")
    
    # Discover tasks
    all_tasks = discover_algotune_tasks(algotune_dir)
    if args.tasks:
        requested_tasks = [t.strip() for t in args.tasks.split(',')]
        tasks_to_run = [t for t in requested_tasks if t in all_tasks]
        missing_tasks = [t for t in requested_tasks if t not in all_tasks]
        if missing_tasks:
            print(f"⚠️  Warning: Tasks not found: {missing_tasks}")
    else:
        tasks_to_run = all_tasks
    
    print(f"📋 Found {len(all_tasks)} total tasks, running {len(tasks_to_run)} tasks")
    print(f"📝 Tasks to run: {', '.join(tasks_to_run)}")
    print()
    
    # Run tasks sequentially
    task_results = {}
    start_time = datetime.now()
    
    for i, task_name in enumerate(tasks_to_run, 1):
        print(f"[{i}/{len(tasks_to_run)}] Running {task_name}...")
        task_dir = algotune_dir / task_name
        
        result = run_openevolve_task(
            task_name=task_name,
            task_dir=task_dir,
            iterations=args.iterations,
            timeout=args.timeout,
            custom_config=args.config,
            verbose=args.verbose
        )
        
        task_results[task_name] = result
        
        # Print speedup if available
        if result.get("speedup") is not None:
            print(f"   Speedup: {result['speedup']:.3f}x")
        print()
    
    # Generate output
    end_time = datetime.now()
    summary = generate_summary(task_results)
    
    output_data = {
        "timestamp": start_time.isoformat(),
        "duration": str(end_time - start_time),
        "config": {
            "iterations": args.iterations,
            "timeout": args.timeout,
            "custom_config": args.config
        },
        "tasks": task_results,
        "summary": summary
    }
    
    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        output_file = Path(f"algotune_benchmark_{timestamp}.json")
    
    # Write results
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total tasks: {summary['total_tasks']}")
    print(f"Successful: {summary['successful_tasks']}")
    print(f"Failed: {summary['failed_tasks']}")
    print(f"Average speedup: {summary['average_speedup']:.3f}x")
    print(f"AlgoTune Score: {summary['algotune_score']:.3f}x")
    print(f"Total runtime: {summary['total_runtime_minutes']:.1f} minutes")
    print()
    
    if summary['speedups']:
        print("Individual speedups:")
        for task_name, result in task_results.items():
            if result.get("speedup") is not None:
                print(f"  {task_name:25}: {result['speedup']:6.3f}x")
    
    print(f"📄 Full results saved to: {output_file}")


if __name__ == "__main__":
    main()