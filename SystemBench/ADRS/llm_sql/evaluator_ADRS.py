import sys
import os
import importlib.util
import traceback
import time
import json
import gc
import csv
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

# Suppress pandas FutureWarning about incompatible dtype
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils import evaluate_df_prefix_hit_cnt
from solver import Algorithm

evaluator_dir = os.path.dirname(os.path.abspath(__file__))

# Check mounted datasets directory first (from main repo datasets folder)
mounted_datasets_dir = "/datasets/llm_sql"
if os.path.exists(mounted_datasets_dir) and os.listdir(mounted_datasets_dir):
    datasets_dir = mounted_datasets_dir
    print(f"[evaluator] Using mounted datasets directory: {datasets_dir}", file=sys.stderr, flush=True)
else:
    datasets_dir = os.path.join(evaluator_dir, "datasets")
    print(f"[evaluator] Using local datasets directory: {datasets_dir}", file=sys.stderr, flush=True)

# Dataset configurations
TEST_FILES = [
    os.path.join(datasets_dir, "movies.csv"),
    os.path.join(datasets_dir, "beer.csv"),
    os.path.join(datasets_dir, "BIRD.csv"),
    os.path.join(datasets_dir, "PDMX.csv"),
    os.path.join(datasets_dir, "products.csv"),
]

COL_MERGES = [
    [['movieinfo', 'movietitle', 'rottentomatoeslink']],
    [['beer/beerId', 'beer/name']],
    [['PostId', 'Body']],
    [['path', 'metadata'], ['hasmetadata', 'isofficial', 'isuserpublisher', 'isdraft', 'hasannotations', 'subsetall']],
    [['product_title', 'parent_asin']],
]


def _process_dataset_worker(args_tuple):
    """Worker function for parallel dataset evaluation - must be at module level for pickling."""
    csv_path, col_merge, program_path, parent_dir_path, evaluator_dir_path, idx, total = args_tuple

    # Ensure paths are in sys.path for imports
    if parent_dir_path not in sys.path:
        sys.path.insert(0, parent_dir_path)
    if evaluator_dir_path not in sys.path:
        sys.path.insert(0, evaluator_dir_path)

    from utils import evaluate_df_prefix_hit_cnt
    from solver import Algorithm

    dataset_name = os.path.basename(csv_path)

    try:
        if not os.path.exists(csv_path):
            return {
                'dataset_name': dataset_name,
                'hit_rate': 0.0,
                'runtime': 0.0,
                'success': False,
                'error': f"Dataset not found: {csv_path}"
            }

        print(f"[evaluator] [{idx}/{total}] Processing {dataset_name}...", file=sys.stderr, flush=True)

        # Load the evolved program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        program.pd = pd
        program.pandas = pd
        program.Algorithm = Algorithm
        spec.loader.exec_module(program)

        if not hasattr(program, 'Evolved'):
            return {
                'dataset_name': dataset_name,
                'hit_rate': 0.0,
                'runtime': 0.0,
                'success': False,
                'error': "No Evolved class in program"
            }

        EvolvedClass = program.Evolved

        # Load dataset (no pre-merging — col_merge is passed to reorder)
        master_df = pd.read_csv(csv_path, low_memory=False)

        # Run algorithm with original-style arguments
        st = time.time()
        reordered = EvolvedClass().reorder(
            master_df,
            early_stop=100000,
            distinct_value_threshold=0.7,
            row_stop=4,
            col_stop=2,
            col_merge=col_merge,
        )
        # Handle case where reorder returns a tuple
        if isinstance(reordered, tuple):
            reordered = reordered[0]
        runtime = time.time() - st

        # Evaluate prefix hit rate
        _, hit_rate_percent = evaluate_df_prefix_hit_cnt(reordered)
        hit_rate = hit_rate_percent / 100.0

        print(f"[evaluator] [{idx}/{total}] {dataset_name}: hit_rate={hit_rate:.4f}, runtime={runtime:.2f}s", file=sys.stderr, flush=True)

        # Release memory
        del master_df, reordered
        gc.collect()

        return {
            'dataset_name': dataset_name,
            'hit_rate': hit_rate,
            'runtime': runtime,
            'success': True,
            'error': ''
        }

    except Exception as e:
        print(f"[evaluator] [{idx}/{total}] {dataset_name} FAILED: {e}", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return {
            'dataset_name': dataset_name,
            'hit_rate': 0.0,
            'runtime': 0.0,
            'success': False,
            'error': str(e)
        }


def _write_dataset_metrics_csv(dataset_metrics, output_dir, aggregate_metrics=None):
    """Write per-dataset metrics to a CSV file."""
    csv_path = os.path.join(output_dir, "dataset_metrics.csv")
    try:
        fieldnames = ['dataset_name', 'hit_rate', 'runtime',
                      'final_score', 'avg_runtime', 'success', 'error']

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for metric in dataset_metrics:
                row = {k: metric.get(k, '') for k in fieldnames}
                writer.writerow(row)

            if aggregate_metrics:
                row = {k: aggregate_metrics.get(k, '') for k in fieldnames}
                writer.writerow(row)

        return csv_path
    except Exception as e:
        print(f"[evaluator] Warning: Failed to write dataset_metrics.csv: {str(e)}", file=sys.stderr, flush=True)
        return None


def evaluate(program_path: str):
    """
    Evaluate the Evolved class from the program file.
    Uses the original ADRS calling convention: col_merge and other parameters
    are passed directly to reorder() (no pre-merging by the evaluator).

    Args:
        program_path: Path to the Python file containing the Evolved class

    Returns:
        Dictionary with evaluation results
    """
    print(f"[evaluator] Starting evaluation of {program_path}", file=sys.stderr, flush=True)

    try:
        # Load the evolved program using importlib
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)

        # Inject required imports into the program namespace BEFORE execution
        program.pd = pd
        program.pandas = pd
        program.Algorithm = Algorithm

        # Add the evaluator directory to sys.path so imports like "from solver import Algorithm" work
        sys.path.insert(0, evaluator_dir)
        try:
            spec.loader.exec_module(program)
        finally:
            if evaluator_dir in sys.path:
                sys.path.remove(evaluator_dir)

        # Create temp directory for output files
        temp_dir = tempfile.mkdtemp(prefix="llm_sql_eval_")

        # Check if the required class exists
        if not hasattr(program, 'Evolved'):
            return {
                "combined_score": 0.0,
                "runs_successfully": 0.0,
                "error": "Code does not define an 'Evolved' class",
                "success": False,
                "sim_dir": temp_dir,
            }

        EvolvedClass = program.Evolved
        print("[evaluator] Evolved class loaded successfully", file=sys.stderr, flush=True)

        # Prepare arguments for parallel dataset evaluation
        args_list = [
            (csv_path, col_merge, program_path, parent_dir, evaluator_dir, idx, len(TEST_FILES))
            for idx, (csv_path, col_merge) in enumerate(zip(TEST_FILES, COL_MERGES), 1)
        ]

        # Use ProcessPoolExecutor for parallel evaluation
        max_workers = min(len(TEST_FILES), os.cpu_count() or 1)
        print(f"[evaluator] Evaluating {len(TEST_FILES)} datasets in parallel with {max_workers} workers...", file=sys.stderr, flush=True)

        dataset_metrics = [None] * len(TEST_FILES)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_process_dataset_worker, args): idx
                for idx, args in enumerate(args_list)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    dataset_metrics[idx] = result
                except Exception as e:
                    print(f"[evaluator] Worker failed for dataset {idx + 1}: {e}", file=sys.stderr, flush=True)
                    print(traceback.format_exc(), file=sys.stderr, flush=True)
                    dataset_metrics[idx] = {
                        'dataset_name': os.path.basename(TEST_FILES[idx]),
                        'hit_rate': 0.0,
                        'runtime': 0.0,
                        'success': False,
                        'error': str(e)
                    }

        # Aggregate results from parallel evaluation
        failed_files = sum(1 for m in dataset_metrics if not m.get('success', False))
        hit_rates = [m['hit_rate'] for m in dataset_metrics if m.get('success', False)]
        total_runtime = sum(m['runtime'] for m in dataset_metrics)

        if failed_files > 0:
            failed_error = next(
                (m['error'] for m in dataset_metrics if not m.get('success', True) and m.get('error')),
                '1 or more files failed to run'
            )
            aggregate_metrics = {
                'dataset_name': 'AGGREGATE',
                'hit_rate': 0.0,
                'runtime': total_runtime,
                'final_score': 0.0,
                'avg_runtime': 0.0,
                'success': False,
                'error': failed_error
            }
            _write_dataset_metrics_csv(dataset_metrics, temp_dir, aggregate_metrics)
            return {
                "combined_score": 0.0,
                "runs_successfully": 0.0,
                "error": failed_error,
                "success": False,
                "sim_dir": temp_dir,
            }

        if len(hit_rates) == 0:
            aggregate_metrics = {
                'dataset_name': 'AGGREGATE',
                'hit_rate': 0.0,
                'runtime': 0.0,
                'final_score': 0.0,
                'avg_runtime': 0.0,
                'success': False,
                'error': 'No datasets were successfully processed'
            }
            _write_dataset_metrics_csv(dataset_metrics, temp_dir, aggregate_metrics)
            return {
                "combined_score": 0.0,
                "runs_successfully": 0.0,
                "error": "No datasets were successfully processed",
                "success": False,
                "sim_dir": temp_dir,
            }

        average_hit_rate = sum(hit_rates) / len(TEST_FILES)
        average_runtime = total_runtime / len(TEST_FILES)

        # Original scoring: no baseline normalization, 12s runtime threshold
        runtime_score = (12.0 - min(12.0, average_runtime)) / 12.0
        final_score = 0.95 * average_hit_rate + 0.05 * runtime_score

        print(f"[evaluator] Evaluation completed successfully", file=sys.stderr, flush=True)
        print(f"[evaluator] Average hit rate: {average_hit_rate * 100:.2f}%", file=sys.stderr, flush=True)
        print(f"[evaluator] Final score: {final_score:.4f}", file=sys.stderr, flush=True)

        # Write aggregate metrics and CSV
        aggregate_metrics = {
            'dataset_name': 'AGGREGATE',
            'hit_rate': average_hit_rate,
            'runtime': total_runtime,
            'final_score': final_score,
            'avg_runtime': average_runtime,
            'success': True,
            'error': ''
        }
        _write_dataset_metrics_csv(dataset_metrics, temp_dir, aggregate_metrics)

        return {
            "combined_score": final_score,
            "runs_successfully": 1.0,
            "hit_rates": hit_rates,
            "total_runtime": total_runtime,
            "final_score": final_score,
            "avg_runtime": average_runtime,
            "avg_hit_rate": average_hit_rate * 100,
            "success": True,
            "sim_dir": temp_dir,
        }

    except Exception as e:
        print(f"[evaluator] Evaluation failed: {str(e)}", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        temp_dir = tempfile.mkdtemp(prefix="llm_sql_eval_")
        return {
            "combined_score": 0.0,
            "runs_successfully": 0.0,
            "error": str(e),
            "success": False,
            "sim_dir": temp_dir,
        }


if __name__ == "__main__":
    # Run evaluation on quick_greedy.py
    sample_file = os.path.join(evaluator_dir, "deepagents_files/GGR_ADRS/initial_program.py")
    if os.path.exists(sample_file):
        result = evaluate(sample_file)
        print(json.dumps(result, indent=2))
    else:
        print(f"Sample file not found: {sample_file}", file=sys.stderr, flush=True)
