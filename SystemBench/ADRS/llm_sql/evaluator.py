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

TARGET_NAME = "Evolved"

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


def _apply_column_merges(df: pd.DataFrame, merge_spec: list) -> pd.DataFrame:
    """Apply column merges as preprocessing (functional dependencies).

    Uses Algorithm.calculate_col_stats to determine the optimal merge order
    within each group (same approach as quick_greedy), then delegates to
    Algorithm.merging_columns for the actual merge.
    """
    if not merge_spec:
        return df
    _, column_stats = Algorithm.calculate_col_stats(df)
    ranked_columns = [col for col, _, _, _ in column_stats]
    for col_to_merge in merge_spec:
        if all(col in df.columns for col in col_to_merge):
            ordered = [col for col in ranked_columns if col in col_to_merge]
            df = Algorithm.merging_columns(df, ordered, prepended=False)
    return df


def _verify_valid_permutation(input_df: pd.DataFrame, evolved_class: Algorithm) -> tuple:
    """Verify output is a valid row+per-row-value permutation of input.

    Each output row must be a permutation of exactly one input row's values
    (multiset equality). Uses type-normalized comparison to tolerate Python's
    type conflation (True==1, 0==0.0, etc.) without requiring pre-stringification.
    """
    from collections import Counter
    new_input_df = input_df.fillna("").astype(str)
    input_df_for_validation = new_input_df.copy()
    output_df = evolved_class().reorder(input_df_for_validation)
    output_df = output_df[0] if isinstance(output_df, tuple) else output_df
    output_df = output_df.fillna("").astype(str) # TODO

    if new_input_df.shape != output_df.shape:
        return False, f"Shape mismatch: input {new_input_df.shape} vs output {output_df.shape}"

    input_sigs = [tuple(sorted(row)) for row in new_input_df.values]
    output_sigs = [tuple(sorted(row)) for row in output_df.values]

    input_counts = Counter(input_sigs)
    output_counts = Counter(output_sigs)

    # Every output signature must match an input signature exactly
    for sig, count in output_counts.items():
        if input_counts[sig] < count:
            excess = count - input_counts[sig]
            sample_vals = list(sig)[:5]
            return False, (
                f"{excess} output row(s) have values {sample_vals}... that don't match "
                f"any remaining input row. Values were fabricated or swapped between rows."
            )

    # Every input signature must be fully consumed
    for sig, count in input_counts.items():
        if output_counts[sig] < count:
            missing = count - output_counts[sig]
            return False, f"{missing} input row(s) were not matched by any output row"

    return True, "Valid permutation"


def _process_dataset_worker(args_tuple):
    """Worker function for parallel dataset evaluation - must be at module level for pickling."""
    csv_path, merge_spec, program_path, parent_dir_path, evaluator_dir_path, idx, total = args_tuple

    # Ensure paths are in sys.path for imports
    if parent_dir_path not in sys.path:
        sys.path.insert(0, parent_dir_path)
    if evaluator_dir_path not in sys.path:
        sys.path.insert(0, evaluator_dir_path)

    from utils import evaluate_df_prefix_hit_cnt
    from solver import Algorithm
    import numpy as np

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

        # Load dataset and pre-merge columns
        master_df = pd.read_csv(csv_path, low_memory=False)
        if merge_spec:
            master_df = _apply_column_merges(master_df, merge_spec)

        # Run algorithm on already-merged data
        input_df_for_validation = master_df.copy()
        st = time.time()
        result = EvolvedClass().reorder(master_df)
        reordered = result[0] if isinstance(result, tuple) else result
        runtime = time.time() - st

        # Validate output
        is_valid, validation_msg = _verify_valid_permutation(input_df_for_validation, EvolvedClass)
        if not is_valid:
            print(f"[evaluator] [{idx}/{total}] {dataset_name} VALIDATION FAILED: {validation_msg}", file=sys.stderr, flush=True)
            return {
                'dataset_name': dataset_name,
                'hit_rate': 0.0,
                'runtime': runtime,
                'success': False,
                'error': validation_msg
            }

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
    Uses the original scoring mechanism (no baseline normalization, 12s runtime threshold).

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
            (csv_path, merge_spec, program_path, parent_dir, evaluator_dir, idx, len(TEST_FILES))
            for idx, (csv_path, merge_spec) in enumerate(zip(TEST_FILES, COL_MERGES), 1)
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
    if len(sys.argv) > 1:
        sample_file = os.path.abspath(sys.argv[1])
    else:
        sample_file = os.path.join(evaluator_dir, "quick_greedy.py")
    if os.path.exists(sample_file):
        result = evaluate(sample_file)
        print(json.dumps(result, indent=2))
    else:
        print(f"Sample file not found: {sample_file}", file=sys.stderr, flush=True)
