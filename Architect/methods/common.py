"""Common base class and utilities for optimization methods."""

from abc import ABC, abstractmethod
from datetime import datetime
import json
import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal
from pathlib import Path

from Architect.task import Task
from Architect.llm.openai_architect import OpenAIArchitect, create_and_test_algorithm
from Architect.utils import make_directories, print_table

def _dummy_handler(signum, frame):
    """Dummy handler that just raises TimeoutError"""
    raise TimeoutError("simulation timed out")

def _convert_paths_to_strings(obj):
    """Recursively convert PosixPath objects to strings in nested data structures."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: _convert_paths_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_paths_to_strings(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_paths_to_strings(item) for item in obj)
    else:
        return obj

def initialize_worker():
    """Initialize worker process."""
    # Set up signal handlers for timeouts
    signal.signal(signal.SIGALRM, _dummy_handler)
    # Disable CUDA in worker processes to avoid conflicts
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

class OptimizationMethod(ABC):
    """Base class for optimization methods."""

    def __init__(
        self,
        task: Task,
        model: str,
        results_dir: str = "./results",
        debug: bool = True,
        run_baselines: bool = True,
        save_every_n_iterations: int = 5,
        resume_from: str = None,
        max_num_simulations: int = 150,  # Maximum number of simulations to run
        max_cost_dollars: float = 30.0,  # Maximum cost in dollars
        seeds_path: str = None,  # Path to seeds file
    ):
        """Initialize the optimization method.

        Args:
            task: The Task object
            model: The name of the model to use
            results_dir: Directory to save results
            debug: Whether to print debug information
            run_baselines: Whether to run baseline evaluations during initialization
            save_every_n_iterations: How often to save results
            resume_from: Path to resume from previous run
            max_num_simulations: Maximum number of simulations to run (default: 1000)
            max_cost_dollars: Maximum cost in dollars (default: $30)
            seeds_path: Path to JSON file containing seed algorithms (default: None)
        """
        self.task = task
        self.model = model
        self.debug = debug
        self.save_every_n_iterations = save_every_n_iterations
        self.resume_from = resume_from
        self.max_num_simulations = max_num_simulations
        self.max_cost_dollars = max_cost_dollars
        self.seeds_path = seeds_path

        # Initialize counters
        self.num_simulations = 0
        self.total_cost = 0.0
        if self.resume_from:
            self.total_cost, self.num_simulations = self.get_total_cost_and_num_simulations(self.resume_from)

        # Create results directories
        self.log_dir, self.plot_dir = make_directories(results_dir, task.name)

        # Get base architect and prompt
        self.architect = OpenAIArchitect(
            model=model,
            task=task
        )
        self.base_implement_prompt = self.architect._implement_prompt

        # Initialize baseline outputs
        self.baseline_outputs = {}
        if run_baselines and not self.resume_from:
            self.baseline_outputs = self.run_baselines()

        # save all the args to a file
        with open(f"{self.log_dir}/args.json", "w") as f:
            json.dump(vars(self), f, indent=2, default=str)

    def get_total_cost_and_num_simulations(self, resume_from: str) -> Tuple[float, int]:
        """Get total cost and number of simulations from output data."""
        total_cost = 0.0
        total_num_simulations = 0

        with open(resume_from, "r") as f:
            output_data = json.load(f)
        for iteration in output_data["all_iterations"]:
            if "usage_stats" not in iteration:
                print(iteration)
            total_cost += iteration.get("usage_stats", {}).get("total_cost", 0.0)
            sim_dirs = iteration.get("sim_dirs", [])
            total_num_simulations += len(sim_dirs) if sim_dirs else 1
        return total_cost, total_num_simulations

    def should_continue(self) -> Tuple[bool, str]:
        """Check if optimization should continue based on simulation count and cost limits.

        Returns:
            Tuple of (should_continue: bool, reason: str)
        """
        # if self.num_simulations >= self.max_num_simulations:
        #     return False, f"Reached {self.num_simulations} of maximum number of simulations ({self.max_num_simulations})"
        if self.total_cost >= self.max_cost_dollars:
            return False, f"Reached cost {self.total_cost:.2f} of maximum cost limit (${self.max_cost_dollars:.2f})"
        return True, ""

    def update_tracking(self, usage_stats: Dict[str, Any], no_simulations: bool = False):
        """Update simulation count and cost tracking.

        Args:
            usage_stats: Dictionary containing usage statistics from the architect
        """
        if not no_simulations:
            self.num_simulations += 1
        self.total_cost += usage_stats.get("total_cost", 0.0)
        if self.debug:
            print(f"Model: {self.model}")
            print(f"\nSimulation {self.num_simulations}/{self.max_num_simulations}")
            print(f"Total cost: ${self.total_cost:.2f}/${self.max_cost_dollars:.2f}")
            print(f"Cost of this iteration: ${usage_stats.get('total_cost', 0.0)}")

    def _baseline_cache_path(self) -> str:
        """Return the path to the baseline cache file.

        Uses evaluator's get_baseline_cache_dir() if available (shared across runs),
        otherwise falls back to the run-specific results directory.
        """
        evaluator = self.task.evaluator
        if hasattr(evaluator, "get_baseline_cache_dir") and callable(evaluator.get_baseline_cache_dir):
            return os.path.join(evaluator.get_baseline_cache_dir(), "baseline_cache.json")
        return os.path.join(os.path.dirname(self.log_dir), "baseline_cache.json")

    def run_baselines(self) -> Dict[str, Dict[str, Any]]:
        """Run baseline evaluations if available, using a cache file to avoid re-evaluation."""
        # Check for cached results first
        cache_path = self._baseline_cache_path()
        if os.path.exists(cache_path):
            print(f"\nLoading cached baseline results from {cache_path}")
            with open(cache_path, "r") as f:
                return json.load(f)

        baselines = {}

        if hasattr(self.task.evaluator, "get_baselines") and callable(self.task.evaluator.get_baselines):
            baseline_implementations = self.task.evaluator.get_baselines()
            for i, (baseline_label, baseline_implementation) in enumerate(baseline_implementations):
                sim_dirs = []
                results = {}
                try:
                    # Set the baseline implementation based on whether it's a class or function
                    self.task.evaluator.set_code(self.task.evaluator.target_name, baseline_implementation)

                    # Evaluate it
                    score, sim_dirs, results = self.task.evaluate()

                    # Store results
                    baselines[f"baseline_v{i}"] = {
                        "success": True,
                        "score": score,
                        "code": baseline_implementation,
                        "reasoning": baseline_label,
                        "sim_dirs": sim_dirs,
                        "results": results
                    }

                    if self.debug:
                        print(f"\nBaseline {i} ({baseline_label}):")
                        print(f"Score: {score}")
                except Exception as e:
                    if self.debug:
                        print(f"\nBaseline {i} ({baseline_label}) failed:")
                        print(f"Error: {str(e)}")
                    baselines[f"baseline_v{i}"] = {
                        "success": False,
                        "score": 0.0,
                        "code": baseline_implementation,
                        "reasoning": baseline_label,
                        "error": str(e),
                        "sim_dirs": sim_dirs,
                        "results": results
                    }

        # Cache results for future runs
        if baselines:
            print(f"\nCaching baseline results to {cache_path}")
            baselines_serializable = _convert_paths_to_strings(baselines)
            with open(cache_path, "w") as f:
                json.dump(baselines_serializable, f, indent=2)

        return baselines

    def summarize_results(self, results: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if the results are successful."""
        if "success" in results:
            return results["success"], results.get("error", "No error message available")
        elif isinstance(results, dict):
            # Aggregate errors from all scenarios
            error_message = ""
            success = True
            for scenario_name, res in results.items():
                success = success and res.get("success", False)
                error_message = error_message + f"- {scenario_name}: {res.get('error', 'No error message available')}\n"
            return success, error_message
        else:
            raise ValueError(f"Invalid results type: {type(results)}")

    def evaluate_code(self, code: str) -> Tuple[float, List[str]]:
        """Run simulation using the evaluator."""
        self.task.evaluator.set_code(self.task.evaluator.target_name, code)
        score, sim_dirs, results = self.task.evaluate()
        success, error_message = self.summarize_results(results)

        if "success" not in results:
            results["success"] = success
            results["error"] = error_message

        return score, sim_dirs, results

    def save_results(
        self,
        output_data: Dict[str, Any],
        method_name: str,
        extra_info: str = "",
    ) -> Tuple[str, str, str]:
        """Save optimization results to file.

        Args:
            output_data: Results data to save
            method_name: Name of the optimization method
            extra_info: Additional info for filename

        Returns:
            Tuple of (history_file_path, plot_path, timestamp)
        """
        timestamp = "" #datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = f"{self.log_dir}/{self.model}-{self.task.name}-{method_name}{extra_info}.json"
        plot_path = f"{self.plot_dir}/{self.model}-{self.task.name}-{method_name}{extra_info}"

        # Aggregate usage stats from all iterations
        total_cost = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        # Check if we have iterations to process
        if "all_iterations" in output_data:
            for iteration in output_data["all_iterations"]:
                if "usage_stats" in iteration:
                    stats = iteration["usage_stats"]
                    total_cost += stats["total_cost"]
                    total_prompt_tokens += stats["total_prompt_tokens"]
                    total_completion_tokens += stats["total_completion_tokens"]
                    total_tokens += stats["total_tokens"]

        # Methods like agentic_handoff track usage at top-level (no per-iteration usage)
        # Add top-level usage_stats when present so cost is accumulated correctly
        if output_data.get("usage_stats"):
            top = output_data["usage_stats"]
            total_cost += top.get("total_cost", 0) or 0
            total_prompt_tokens += top.get("total_prompt_tokens", 0) or 0
            total_completion_tokens += top.get("total_completion_tokens", 0) or 0
            total_tokens += top.get("total_tokens", 0) or 0

        # Create aggregated usage stats
        usage_stats = {
            "total_cost": round(total_cost, 6),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "model": self.model
        }

        usage_stats_file = f"{self.log_dir}/{self.model}-{self.task.name}-{method_name}{extra_info}_usage_stats.json"

        print(f"\nFinal Usage Statistics:")
        print(f"Total tokens used: {usage_stats['total_tokens']}")
        print(f"Total cost: ${usage_stats['total_cost']:.6f}")

        with open(usage_stats_file, "w") as f:
            json.dump(usage_stats, f, indent=2)

        # Only include baselines if they were already computed
        if "baselines" not in output_data:
            if self.baseline_outputs:
                output_data["baselines"] = self.baseline_outputs

        # Convert any PosixPath objects to strings before JSON serialization
        output_data_serializable = _convert_paths_to_strings(output_data)
        
        with open(history_file, "w") as f:
            json.dump(output_data_serializable, f, indent=2)

        if self.debug:
            print(f"\nResults saved to {history_file}")

        # Also plot the results
        try:
            self.plot_results(plot_path, output_data, f"{self.model}-{self.task.name}-{method_name}{extra_info}", "Sample index", "Score", "stairs")
        except Exception as e:
            print(f"Warning: Error plotting results: {e}")
        return history_file, plot_path, timestamp

    def get_growth_rate(self, score_generations: List[float]):
        # Calculate growth rate metrics
        score_generations_array = np.array(score_generations)
        max_score = score_generations_array[-1]
        min_score = score_generations_array[0]
        n_samples = len(score_generations)

        # Calculate improvement steps
        improvements = np.diff(score_generations_array)
        improvement_indices = np.where(improvements > 0)[0] + 1
        n_improvements = len(improvement_indices)

        if n_improvements > 0:
            # Calculate average improvement metrics
            avg_samples_per_improvement = n_samples / n_improvements
            avg_improvement_size = np.mean(improvements[improvements > 0])

            # Calculate growth rate as weighted sum of (improvement_size / samples_until_improvement)
            improvement_sizes = improvements[improvements > 0]
            samples_until = improvement_indices
            growth_rates = improvement_sizes / samples_until
            growth_rate = np.mean(growth_rates)
        else:
            avg_samples_per_improvement = float('inf')
            avg_improvement_size = 0
            growth_rate = 0

        print("\nGrowth Rate Metrics:")
        print(f"Growth rate (improvement/sample): {growth_rate:.3f}")
        print(f"Number of improvements: {n_improvements}")
        print(f"Average samples between improvements: {avg_samples_per_improvement:.1f}")
        print(f"Average improvement size: {avg_improvement_size:.3f}")

    def plot_results(
        self,
        save_path: str,
        output_dict: Dict[str, Any],
        title: str,
        x_label: str,
        y_label: str,
        plot_type: str = "stairs",
    ):
        """Plot optimization results."""
        results_dict = output_dict
        if "baselines" not in results_dict:
            if self.baseline_outputs:
                results_dict["baselines"] = self.baseline_outputs
        plt.style.use("seaborn-v0_8-paper")
        plt.rcParams.update(
            {
                "text.usetex": False,
                "legend.fontsize": 6,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "font.family": "sans-serif",
                "font.serif": "Computer Modern Roman",
                "axes.labelsize": 10,
                "axes.titlesize": 10,
                "figure.labelsize": 10,
                "figure.titlesize": 10,
                "hatch.linewidth": 0.5,
            }
        )

        # Debug information
        print("\nDebugging results_dict:")
        print("Keys in results_dict:", list(results_dict.keys()))

        # Print baseline scores table
        table = []
        if "baselines" in results_dict:
            for key, data in results_dict["baselines"].items():
                score = data.get("score")
                if score is not None:
                    try:
                        score_float = float(score)
                        if np.isfinite(score_float):
                            lbl = data.get("reasoning", "N/A")
                            score_str = f"{score_float:.2f}"
                            table.append([key, lbl, score_str])
                        else:
                            print(f"Warning: Skipping non-finite baseline score {score} for {key}")
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Invalid baseline score {score} for {key}: {e}")

        if table:
            table.sort(key=lambda row: float(row[2]), reverse=True)
            print_table(
                table, ["Name", "Explanation", "Score"], show_lines=True, header_style="bold", title="Baseline Scores"
            )
            print(f"Best baseline: {table[0][1]} (score: {table[0][2]})")
        else:
            print("\nNo valid baseline scores to display in table.")

        if plot_type == "one-shot":
            # Get sample scores with validation
            sample_scores = []
            for key in results_dict:
                if "_sample" in key:
                    score = results_dict[key].get("score")
                    if score is not None:
                        try:
                            score_float = float(score)
                            if np.isfinite(score_float):  # Only include finite values
                                sample_scores.append(score_float)
                            else:
                                print(f"Warning: Non-finite score {score} for {key}")
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Invalid score {score} for {key}: {e}")

            if not sample_scores:
                raise ValueError("No valid sample scores found in results. Cannot create histogram.")

            one_shot_times = np.array(sample_scores)
            print(f"Number of valid samples: {len(one_shot_times)}")
            print("Sample scores:", one_shot_times)

            # Get baseline scores for range calculation
            baseline_scores = []
            if "baselines" in results_dict:
                for key, data in results_dict["baselines"].items():
                    score = data.get("score")
                    if score is not None:
                        try:
                            score_float = float(score)
                            if np.isfinite(score_float):  # Only include finite values
                                baseline_scores.append(score_float)
                            else:
                                print(f"Warning: Non-finite baseline score {score} for {key}")
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Invalid baseline score {score} for {key}: {e}")

            # Calculate range including both samples and baselines
            all_scores = np.concatenate([one_shot_times, baseline_scores]) if baseline_scores else one_shot_times
            if len(all_scores) == 0:
                raise ValueError("No valid scores found (all scores are infinite or NaN)")

            # Set range to min/max of finite values
            range_data = (np.min(all_scores), np.max(all_scores))
            print(f"Score range: {range_data}")

            # Create first plot: Histogram
            fig1, ax1 = plt.subplots(1, 1, figsize=(3.25, 2))
            counts, bins = np.histogram(one_shot_times, bins=50, range=range_data)
            ax1.hist(bins[:-1], bins, weights=counts / counts.sum() * 100, alpha=0.5, edgecolor="black", linewidth=0.5, label='Sample distribution')

            # Plot baselines on histogram
            if "baselines" in results_dict:
                for key, data in results_dict["baselines"].items():
                    x = data.get("score")
                    if x is not None:
                        try:
                            x_float = float(x)
                            if np.isfinite(x_float):
                                ax1.axvline(
                                    x=x_float,
                                    color="red",
                                    linestyle="--",
                                    alpha=0.8,
                                )
                            else:
                                print(f"Warning: Skipping non-finite baseline value {x} for {key}")
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Invalid baseline value {x} for {key}: {e}")

            ax1.set_xlabel(x_label)
            ax1.set_ylabel("Percentage")
            ax1.set_title(title)
            ax1.spines["right"].set_visible(False)
            ax1.spines["top"].set_visible(False)
            ax1.plot(1, 0, ">k", transform=ax1.transAxes, clip_on=False)
            ax1.plot(0, 1, "^k", transform=ax1.transAxes, clip_on=False)
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{save_path}_histogram.pdf", dpi=300, bbox_inches="tight")
            plt.savefig(f"{save_path}_histogram.png", dpi=300, bbox_inches="tight")
            plt.close()

            # Create second plot: Stairs
            fig2, ax2 = plt.subplots(1, 1, figsize=(3.25, 2))

            # Calculate score progression (using original order)
            score_generations = [one_shot_times[0]]  # Start with first score
            current_best = one_shot_times[0]
            for score in one_shot_times[1:]:
                current_best = max(current_best, score)
                score_generations.append(current_best)

            # Create stairs plot for max envelope (best scores so far)
            ax2.stairs(
                score_generations,
                np.arange(len(score_generations) + 1),
                edgecolor="C2",
                linewidth=1,
                baseline=None,
                label='Best score'
            )

            # Add plot for raw scores (in original order)
            ax2.stairs(
                one_shot_times,
                np.arange(len(one_shot_times) + 1),
                edgecolor="C0",
                linewidth=1,
                linestyle='--',
                baseline=None,
                label='Raw scores'
            )

            # Add legend
            ax2.legend()

            # Plot baselines on stairs plot
            if "baselines" in results_dict:
                for key, data in results_dict["baselines"].items():
                    y = data.get("score")
                    if y is not None:
                        try:
                            y_float = float(y)
                            if np.isfinite(y_float):
                                ax2.axhline(
                                    y=y_float,
                                    color="red",
                                    linestyle="--",
                                    alpha=0.8,
                                )
                            else:
                                print(f"Warning: Skipping non-finite baseline value {y} for {key}")
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Invalid baseline value {y} for {key}: {e}")

            ax2.set_xlabel("Simulation Number")
            ax2.set_ylabel(y_label)
            ax2.set_title(title)
            ax2.spines["right"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax2.plot(1, 0, ">k", transform=ax2.transAxes, clip_on=False)
            ax2.plot(0, 1, "^k", transform=ax2.transAxes, clip_on=False)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{save_path}_stairs.pdf", dpi=300, bbox_inches="tight")
            plt.savefig(f"{save_path}_stairs.png", dpi=300, bbox_inches="tight")
            plt.close()

        elif plot_type == "stairs":
            # Get generation results with validation
            if "all_iterations" not in results_dict:
                print("Warning: No 'all_iterations' found in results")
                print("Available keys:", list(results_dict.keys()))
                raise ValueError("Missing 'all_iterations' in results dictionary")

            gen_res = results_dict["all_iterations"]
            if not gen_res:
                print("Warning: Empty 'all_iterations' list")
                raise ValueError("No iterations found in results")

            # Validate scores in iterations
            valid_scores = []
            for r in gen_res:
                score = r.get("score")
                if score is not None:
                    try:
                        score_float = float(score)
                        if np.isfinite(score_float):
                            valid_scores.append(score_float)
                        else:
                            print(f"Warning: Skipping non-finite score {score}")
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Invalid score {score}: {e}")

            if not valid_scores:
                raise ValueError("No valid scores found in iterations")

            # Calculate score progression
            score_generations = [valid_scores[0]]  # Start with first score
            current_best = valid_scores[0]
            for score in valid_scores[1:]:
                current_best = max(current_best, score)
                score_generations.append(current_best)

            print(f"Number of generations: {len(score_generations)}")
            print("Score progression:", score_generations)

            # Calculate growth rate metrics
            self.get_growth_rate(score_generations)

            # Create figure and axis
            fig, ax = plt.subplots(1, 1, figsize=(3.25, 2))

            # Create stairs plot for max envelope (best scores so far)
            ax.stairs(
                score_generations, 
                np.arange(len(score_generations) + 1), 
                edgecolor="C2", 
                linewidth=1,
                baseline=None,
                label='Best score'
            )

            # Add plot for raw scores
            ax.stairs(
                valid_scores,
                np.arange(len(valid_scores) + 1),
                edgecolor="C0",
                linewidth=1,
                linestyle='--',
                baseline=None,
                label='Raw scores'
            )

            # Add legend
            ax.legend()

            # Plot baselines
            if "baselines" in results_dict:
                for key, data in results_dict["baselines"].items():
                    y = data.get("score")
                    if y is not None:
                        try:
                            y_float = float(y)
                            if np.isfinite(y_float):
                                ax.axhline(
                                    y=y_float,
                                    color="red",
                                    linestyle="--",
                                    alpha=0.8,
                                )
                            else:
                                print(f"Warning: Skipping non-finite baseline value {y} for {key}")
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Invalid baseline value {y} for {key}: {e}")

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)

            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
            ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight")
            plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
            plt.close()

    @abstractmethod
    def optimize(self) -> Dict[str, Any]:
        """Run optimization and return results."""
        pass
