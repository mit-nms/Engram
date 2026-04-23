"""Main entry point for running optimization with different methods."""

import click
from typing import Dict, Any, Optional
import os
import glob
import json
import re
from .task import load_task_from_paths
from .methods.one_shot import OneShotOptimizer
from .methods.best_shot import BestShotOptimizer
from .methods.evolution import EvolutionOptimizer
from .methods.openevolve_optimizer import OpenEvolveOptimizer
from .methods.deepagents_tree import DeepAgentsTreeOptimizer

OPTIMIZERS = {
    "one-shot": OneShotOptimizer,
    "best-shot": BestShotOptimizer,
    "evolution": EvolutionOptimizer,
    "openevolve": OpenEvolveOptimizer,
    "deepagents_tree": DeepAgentsTreeOptimizer,
}


def get_optimizer_params(
    method: str, task: Any, model: str, results_dir: str, debug: bool, just_aggregate: bool, **kwargs
) -> Dict[str, Any]:
    """Create optimizer parameters based on method type."""
    base_params = {
        "task": task,
        "model": model,
        "results_dir": results_dir,
        "debug": debug,
        "run_baselines": not just_aggregate and kwargs.get("resume_from") == None,  # Don't run baselines when just aggregating
        "resume_from": kwargs.get("resume_from"),
        "seeds_path": kwargs.get("seeds_path"),
    }

    method_specific_params = {
        "one-shot": {
            "rounds": kwargs.get("rounds"),
        },
        "best-shot": {
            "num_generations": kwargs.get("num_generations"),
            "best_shot_count": kwargs.get("best_shot_count"),
            "island_max_count": kwargs.get("island_max_count"),
        },
        "evolution": {
            "population_size": kwargs.get("population_size"),
            "tournament_size": kwargs.get("tournament_size"),
            "mutation_rate": kwargs.get("mutation_rate"),
            "elitism_count": kwargs.get("elitism_count"),
            "num_generations": kwargs.get("num_generations"),
            "random_selection": kwargs.get("random_selection"),
            "seed_protection_generations": kwargs.get("seed_protection_generations"),
            "disable_diversify": kwargs.get("disable_diversify"),
        },
        "openevolve": {
            "openevolve_config_path": kwargs.get("openevolve_config_path"),
            "initial_program_path": kwargs.get("initial_program_path"),
            "evaluator_path": kwargs.get("evaluator_path"),
        },
        "deepagents_tree": {
            "tree_rounds": kwargs.get("tree_rounds"),
            "initial_program_path": kwargs.get("tree_initial_program_path"),
            "task_prompt_path": kwargs.get("tree_task_prompt_path"),
            "system_prompt_path": kwargs.get("tree_system_prompt_path"),
            "max_review_iterations": kwargs.get("tree_max_review_iterations"),
            "early_stop_patience": kwargs.get("tree_early_stop_patience"),
            "capture_simulation_output": kwargs.get("tree_capture_simulation_output"),
            "lambda_age": kwargs.get("tree_lambda_age"),
            "mu_child_improvement": kwargs.get("tree_mu_child_improvement"),
            "nu_times_selected": kwargs.get("tree_nu_times_selected"),
            "save_every_n_rounds": kwargs.get("tree_save_every_n_rounds"),
        },
    }

    base_params.update(method_specific_params[method])
    return base_params


def get_results_pattern(method: str, log_dir: str, model: str, task_name: str) -> str:
    """Generate the pattern for finding result files."""
    patterns = {
        "one-shot": f"{log_dir}/{model}-{task_name}-one-shot.json",
        "best-shot": f"{log_dir}/{model}-{task_name}-best-shot_*gen.json",
        "evolution": f"{log_dir}/{model}-{task_name}-evolution_*gen.json",
        "openevolve": f"{log_dir}/*.json",  # OpenEvolve may have different file naming
        "deepagents_tree": f"{log_dir}/{model}-{task_name}-deepagents_tree_*rounds.json",
    }
    if method not in patterns:
        raise ValueError(f"Unknown method: {method}")
    return patterns[method]


def get_plot_config(method: str, results_file: str) -> Dict[str, str]:
    """Get plotting configuration based on method."""
    if method == "one-shot":
        match = re.search(r'_([0-9]+)rounds\.json$', results_file)
        if match:
            num_rounds = int(match.group(1))
        else:
            raise ValueError(f"Could not find number of rounds in {results_file}")
        return {
            "extra_info": f"_{num_rounds}rounds",
            "plot_type": "one-shot",
            "x_label": "Reward",
            "y_label": "Frequency (%)",
        }
    else:
        # find where gen is, then get the number before it
        match = re.search(r'_([0-9]+)gen\.json$', results_file)
        if match:
            num_generations = int(match.group(1))
        else:
            raise ValueError(f"Could not find number of generations in {results_file}")
        return {
            "extra_info": f"_{num_generations}gen",
            "plot_type": "stairs",
            "x_label": "Generation Number",
            "y_label": "Score",
        }


def setup_directories(results_dir: str, task_name: str) -> tuple[str, str]:
    """Create and return log and plot directories."""
    log_dir = os.path.join(results_dir, task_name, "logs")
    plot_dir = os.path.join(results_dir, task_name, "plots")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    return log_dir, plot_dir


@click.command()
@click.option(
    "--method",
    required=True,
    type=click.Choice(list(OPTIMIZERS.keys()), case_sensitive=False),
    help="Which optimization method to use",
)
@click.option(
    "--model",
    required=False,
    type=click.Choice(["gpt-4o", "o1", "o3-mini", "o4-mini", "gpt-4.1", "o3", "gpt-5.1", "gpt-5", "gpt-5.2", "gpt-5.2-high", "gpt-5.2-xhigh"], case_sensitive=False),
    help="Which LLM model to use (not required for openevolve)",
)
@click.option(
    "--task_prompt_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the task prompt text file",
)
@click.option(
    "--evaluator_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the evaluator .py file or directory containing evaluator.py / evaluate.py",
)
@click.option(
    "--task_name",
    default=None,
    type=str,
    help="Task name (defaults to evaluator directory basename)",
)
@click.option(
    "--results_dir",
    default="./results",
    type=click.Path(dir_okay=True, file_okay=False),
    help="Directory to save results",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug output",
)
@click.option(
    "--just_aggregate",
    is_flag=True,
    help="Just aggregate the results, don't run the optimization",
)
# One-shot specific options
@click.option(
    "--rounds",
    default=100,
    type=int,
    help="Number of optimization rounds (for one-shot)",
)
# Best-shot specific options
@click.option(
    "--num_generations",
    default=100,
    type=int,
    help="Number of generations (for best-shot and evolution)",
)
@click.option(
    "--best_shot_count",
    default=3,
    type=int,
    help="Number of best examples to include (for best-shot)",
)
@click.option(
    "--island_max_count",
    default=20,
    type=int,
    help="Maximum population size (for best-shot)",
)
# Evolution specific options
@click.option(
    "--population_size",
    default=20,
    type=int,
    help="Size of the population (for evolution)",
)
@click.option(
    "--tournament_size",
    default=3,
    type=int,
    help="Tournament size for selection (for evolution)",
)
@click.option(
    "--mutation_rate",
    default=0.1,
    type=float,
    help="Mutation rate (for evolution)",
)
@click.option(
    "--elitism_count",
    default=2,
    type=int,
    help="Number of elite individuals to preserve (for evolution)",
)
@click.option(
    "--random_selection",
    is_flag=True,
    help="Use random selection instead of tournament selection (for evolution)",
)
@click.option(
    "--seed_protection_generations",
    default=0,
    type=int,
    help="Number of generations to protect seeds (for evolution)",
)
@click.option(
    "--disable_diversify",
    is_flag=True,
    help="Disable the diversify operator (for evolution)",
)
@click.option(
    "--resume_from",
    type=click.Path(exists=True),
    help="Path to resume from. For most methods, a previous log JSON file. For openevolve, a checkpoint directory (e.g., openevolve_output/checkpoints/checkpoint_50).",
)
@click.option(
    "--seeds_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the seeds file",
    default=None,
)
@click.option(
    "--initial_program_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to initial program file (used by openevolve)",
)
# OpenEvolve specific options
@click.option(
    "--openevolve_config_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to OpenEvolve config YAML file (for openevolve)",
)
# DeepAgents Tree specific options
@click.option(
    "--tree_rounds",
    default=10,
    type=int,
    help="Number of tree expansion rounds (for deepagents_tree)",
)
@click.option(
    "--tree_initial_program_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to initial program file for tree root (for deepagents_tree)",
)
@click.option(
    "--tree_task_prompt_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to user prompt file (for deepagents_tree)",
)
@click.option(
    "--tree_system_prompt_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to system prompt file (for deepagents_tree)",
)
@click.option(
    "--tree_max_review_iterations",
    default=40,
    type=int,
    help="Max iterations per AgenticDeepAgents run (for deepagents_tree)",
)
@click.option(
    "--tree_early_stop_patience",
    default=10,
    type=int,
    help="Stop AgenticDeepAgents if no improvement for this many iterations (for deepagents_tree)",
)
@click.option(
    "--tree_capture_simulation_output",
    is_flag=True,
    help="Capture simulation output (for deepagents_tree)",
)
@click.option(
    "--tree_lambda_age",
    default=0.01,
    type=float,
    help="Age penalty weight in selection score (for deepagents_tree)",
)
@click.option(
    "--tree_mu_child_improvement",
    default=0.1,
    type=float,
    help="Child improvement bonus weight in selection score (for deepagents_tree)",
)
@click.option(
    "--tree_nu_times_selected",
    default=0.05,
    type=float,
    help="Times-selected penalty weight in selection score (for deepagents_tree)",
)
@click.option(
    "--tree_save_every_n_rounds",
    default=1,
    type=int,
    help="How often to save intermediate results (for deepagents_tree)",
)

def main(
    method: str,
    model: Optional[str],
    task_prompt_path: str,
    evaluator_path: str,
    task_name: Optional[str],
    results_dir: str,
    debug: bool,
    just_aggregate: bool,
    # Method-specific parameters
    rounds: int,
    num_generations: int,
    best_shot_count: int,
    island_max_count: int,
    population_size: int,
    tournament_size: int,
    mutation_rate: float,
    elitism_count: int,
    random_selection: bool,
    seed_protection_generations: int,
    disable_diversify: bool,
    resume_from: str,
    seeds_path: str,
    initial_program_path: Optional[str],
    # OpenEvolve parameters
    openevolve_config_path: Optional[str],
    # DeepAgents Tree parameters
    tree_rounds: int,
    tree_initial_program_path: Optional[str],
    tree_task_prompt_path: Optional[str],
    tree_system_prompt_path: Optional[str],
    tree_max_review_iterations: int,
    tree_early_stop_patience: int,
    tree_capture_simulation_output: bool,
    tree_lambda_age: float,
    tree_mu_child_improvement: float,
    tree_nu_times_selected: float,
    tree_save_every_n_rounds: int,
):
    """Run optimization with the specified method."""
    task = None
    try:
        # Load the task
        if debug:
            print(f"Loading task from prompt={task_prompt_path}, evaluator={evaluator_path}")
        task = load_task_from_paths(task_prompt_path, evaluator_path, task_name)

        # Setup directories
        log_dir, plot_dir = setup_directories(results_dir, task.name)

        # Check if log directory exists or empty for aggregation
        if not os.path.exists(log_dir) or len(os.listdir(log_dir)) == 0:
            print(f"Results directory {log_dir} does not exist or is empty. Setting just_aggregate to False.")
            just_aggregate = False

        # Get optimizer parameters
        optimizer_params = get_optimizer_params(
            method=method,
            task=task,
            model=model,
            results_dir=results_dir,
            debug=debug,
            just_aggregate=just_aggregate,
            rounds=rounds,
            num_generations=num_generations,
            best_shot_count=best_shot_count,
            island_max_count=island_max_count,
            population_size=population_size,
            tournament_size=tournament_size,
            mutation_rate=mutation_rate,
            elitism_count=elitism_count,
            random_selection=random_selection,
            seed_protection_generations=seed_protection_generations,
            disable_diversify=disable_diversify,
            resume_from=resume_from,
            seeds_path=seeds_path,
            evaluator_path=evaluator_path,
            initial_program_path=initial_program_path,
            # OpenEvolve parameters
            openevolve_config_path=openevolve_config_path,
            # DeepAgents Tree parameters
            tree_rounds=tree_rounds,
            tree_initial_program_path=tree_initial_program_path,
            tree_task_prompt_path=tree_task_prompt_path,
            tree_system_prompt_path=tree_system_prompt_path,
            tree_max_review_iterations=tree_max_review_iterations,
            tree_early_stop_patience=tree_early_stop_patience,
            tree_capture_simulation_output=tree_capture_simulation_output,
            tree_lambda_age=tree_lambda_age,
            tree_mu_child_improvement=tree_mu_child_improvement,
            tree_nu_times_selected=tree_nu_times_selected,
            tree_save_every_n_rounds=tree_save_every_n_rounds,
        )

        # Create optimizer instance
        optimizer = OPTIMIZERS[method](**optimizer_params)

        if just_aggregate:
            # Find and load results
            pattern = get_results_pattern(method, log_dir, model, task.name)
            if debug:
                print(f"Looking for results files matching pattern: {pattern}")

            matching_files = glob.glob(pattern)
            if not matching_files:
                print(f"No results files found matching pattern: {pattern}")
                print(f"Contents of {log_dir}:")
                all_files = os.listdir(log_dir) if os.path.exists(log_dir) else []
                for f in all_files:
                    print(f"  {f}")
                raise ValueError(f"No results files found for {method} method")

            # Load most recent results
            results_file = max(matching_files, key=os.path.getctime)
            if debug:
                print(f"Loading results from {results_file}")

            with open(results_file, "r") as f:
                results = json.load(f)

            # Get plot configuration
            plot_config = get_plot_config(method, results_file)

            # Save and plot results
            history_file, plot_path, _ = optimizer.save_results(
                output_data=results, method_name=method, extra_info=plot_config["extra_info"]
            )

            optimizer.plot_results(
                save_path=plot_path,
                output_dict=results,
                title=f"{method.title()} Optimization Results",
                x_label=plot_config["x_label"],
                y_label=plot_config["y_label"],
                plot_type=plot_config["plot_type"],
            )
        else:
            # Run optimization
            results = optimizer.optimize()

            # Print final results
            print("\nOptimization completed!")
            print(f"Best score achieved: {results['best_solution']['score']:.2f}")
            print("\nBest solution found:")
            print(results["best_solution"]["code"])

    finally:
        pass


if __name__ == "__main__":
    main()
