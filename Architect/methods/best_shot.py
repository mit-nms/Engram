"""Best-shot optimization method."""

from typing import Dict, Any, List, Optional
import numpy as np
from tqdm.auto import trange
import json

from .common import OptimizationMethod
from Architect.llm.openai_architect import create_and_test_algorithm


class BestShotOptimizer(OptimizationMethod):
    """Best-shot optimization method that iteratively improves solutions."""

    def __init__(
        self,
        *args,
        num_generations: int = 100,
        best_shot_count: int = 3,
        island_max_count: int = 20,
        resume_from: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the best-shot optimizer.

        Args:
            *args: Arguments to pass to parent class
            num_generations: Number of generations to run
            best_shot_count: Number of best examples to include in prompts
            island_max_count: Maximum size of the solution database
            resume_from: Path to previous run's log file to resume from
            **kwargs: Keyword arguments to pass to parent class
        """
        # Forward resume_from to parent class
        kwargs["resume_from"] = resume_from
        super().__init__(*args, **kwargs)

        self.num_generations = num_generations
        self.best_shot_count = best_shot_count
        self.island_max_count = island_max_count
        self.resume_from = resume_from
        self.rng = np.random.RandomState(9001)
        self.start_generation = 0
        self.all_iterations = []
        self.solution_db = []

    def _load_previous_run(self) -> None:
        """Load solution database and history from a previous run's log file."""
        if not self.resume_from:
            return

        try:
            with open(self.resume_from, 'r') as f:
                previous_run = json.load(f)

            # Load previous iterations
            self.all_iterations = previous_run.get('all_iterations', [])

            # Load final solution database
            self.solution_db = previous_run.get('final_island', [])
            self.solution_db = sorted(self.solution_db, key=lambda x: x["score"])

            # Set starting generation
            if self.all_iterations:
                self.start_generation = len(self.all_iterations)

            if self.debug:
                print(f"Resumed from generation {self.start_generation} with solution database size {len(self.solution_db)}")
                print(f"Best score from previous run: {self.solution_db[-1]['score']:.2f}")

        except Exception as e:
            if self.debug:
                print(f"Error loading previous run: {str(e)}")
            self.start_generation = 0
            self.all_iterations = []
            self.solution_db = []

    def _add_examples_to_prompt(self, examples: List[Dict[str, Any]]) -> str:
        """Create a prompt that includes selected examples as inspiration."""
        examples_text = (
            "\n\nSome plausible implementations and their achieved scores (ordered from highest to lowest score):\n"
        )
        for i, res in enumerate(examples, 1):
            examples_text += f"\n# Implementation #{i} (score: {res['score']:.2f}):\n\n"
            examples_text += f"```python\n{res['code']}\n```\n\n"

        examples_text += "Implement your function. ONLY RETURN THE FUNCTION IMPLEMENTATION, NOTHING ELSE.\nTry to improve upon the above implementations."

        base_prompt_without_last_line = self.base_implement_prompt.split("\n")[:-1]
        base_prompt_without_last_line = "\n".join(base_prompt_without_last_line)
        return base_prompt_without_last_line + examples_text

    def optimize(self) -> Dict[str, Any]:
        """Run best-shot optimization.

        Returns:
            Dictionary containing optimization results
        """
        output_data = {}
        current_gen = self.start_generation # Initialize current_gen at the start

        # Load previous run if specified
        self._load_previous_run()

        # Phase 1: Initialize cohort if starting fresh
        if len(self.solution_db) == 0:
            for i in trange(self.best_shot_count, desc="Phase 1: Initializing Cohort"):
                # Check if we should continue
                should_continue, reason = self.should_continue()
                if not should_continue:
                    if self.debug:
                        print(f"\nStopping optimization: {reason}")
                    break

                architect = self.architect.__class__(
                    model=self.model,
                    task=self.task,
                )
                results = create_and_test_algorithm(architect=architect, debug=self.debug)

                # Update tracking after each evaluation
                self.update_tracking(architect.get_usage_stats())

                self.solution_db.append(results)
                self.all_iterations.append(results)
                if self.debug:
                    print(f"Initial solution {i + 1}/{self.best_shot_count}, Score: {results['score']:.2f}")

            self.solution_db = sorted(self.solution_db, key=lambda x: x["score"])
            if self.debug:
                print(f"\nInitial solutions found. Best score: {self.solution_db[-1]['score']:.2f}")

        # Phase 2: Iterative improvement
        for i in trange(self.num_generations, desc="Phase 2: Iterative improvement"):
            # Check if we should continue
            should_continue, reason = self.should_continue()
            if not should_continue:
                if self.debug:
                    print(f"\nStopping optimization: {reason}")
                break

            current_gen = self.start_generation + i
            sorted_examples = [
                self.solution_db[i] for i in self.rng.choice(len(self.solution_db), replace=False, size=self.best_shot_count)
            ]
            current_prompt = self._add_examples_to_prompt(sorted_examples)

            architect = self.architect.__class__(
                model=self.model,
                task=self.task,
            )
            architect.override_implement_prompt(current_prompt)
            results = create_and_test_algorithm(architect=architect, debug=self.debug)

            # Update tracking after each evaluation
            self.update_tracking(architect.get_usage_stats())

            self.solution_db.append(results)
            self.all_iterations.append(results)

            self.solution_db = sorted(self.solution_db, key=lambda x: x["score"])[-self.island_max_count :]

            if self.debug:
                print(
                    f"Iteration {current_gen + 1}/{self.start_generation + self.num_generations}, "
                    f"Score: {results['score']:.2f}, Best score so far: {self.solution_db[-1]['score']:.2f}"
                )

            if i % self.save_every_n_iterations == 0:
                # Prepare output data
                output_data.update(
                    {
                        "final_island": self.solution_db,
                        "all_iterations": self.all_iterations,
                        "best_solution": {"code": self.solution_db[-1]["code"], "score": float(self.solution_db[-1]["score"])},
                    }
                )

                # Save and plot intermediate results
                history_file, plot_path, _ = self.save_results(
                    output_data=output_data, method_name="best-shot", extra_info=f"_{current_gen + 1}gen"
                )

        # Prepare final output data
        output_data.update(
            {
                "final_island": self.solution_db,
                "all_iterations": self.all_iterations,
                "best_solution": {"code": self.solution_db[-1]["code"], "score": float(self.solution_db[-1]["score"])},
            }
        )

        # Save and plot final results
        history_file, plot_path, _ = self.save_results(
            output_data=output_data, method_name="best-shot", extra_info=f"_{current_gen + 1}gen"
        )

        # Plot results
        self.plot_results(
            save_path=plot_path,
            output_dict=output_data,
            title="Best-shot Optimization Results",
            x_label="Num Simulations",
            y_label="Reward",
            plot_type="stairs",
        )

        return output_data
