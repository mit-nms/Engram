"""One-shot optimization method."""

from typing import Dict, Any, Optional
from tqdm.auto import trange
import json

from .common import OptimizationMethod
from Architect.llm.openai_architect import create_and_test_algorithm


class OneShotOptimizer(OptimizationMethod):
    """One-shot optimization method that generates multiple independent solutions."""

    def __init__(
        self,
        *args,
        rounds: int = 1,
        resume_from: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the one-shot optimizer.

        Args:
            *args: Arguments to pass to parent class
            rounds: Number of optimization rounds
            resume_from: Path to previous run's log file to resume from
            **kwargs: Keyword arguments to pass to parent class
        """
        # Forward resume_from to parent class
        kwargs["resume_from"] = resume_from
        super().__init__(*args, **kwargs)

        self.rounds = rounds
        self.resume_from = resume_from
        self.start_round = 0
        self.all_iterations = []
        self.output_data = {}

    def _load_previous_run(self) -> None:
        """Load previous iterations from a previous run's log file."""
        if not self.resume_from:
            return

        try:
            with open(self.resume_from, 'r') as f:
                previous_run = json.load(f)

            # Load previous iterations
            self.all_iterations = previous_run.get('all_iterations', [])

            # Load individual samples
            for key, value in previous_run.items():
                if key.startswith(f"{self.model}_sample"):
                    self.output_data[key] = value

            # Set starting round
            if self.all_iterations:
                self.start_round = len(self.all_iterations)

            if self.debug:
                print(f"Resumed from round {self.start_round}")
                best_solution = max(self.all_iterations, key=lambda x: x["score"])
                print(f"Best score from previous run: {best_solution['score']:.2f}")

        except Exception as e:
            if self.debug:
                print(f"Error loading previous run: {str(e)}")
            self.start_round = 0
            self.all_iterations = []
            self.output_data = {}

    def optimize(self) -> Dict[str, Any]:
        """Run one-shot optimization.

        Returns:
            Dictionary containing optimization results
        """
        # Load previous run if specified
        self._load_previous_run()

        # Run optimization rounds
        print(f"\nRunning optimization with {self.model}...")
        for i in trange(self.rounds, desc="One-shot Optimization"):
            # Check if we should continue
            should_continue, reason = self.should_continue()
            if not should_continue:
                if self.debug:
                    print(f"\nStopping optimization: {reason}")
                break

            current_round = self.start_round + i
            if self.debug:
                print(f"\nRound {current_round+1}/{self.start_round + self.rounds}")

            # Create architect instance for this round
            architect = self.architect.__class__(
                model=self.model,
                task=self.task,
            )

            # Generate and test solution
            simulation_results = create_and_test_algorithm(
                architect=architect,
                debug=self.debug,
            )

            # Update tracking after each evaluation
            self.update_tracking(architect.get_usage_stats())

            # Store the results
            self.output_data[f"{self.model}_sample{current_round}"] = simulation_results
            self.all_iterations.append(simulation_results)

            if self.debug:
                print(f"Score: {simulation_results.get('score', 'N/A')}")
                if not simulation_results.get("success", False):
                    print(f"Error: {simulation_results.get('error', 'Unknown error')}")

            # Store intermediate results periodically
            if i % self.save_every_n_iterations == 0:
                print(f"Saving results for round {current_round}")
                # Add current iterations to output data
                self.output_data["all_iterations"] = self.all_iterations

                # Find current best solution
                best_solution = max(self.all_iterations, key=lambda x: x["score"])
                self.output_data["best_solution"] = {"code": best_solution["code"], "score": float(best_solution["score"])}

                # Save intermediate results
                self.save_results(
                    output_data=self.output_data,
                    method_name="one-shot",
                    extra_info=f"_{current_round+1}rounds"
                )

        # Add final iterations to output data
        self.output_data["all_iterations"] = self.all_iterations

        # Find best solution
        best_solution = max(self.all_iterations, key=lambda x: x["score"])
        self.output_data["best_solution"] = {"code": best_solution["code"], "score": float(best_solution["score"])}

        # Save and plot final results
        history_file, plot_path, _ = self.save_results(
            output_data=self.output_data,
            method_name="one-shot",
            extra_info=f"_{current_round+1}rounds"
        )

        # Plot results
        self.plot_results(
            save_path=plot_path,
            output_dict=self.output_data,
            title="One-shot Optimization Results",
            x_label="Reward",
            y_label="Frequency (%)",
            plot_type="one-shot",
        )

        return self.output_data
