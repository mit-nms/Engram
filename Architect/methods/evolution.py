"""
Evolutionary optimization method.
https://raw.githubusercontent.com/mlresearch/v235/main/assets/liu24bs/liu24bs.pdf
"""

from typing import Dict, Any, List, Union, Optional, Type
import numpy as np
from tqdm.auto import trange
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import signal
import os

from .common import OptimizationMethod
from Architect.llm.openai_architect import create_and_test_algorithm

def _dummy_handler(signum, frame):
    """Dummy handler that just raises TimeoutError"""
    raise TimeoutError("simulation timed out")

def initialize_worker():
    """Initialize worker process."""
    # Set up signal handlers for timeouts
    signal.signal(signal.SIGALRM, _dummy_handler)
    # Disable CUDA in worker processes to avoid conflicts
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

def _create_offspring_wrapper(args):
    """Wrapper function for multiprocessing."""
    return args[0]._create_offspring(*args[1:])

class Individual:
    """Represents a single heuristic candidate: its code, fitness score, and reasoning."""

    def __init__(self, code: str, score: float, reasoning: str = "", metadata: Dict[str, Any] = None):
        self.code = code
        self.score = score
        self.reasoning = reasoning
        self.age = 0
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "score": self.score, "reasoning": self.reasoning, "metadata": self.metadata}


class Population:
    """Maintains a sorted list of Individuals, up to max_size."""

    def __init__(self, size: int = 20):
        self.individuals: List[Individual] = [] # all the individuals in the population
        self.parents: List[Individual] = [] # the M individuals in the population
        self.children: List[Individual] = [] # the N-M individuals in the population
        self.max_size = size

    def add(self, individual: Individual, role: str = "parent"):
        if role == "child":
            self.children.append(individual)
        elif role == "parent":
            self.parents.append(individual)
        else:
            raise ValueError(f"Invalid role: {role}")
        self.individuals.append(individual)
        self.individuals.sort(key=lambda x: x.score, reverse=True) # highest score first
        if len(self.individuals) > self.max_size:
            self.individuals = self.individuals[: self.max_size]

    def get_best_individuals(self, k: int) -> List[Individual]:
        if len(self.individuals) <= k:
            return list(self.individuals)
        return self.individuals[:k]

    def get_random_parents(self, k: int, rng: np.random.RandomState) -> List[Individual]:
        if len(self.parents) <= k:
            return list(self.parents)
        idx = rng.choice(len(self.parents), size=k, replace=False)
        return [self.parents[i] for i in idx]


class EvolutionOptimizer(OptimizationMethod):
    """Full EoH: implements E1/E2 exploration and M1/M2/M3 mutation operators."""

    def __init__(
        self,
        *args,
        population_size: int = 40, # N
        tournament_size: int = 3, # tournament selection for choosing the best parent, used if random_selection is False
        mutation_rate: float = 0.1,
        elitism_count: int = 20, # M
        num_generations: int = 100,
        resume_from: Optional[str] = None,
        seeds_path: Optional[str] = None,
        seed_protection_generations: int = 0,  # Number of generations to protect seeds
        random_selection: bool = True, # If True, use completely random selection instead of tournament selection
        disable_diversify: bool = True, # If True, disable the diversify operator
        **kwargs,
    ):
        # Forward kwargs to parent class
        kwargs["resume_from"] = resume_from
        super().__init__(*args, **kwargs)

        self.save_every_n_iterations = 5
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.num_generations = num_generations
        self.resume_from = resume_from
        self.seeds_path = seeds_path
        self.seed_protection_generations = seed_protection_generations
        self.random_selection = random_selection
        self.rng = np.random.RandomState(42)
        self.population = Population(population_size) # starts off with the M individuals, expands to N
        self.start_generation = 0
        self.disable_diversify = disable_diversify
        self.all_iters = []
        # save all the args in a json file
        with open(f"{self.log_dir}/args.json", "w") as f:
            json.dump(vars(self), f, indent=2, default=str)
        # print all the args in a nice way in red
        print("\033[91m" + "\n".join([f"{k}: {v}" for k, v in vars(self).items()]) + "\033[0m")

    def _load_previous_run(self) -> None:
        """Load population and history from a previous run's log file."""
        if not self.resume_from:
            return

        try:
            with open(self.resume_from, "r") as f:
                previous_run = json.load(f)

            # Load previous iterations
            self.all_iters = previous_run.get("all_iterations", [])

            # Load final population
            final_pop = previous_run.get("final_population", [])
            for ind_dict in final_pop:
                ind = Individual(
                    code=ind_dict["code"],
                    score=ind_dict["score"],
                    reasoning=ind_dict.get("reasoning", ""),
                    metadata=ind_dict.get("metadata", {}),
                )
                role = ind_dict.get("metadata", {}).get("role", "parent")
                self.population.add(ind, role=role)

            # Set starting generation
            if self.all_iters:
                max_gen = max(iter["metadata"].get("generation", 0) for iter in self.all_iters)
                self.start_generation = max_gen + 1

            if self.debug:
                print(
                    f"Resumed from generation {self.start_generation} with population size {len(self.population.individuals)}"
                )
                print(f"Best score from previous run: {self.population.get_best_individuals(1)[0].score:.2f}")

        except Exception as e:
            if self.debug:
                print(f"Error loading previous run: {str(e)}")
            self.start_generation = 0
            self.all_iters = []
            self.population = Population(self.population_size)

    def _load_seeds(self) -> List[Individual]:
        """Load seed algorithms from JSON file and evaluate them."""
        if self.debug:
            print(f"_load_seeds called with seeds_path: {self.seeds_path}")

        if not self.seeds_path:
            if self.debug:
                print("No seeds_path provided, returning empty list")
            return []

        if self.debug:
            print(f"Attempting to load seeds from: {self.seeds_path}")

        try:
            with open(self.seeds_path, "r") as f:
                seeds_data = json.load(f)

            if self.debug:
                print(f"Successfully loaded JSON with keys: {list(seeds_data.keys())}")

            seeds = []
            helper_code = seeds_data.get("helper_code", "")
            if self.debug:
                print(f"Helper code length: {len(helper_code) if helper_code else 0}")

            if not helper_code and self.debug:
                print("Warning: No helper_code found in seeds file")

            # Evaluate each seed
            for seed_id, seed in enumerate(seeds_data["seeds"]):
                if self.debug:
                    print(f"Looking for seed {seed_id}...")

                seed_code = seed["code"]
                seed_reasoning = seed["reasoning"]

                # Combine helper code with seed implementation
                full_code = helper_code + "\n\n" + seed_code

                try:
                    # Set the code directly and evaluate
                    self.task.evaluator.set_code(self.task.evaluator.target_name, full_code)
                    score, sim_dirs, results = self.task.evaluate()
                    print(f"\033[91mSeed reasoning: {seed_reasoning}: score = {score:.10f}\033[0m")
                    metadata = {
                        "generation": 0,
                        "operation": "seed_initialization",
                        "seed_idea": seed_id,
                        "sim_dirs": sim_dirs,
                        "results": results
                    }

                    seed_individual = Individual(
                        code=full_code, score=score, reasoning=seed_reasoning, metadata=metadata
                    )
                    seeds.append(seed_individual)

                    if self.debug:
                        print(f"Seed {seed_id}: score = {score:.2f} - SUCCESS")

                except Exception as e:
                    if self.debug:
                        print(f"Error evaluating seed {seed_id}: {str(e)}")
                        import traceback

                        traceback.print_exc()
                    continue

            if self.debug:
                print(f"Successfully loaded {len(seeds)} seeds out of 7 attempted")

            return seeds

        except Exception as e:
            if self.debug:
                print(f"Error loading seeds from {self.seeds_path}: {str(e)}")
                import traceback

                traceback.print_exc()
            return []

    def _create_prompt_with_examples(self, examples: List[Individual]) -> str:
        """Include parent implementations and reasoning as context for the LLM."""
        examples_text = "\n\nHere are some example implementations and their performance:\n"
        for i, ex in enumerate(examples, start=1):
            examples_text += f"\n# Example {i} (score: {ex.score:.2f}):\n````python\n{ex.code}\n```\n"
            if ex.reasoning:
                examples_text += f"Reasoning: {ex.reasoning}\n"
        examples_text += (
            "\nImplement your function, building on these examples. "
            "ONLY RETURN THE FUNCTION IMPLEMENTATION, NOTHING ELSE."
        )
        # Remove last line of base prompt to append examples
        base = self.base_implement_prompt.split("\n")[:-1]
        return "\n".join(base) + examples_text

    def _parent_tournament_selection(self, k: int) -> List[Individual]:
        """Select k parents via tournament from current population."""
        selected = []
        if self.random_selection:
            selected = self.population.get_random_parents(k, self.rng)
        else:
            for _ in range(k):
                tour = self.population.get_random_parents(self.tournament_size, self.rng)
                winner = max(tour, key=lambda x: x.score)
                selected.append(winner)
        return selected

    # --- Exploration Operators ---
    def _diversify(self, parents: List[Individual]) -> str:
        prompt = self._create_prompt_with_examples(parents)
        prompt += "\nCreate a brand-new solution that is as different as possible from these examples."
        return prompt

    def _recombine(self, parents: List[Individual]) -> str:
        prompt = self._create_prompt_with_examples(parents)
        prompt += "\nAnalyze the strengths of each implementation and combine their best aspects into a new solution."
        return prompt

    # --- Mutation Operators ---
    def _modify(self, individual: Individual) -> str:
        prompt = self._create_prompt_with_examples([individual])
        prompt += "\nIdentify weaknesses in this implementation and create an improved version that addresses them."
        return prompt

    def _tweak(self, individual: Individual) -> str:
        prompt = self._create_prompt_with_examples([individual])
        prompt += "\nTweak internal parameters (e.g., weights, thresholds) to boost performance."
        return prompt

    def _simplify(self, individual: Individual) -> str:
        prompt = self._create_prompt_with_examples([individual])
        prompt += "\nSimplify this implementation by removing redundant or unnecessary components."
        return prompt

    def optimize(self) -> Dict[str, Any]:
        """Run the full EoH evolution for num_generations."""
        output = {}

        # Load previous run if specified
        self._load_previous_run()

        # Check if we should continue
        should_continue, reason = self.should_continue()
        if not should_continue:
            if self.debug:
                print(f"\nStopping optimization: {reason}")
            return output

        # Initialize population if starting fresh
        if len(self.population.individuals) == 0:
            # First, load seeds if available
            seeds = self._load_seeds()
            if len(seeds) > 0:
                print(f"Found {len(seeds)} seeds, setting population size to {len(seeds)}")
                for seed in seeds:
                    # Mark this individual as a seed in metadata
                    seed.metadata["is_seed"] = True
                    seed.metadata["role"] = "parent"
                    self.population.add(seed, role="parent")
                    # Add to iterations for tracking
                    self.all_iters.append(
                        {
                            "code": seed.code,
                            "score": seed.score,
                            "reasoning": seed.reasoning,
                            "metadata": seed.metadata,
                            "usage_stats": {
                                "total_cost": 0.0,
                                "total_prompt_tokens": 0,
                                "total_completion_tokens": 0,
                                "total_tokens": 0,
                            },  # Seeds don't cost anything
                        }
                    )

            # Fill remaining population slots with random generation
            remaining_slots = self.population_size - len(self.population.individuals) if self.seeds_path is None else 0
            print(f"Population size: {self.population_size}")
            print(f"Remaining slots: {remaining_slots} since we have {len(seeds)} seeds")
            if remaining_slots > 0:
                for _ in trange(remaining_slots, desc="Generating Random Individuals"):
                    should_continue, reason = self.should_continue()
                    if not should_continue:
                        if self.debug:
                            print(f"\nStopping optimization: {reason}")
                        break
                    architect = self.architect.__class__(
                        model=self.model,
                        task=self.task,
                    )
                    res = create_and_test_algorithm(architect=architect, debug=self.debug)
                    self.update_tracking(architect.get_usage_stats())
                    metadata = {
                        "generation": 0,
                        "operation_type": "random_initialization",
                        "llm_conversation": architect.get_conversation_history(),
                        "role": "parent",  # All initial individuals are parents
                        "is_seed": False,
                    }
                    ind = Individual(
                        code=res["code"], score=res["score"], reasoning=res.get("reasoning", ""), metadata=metadata
                    )
                    self.population.add(ind, role="parent")  # Mark as parent
                    self.all_iters.append({**res, "metadata": metadata})

        # Ensure we have exactly M parents by selecting the best M individuals
        if len(self.population.individuals) > self.elitism_count:
            # Sort all individuals by score and select top M as parents
            all_individuals = self.population.individuals.copy()
            all_individuals.sort(key=lambda x: x.score, reverse=True)

            # Create new population with only the top M as parents
            new_pop = Population(self.population_size)
            for i, ind in enumerate(all_individuals[:self.elitism_count]):
                ind.metadata["role"] = "parent"
                new_pop.add(ind, role="parent")

            self.population = new_pop
            if self.debug:
                print(f"\033[91mSelected top {self.elitism_count} individuals as initial parents\033[0m")
                print(f"\033[91mBest initial score: {self.population.get_best_individuals(1)[0].score:.2f}\033[0m")

        # Get serializable model configuration
        model_config = {'model': self.model} if isinstance(self.model, str) else (
            self.model.to_dict() if hasattr(self.model, 'to_dict') else vars(self.model)
        )

        # Evolution loop
        for gen in trange(self.num_generations, desc="Evolution Progress"):
            should_continue, reason = self.should_continue()
            if not should_continue:
                if self.debug:
                    print(f"\nStopping optimization: {reason}")
                break
            current_gen = self.start_generation + gen
            new_pop = Population(self.population_size)

            # Get current parents (M individuals)
            current_parents = self.population.parents

            # Generate N-M children from parents
            children_to_generate = self.population_size - self.elitism_count

            if self.debug:
                print(f"\033[91mCurrent parents: {len(current_parents)}\033[0m")
                print(f"\033[91mGenerating {children_to_generate} children\033[0m")
                print(f"\033[91mPopulation size: {self.population_size}\033[0m")
                print(f"\033[91mElitism count: {self.elitism_count}\033[0m")
                print(f"\033[91mMutation rate: {self.mutation_rate}\033[0m")
                print(f"\033[91mRandom selection: {self.random_selection}\033[0m")
                print(f"\033[91mSeed protection generations: {self.seed_protection_generations}\033[0m")
                print(f"\033[91mDisable diversify: {self.disable_diversify}\033[0m")
            # Generate children
            children = []
            for _ in range(children_to_generate):
                operation_type = "mutation" if self.rng.random() < self.mutation_rate else "exploration"
                if operation_type == "mutation":
                    parents = self._parent_tournament_selection(1)
                else:
                    parents = self._parent_tournament_selection(2)

                try:
                    result = self._create_offspring(
                        operation_type,
                        parents,
                        current_gen,
                        self.architect.__class__,
                        model_config,
                        self.debug
                    )

                    # Update tracking stats
                    self.update_tracking(result["usage_stats"])

                    # Create new individual
                    ind = Individual(
                        code=result["code"],
                        score=result["score"],
                        reasoning=result["reasoning"],
                        metadata=result["metadata"]
                    )
                    ind.metadata["role"] = "child"
                    children.append(ind)
                    self.all_iters.append({**result["res"], "metadata": result["metadata"]})

                except Exception as e:
                    if self.debug:
                        print(f"Error in offspring creation: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    continue

            # Combine parents and children, then select top M
            combined_population = current_parents + children
            combined_population.sort(key=lambda x: x.score, reverse=True)  # Sort by score descending

            # Handle seed protection during early generations
            if current_gen < self.seed_protection_generations:
                # Find all seeds in the combined population
                seeds = [ind for ind in combined_population if ind.metadata.get("is_seed", False)]
                non_seeds = [ind for ind in combined_population if not ind.metadata.get("is_seed", False)]

                # Prioritize seeds, then fill remaining slots with best non-seeds
                new_parents = seeds[:self.elitism_count]  # Take up to M seeds
                remaining_slots = self.elitism_count - len(new_parents)
                if remaining_slots > 0 and non_seeds:
                    new_parents.extend(non_seeds[:remaining_slots])
            else:
                # Normal selection: just take top M
                new_parents = combined_population[:self.elitism_count]

            if self.debug:
                print(f"\033[91mCombined population size: {len(combined_population)}\033[0m")
                print(f"\033[91mSelected {len(new_parents)} new parents\033[0m")
                print(f"\033[91mBest score: {new_parents[0].score:.2f}\033[0m")
                if current_gen < self.seed_protection_generations:
                    seeds_in_new_parents = sum(1 for p in new_parents if p.metadata.get("is_seed", False))
                    print(f"\033[91mSeeds in new parents: {seeds_in_new_parents}\033[0m")

            # Add new parents to population
            for parent in new_parents:
                parent.metadata["survived_generations"] = parent.metadata.get("survived_generations", 0) + 1
                parent.metadata["role"] = "parent"
                new_pop.add(parent, role="parent")

            self.population = new_pop
            if self.debug:
                print(
                    f"Gen {current_gen+1}/{self.start_generation + self.num_generations}, Best: {self.population.get_best_individuals(1)[0].score:.2f}"
                )

            if gen % self.save_every_n_iterations == 0:
                final_db = [ind.to_dict() for ind in self.population.individuals]
                output.update(
                    {
                        "final_population": final_db,
                        "all_iterations": self.all_iters,
                        "best_solution": {
                            "code": final_db[0]["code"],
                            "score": float(final_db[0]["score"]),
                            "metadata": final_db[0]["metadata"],
                        },
                    }
                )

                # Save & plot intermediate results
                hist_file, plot_path, _ = self.save_results(
                    output_data=output, method_name="evolution", extra_info=f"_{current_gen+1}gen"
                )

        # Collect final results
        final_db = [ind.to_dict() for ind in self.population.individuals]
        output.update(
            {
                "final_population": final_db,
                "all_iterations": self.all_iters,
                "best_solution": {
                    "code": final_db[0]["code"],
                    "score": float(final_db[0]["score"]),
                    "metadata": final_db[0]["metadata"],
                },
            }
        )

        # Save & plot final results
        hist_file, plot_path, _ = self.save_results(
            output_data=output, method_name="evolution", extra_info=f"_{current_gen+1}gen"
        )
        self.plot_results(
            save_path=plot_path,
            output_dict=output,
            title="Evolution Optimization Results",
            x_label="Num Simulations",
            y_label="Reward",
            plot_type="stairs",
        )

        return output

    def _create_offspring(self, operation_type: str, parents: List[Individual], current_gen: int, architect_class, model_config: Dict, debug: bool) -> Dict[str, Any]:
        """Helper function to create a single offspring for parallel execution."""
        if operation_type == "mutation":
            # Mutation family: choose one of M1/M2/M3
            parent = parents[0]
            op = np.random.choice([self._modify, self._tweak, self._simplify])
            operation_name = op.__name__[1:]  # Remove leading underscore
            prompt = op(parent)
            parent_info = {"parent_code": parent.code, "parent_score": parent.score}
        else:
            # Exploration family: choose E1/E2
            if self.disable_diversify:
                op = self._recombine
            else:
                op = np.random.choice([self._diversify, self._recombine])
            operation_name = op.__name__[1:]  # Remove leading underscore
            prompt = op(parents)
            parent_info = {
                "parent1_code": parents[0].code,
                "parent1_score": parents[0].score,
                "parent2_code": parents[1].code,
                "parent2_score": parents[1].score,
            }

        # Use the task and model directly
        task = self.task

        # Handle string model identifiers
        model = model_config['model'] if isinstance(model_config, dict) and 'model' in model_config else (
            type(self.model)(**model_config)
        )

        architect = architect_class(
            model=model,
            task=task,
        )
        architect.override_implement_prompt(prompt)
        res = create_and_test_algorithm(architect=architect, debug=debug)
        usage_stats = architect.get_usage_stats()

        metadata = {
            "generation": current_gen + 1,
            "operation_type": operation_type,
            "operation_name": operation_name,
            "parent_info": parent_info,
            "llm_conversation": architect.get_conversation_history(),
            "prompt_used": prompt,
            "is_seed": False,
        }

        return {
            "code": res["code"],
            "score": res["score"],
            "reasoning": res.get("reasoning", ""),
            "metadata": metadata,
            "usage_stats": usage_stats,
            "res": res
        }
