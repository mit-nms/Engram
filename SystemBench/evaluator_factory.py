"""
Evaluator factory module for creating task-specific evaluators.
"""

from typing import Dict, Type, List
from .evaluator import Evaluator
import traceback


class SysBench:
    """Factory class for creating evaluators based on task type."""

    _evaluator_registry: Dict[str, Type[Evaluator]] = {}

    @classmethod
    def register(cls, task_type: str, evaluator_class: Type[Evaluator]) -> None:
        """Register an evaluator class for a specific task type.

        Args:
            task_type: The task type identifier
            evaluator_class: The evaluator class to register
        """
        cls._evaluator_registry[task_type] = evaluator_class

    @classmethod
    def get_evaluator(cls, task_type: str, **kwargs) -> Evaluator:
        """Create and return an evaluator instance for the specified task type.

        Args:
            task_type: The task type identifier

        Returns:
            An instance of the appropriate evaluator

        Raises:
            ValueError: If the task type is not supported
        """
        if task_type not in cls._evaluator_registry:
            available_types = ", ".join(cls._evaluator_registry.keys())
            raise ValueError(f"Unsupported task type: {task_type}. Available types are: {available_types}")

        evaluator_class = cls._evaluator_registry[task_type]
        return evaluator_class(**kwargs)

    @classmethod
    def list_available_evaluators(cls) -> List[str]:
        """List all available evaluator task types.

        Returns:
            List of available task types
        """
        return list(cls._evaluator_registry.keys())


# Register known evaluator types
try:
    from .vidur.env_evaluator import VidurEvaluator

    SysBench.register("vidur", VidurEvaluator)
except ImportError as e:
    print(traceback.format_exc())
    # This allows the module to be imported even if some evaluators are not available
    pass

try:
    from .park_bridge.park_evaluator import ParkEvaluator

    # Register a generic alias; specific envs are resolved via get_evaluator
    SysBench.register("park", ParkEvaluator)  # allows SysBench.get_evaluator("park", env_id="abr_sim")
except ImportError:
    pass

try:
    from .FrontierCS.frontier_cs_evaluator import FrontierCSEvaluator

    SysBench.register("frontier_cs", FrontierCSEvaluator)
except ImportError:
    pass
