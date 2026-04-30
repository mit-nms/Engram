import time
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from abc import ABC, abstractmethod
import signal
import numpy as np

from Architect.types import Parameter, Function, Class, DesignConfig, Scenario, ProblemSpec, CodeBlock


class Evaluator(ABC):
    """Abstract base class for running algorithms"""

    @staticmethod
    def timeout_handler(signum, frame):
        """Handle timeout signal"""
        raise TimeoutError("simulation timed out")

    def __init__(self, runtime_threshold: Optional[int] = None):
        self.runtime_history = []
        self.runtime_threshold = runtime_threshold
        self.min_len_runtime_history = 10
        # Initialize protected dictionaries for parameters and functions
        self._parameters: Dict[str, Parameter] = {}
        self._functions: Dict[str, Function] = {}
        self._classes: Dict[str, Class] = {}
        self._code_blocks: Dict[str, CodeBlock] = {}
        self._output_metrics: List[str] = []

    def get_parameter(self, name: str) -> Parameter:
        """Get a parameter by name"""
        if name not in self._parameters:
            raise ValueError(f"Parameter {name} not found")
        return self._parameters[name]

    def get_function(self, name: str) -> Function:
        """Get a function by name"""
        if name not in self._functions:
            raise ValueError(f"Function {name} not found")
        return self._functions[name]

    def get_class(self, name: str) -> Class:
        """Get a class by name"""
        if name not in self._classes:
            raise ValueError(f"Class {name} not found")
        return self._classes[name]

    def get_code_block(self, name: str) -> CodeBlock:
        """Get a code block by name"""
        if name not in self._code_blocks:
            raise ValueError(f"Code block {name} not found")
        return self._code_blocks[name]

    def get_changeable_parameters(self) -> List[Parameter]:
        """Get list of parameters that can be modified"""
        return list(self._parameters.values())

    def get_changeable_functions(self) -> List[Function]:
        """Get list of functions that can be modified"""
        return list(self._functions.values())

    def get_changeable_classes(self) -> List[Class]:
        """Get list of classes that can be modified"""
        return list(self._classes.values())

    def get_changeable_code_blocks(self) -> List[CodeBlock]:
        """Get list of code blocks that can be modified"""
        return list(self._code_blocks.values())

    def get_output_metrics(self) -> List[str]:
        """Get list of output metrics"""
        return list(self._output_metrics)

    def set_parameter(self, parameter_or_name: Union[Parameter, str], value: Optional[Any] = None):
        """Set a parameter value. Can be called with either:
        1. set_parameter(parameter: Parameter)
        2. set_parameter(name: str, value: Any)
        """
        if isinstance(parameter_or_name, Parameter):
            parameter = parameter_or_name
            if parameter.name in self._parameters:
                self._parameters[parameter.name] = parameter
        else:
            name = parameter_or_name
            if name not in self._parameters:
                raise ValueError(f"Parameter {name} not found")
            if value is None:
                raise ValueError("Value must be provided when calling with parameter name")
            self._parameters[name].value = value

    def set_function(self, function_or_name: Union[Function, str], code: Optional[str] = None):
        """Set a function implementation. Can be called with either:
        1. set_function(function: Function)
        2. set_function(name: str, code: str)
        """
        if isinstance(function_or_name, Function):
            # Case 1: Called with Function object
            function = function_or_name
            if function.name in self._functions:
                self._functions[function.name] = function
                if hasattr(self, 'algorithm_code') and function.name == getattr(self, '_func_name', None):
                    self.algorithm_code = str(function.implementation)
        else:
            # Case 2: Called with name and code
            name = function_or_name
            if name not in self._functions:
                raise ValueError(f"Function {name} not found")
            if code is None:
                raise ValueError("Code must be provided when calling with function name")
            self._functions[name].implementation.code = code
            if hasattr(self, 'algorithm_code') and name == getattr(self, '_func_name', None):
                self.algorithm_code = code

    def set_class(self, class_or_name: Union[Class, str], code: Optional[str] = None, helper_code: Optional[str] = None):
        """Set a class implementation. Can be called with either:
        1. set_class(class: Class)
        2. set_class(name: str, code: str)
        """
        if isinstance(class_or_name, Class):
            # Case 1: Called with Class object
            cls = class_or_name
            if cls.name in self._classes:
                self._classes[cls.name] = cls
                if hasattr(self, 'algorithm_code') and cls.name == getattr(self, '_class_name', None):
                    self.algorithm_code = str(cls.implementation)
                if helper_code is not None:
                    self._classes[cls.name].helper_code = helper_code
        else:
            # Case 2: Called with name and code
            name = class_or_name
            if name not in self._classes:
                raise ValueError(f"Class {name} not found")
            if code is None:
                raise ValueError("Code must be provided when calling with class name")
            self._classes[name].implementation.code = code
            if hasattr(self, 'algorithm_code') and name == getattr(self, '_class_name', None):
                self.algorithm_code = code

    # removed duplicate earlier definition of set_code in favor of a single unified method below

    def set_code_block(self, code_block_or_name: Union[CodeBlock, str], code: Optional[str] = None):
        """Set a code block implementation. Can be called with either:
        1. set_code_block(code_block: CodeBlock)
        2. set_code_block(name: str, code: str)
        """
        if isinstance(code_block_or_name, CodeBlock):
            code_block = code_block_or_name
            if code_block.name in self._code_blocks:
                self._code_blocks[code_block.name] = code_block
                if hasattr(self, 'algorithm_code') and code_block.name == getattr(self, '_code_block_name', None):
                    self.algorithm_code = str(code_block.implementation)
        else:
            name = code_block_or_name
            if name not in self._code_blocks:
                raise ValueError(f"Code block {name} not found")
            if code is None:
                raise ValueError("Code must be provided when calling with code block name")
            self._code_blocks[name].implementation.code = code + "\n\n" + self._code_blocks[name].helper_code
            if hasattr(self, 'algorithm_code') and name == getattr(self, '_code_block_name', None):
                self.algorithm_code = code + "\n\n" + self._code_blocks[name].helper_code

    def set_code(self, name: str, code: str):
        """Set the code for a parameter, function, class, or code block"""
        if name in self._parameters:
            self._parameters[name].value = code
            self.set_parameter(name, code)
        elif name in self._functions:
            self._functions[name].implementation.code = code
            self.set_function(name, code)
        elif name in self._classes:
            self._classes[name].implementation.code = code
            self.set_class(name, code)
        elif name in self._code_blocks:
            self._code_blocks[name].implementation.code = code
            self.set_code_block(name, code)
        else:
            raise ValueError(f"Name {name} not found")

    def get_current_design_config(self) -> DesignConfig:
        """Get the current design config"""
        return DesignConfig(
            parameters=list(self._parameters.values()),
            functions=list(self._functions.values()),
            classes=list(self._classes.values()),
            code_blocks=list(self._code_blocks.values())
        )


    def get_default_scenario(self) -> Scenario:
        """Return the default scenario used by Task.evaluate().

        Override in subclasses that require specific scenario configuration.
        """
        return Scenario(name="default", config={})

    def get_objective_value(self, problem_spec: ProblemSpec) -> Tuple[float, List[str]]:
        """Get objective for given config using the selected scenarios
        Args:
            problem_spec: Specification for the problem
        Returns:
            float value of the objective
        """
        results = {}
        sim_dirs = []
        for scenario in problem_spec.scenarios:
            result = self.run_simulation(problem_spec.design_config, scenario)
            results[scenario.name] = result
            # Only append sim_dir when present (failed runs may omit it)
            sim_dir = result.get("sim_dir")
            if sim_dir:
                sim_dirs.append(sim_dir)
        return problem_spec.objective_fn(results), sim_dirs, results

    def get_system_model(self) -> str:
        """Get the system model description. Optional -- only needed by legacy prompt compilation."""
        return ""

    @abstractmethod
    def run_simulation(self, design_config: DesignConfig, scenario: Scenario) -> Dict[str, Any]:
        """Run simulation with the given design configuration on the selected scenario that has been set

        Args:
            design_config: Configuration including parameter values and function implementations
            scenario: Scenario to run the simulation for

        Returns:
            A dictionary containing the results of the simulation
            - 'success': bool           -> Was the evaluation successful?
            - 'error_type': str         -> e.g., timeout, syntax, runtime, unknown, etc.
            - 'error': str              -> If there was an error or exception, description of the error, e.g., str(Exception e)
            - 'metrics': Dict[str, Any] -> Metrics of the simulation
            - 'info': Dict[str, Any]    -> Extra information with no particular structure
            - 'sim_dir': str            -> Path to the simulation output directory
        """
        pass

    def analyze_results(self, results: Dict[str, Any]) -> None:
        """Analyze the results of the simulation. Optional."""
        pass

    def get_baselines(self) -> List[Tuple[str, str]]:
        """Get the baselines of the simulation along with descriptions. Optional."""
        return []

    @property
    def _timeout(self) -> Optional[float]:
        if self.runtime_threshold is not None:
            return self.runtime_threshold
        elif len(self.runtime_history) >= self.min_len_runtime_history:
            x = np.array(self.runtime_history)
            # Heuristic timeout threshold. Set once and never revisited (if dynamic, can mess with score consistency).
            # Why x[ x <= (np.mean( x ) + 2 * np.std( x )) ]: To remove outliers.
            # Why multiply by 2: To give some flexibility.
            self.runtime_threshold = np.ceil(2 * x[x <= (np.mean(x) + 2 * np.std(x))].mean()).astype(int).item()
            print(f"Timeout threshold moved to {self.runtime_threshold}")
            return self.runtime_threshold
        else:
            return None
