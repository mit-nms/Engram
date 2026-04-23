from dataclasses import dataclass, field
from typing import Dict, Any, List, Union, Optional, Callable
from enum import Enum


@dataclass
class ParameterValue:
    value: Any
    valid: bool = True
    error: str = ""


@dataclass
class Parameter:
    """Parameter specification"""
    name: str
    description: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    possible_values: Optional[List[Any]] = None
    value: Optional[Any] = None


@dataclass
class CodeBlockImplementation:
    """Class to store function implementation details"""
    implementation: Optional[str] = None
    default: Optional[str] = None

    def __str__(self):
        """Return the implementation string or default if implementation is None"""
        return self.implementation if self.implementation is not None else self.default

    def __eq__(self, other):
        """Allow direct comparison with strings"""
        if isinstance(other, str):
            return str(self) == other
        return super().__eq__(other)

    @property
    def code(self):
        """Get the actual implementation code"""
        return self.implementation if self.implementation is not None else self.default

    @code.setter
    def code(self, value: str):
        """Set the implementation code"""
        self.implementation = value

@dataclass
class Function:
    name: str
    description: str
    constraints: List[str] = field(default_factory=list)
    required_inputs: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    implementation: CodeBlockImplementation = field(default_factory=CodeBlockImplementation)
    dependencies: str = ""  # Classes and methods to use for the implementation

    def set_implementation(self, new_implementation: str):
        self.implementation.code = new_implementation


@dataclass
class CodeBlock:
    name: str
    description: str
    implementation: CodeBlockImplementation = field(default_factory=CodeBlockImplementation)
    evolvable_code: str = ""
    helper_code: str = ""
    dependencies: str = ""  # Classes and methods to use for the implementation

    def set_implementation(self, new_implementation: str):
        self.implementation.code = "\n\n" + self.helper_code + "\n\n" + new_implementation

@dataclass
class ClassImplementation:
    """Class to store class implementation details"""
    implementation: Optional[str] = None
    default: Optional[str] = None
    base_class: str = ""
    required_methods: List[str] = field(default_factory=list)

    def __str__(self):
        """Return the implementation string or default if implementation is None"""
        return self.implementation if self.implementation is not None else self.default

    def __eq__(self, other):
        """Allow direct comparison with strings"""
        if isinstance(other, str):
            return str(self) == other
        return super().__eq__(other)

    @property
    def code(self):
        """Get the actual implementation code"""
        return self.implementation if self.implementation is not None else self.default

    @code.setter
    def code(self, value: str):
        """Set the implementation code"""
        self.implementation = value


@dataclass
class Class:
    name: str
    description: str
    base_class: str
    required_methods: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    implementation: ClassImplementation = field(default_factory=ClassImplementation)
    dependencies: str = ""  # Classes and methods to use for the implementation
    helper_code: str = ""  # Code to be added before the class definition

    def set_implementation(self, new_implementation: str):
        self.implementation.code = new_implementation


@dataclass
class Scenario:
    """Test scenario specification"""
    name: str
    config: Any
    description: str = ""


@dataclass
class DesignConfig:
    """Configuration for a system design"""
    parameters: List[Parameter]
    functions: List[Function]
    classes: List[Class] = field(default_factory=list)
    code_blocks: List[CodeBlock] = field(default_factory=list)


@dataclass
class ProblemSpec:
    """Specification for a system problem"""
    design_config: DesignConfig                      # Design configuration
    scenarios: List[Scenario]                        # Test scenarios
    objective_fn: Callable[[Dict[str, Any]], float]  # Objective function


# All the variable names given for maximization or minimization must have explanations,
# then use those explanations in compile time
class OptimizationType(Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    GOAL = "goal"


@dataclass
class Objective:
    type: OptimizationType
    description: str = ""
    objective_fn: Callable[[Dict[str, Any]], float] = field(default=None)
    target_value: Optional[float] = None
    constraints: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of validation"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"valid": self.valid, "errors": self.errors, "warnings": self.warnings, "details": self.details}
