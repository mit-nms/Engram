"""
Generic ADRS Evaluator Wrapper

This module provides a generic wrapper that can connect any ADRS simulation environment
to Glia's optimization framework. It automatically discovers and wraps the evaluation
function from ADRS environments.
"""

import os
import sys
import importlib.util
import hashlib
import time
import json
import tempfile
import multiprocessing
import uuid
import yaml
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import uuid

# Add the root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from SystemBench.evaluator import Evaluator
from Architect.types import DesignConfig, Scenario, CodeBlock, CodeBlockImplementation


class ADRSEvaluator(Evaluator):
    """
    Generic evaluator wrapper for ADRS simulation environments.
    
    This class automatically discovers and wraps the evaluation function from any
    ADRS environment directory, making it compatible with Glia's optimization framework.
    """
    
    def __init__(self, adrs_env_path: str, target_name: str = None,
                 timeout: Optional[int] = None, evaluator_file: Optional[str] = None,
                 name: Optional[str] = None, **kwargs):
        """
        Initialize the ADRS evaluator.

        Args:
            adrs_env_path: Path to the ADRS environment directory
            target_name: Name of the code block to optimize (default: None, auto-discovered from evaluator module)
            timeout: Timeout in seconds for evaluation (default: None, uses runtime_threshold from parent)
            evaluator_file: Name of evaluator file to use (default: None, auto-discovers evaluator.py or evaluate.py)
            name: Display name for the environment (default: None, uses directory basename)
            **kwargs: Additional arguments passed to parent Evaluator
        """
        super().__init__(**kwargs)

        self.adrs_env_path = os.path.abspath(adrs_env_path)
        self.env_name = name if name else os.path.basename(self.adrs_env_path)
        self.random_hash = hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]
        self.evaluator_file = evaluator_file  # Store custom evaluator filename

        # Set timeout (use provided timeout, or runtime_threshold from parent, or default 600 seconds)
        self._timeout_seconds = timeout if timeout is not None else (self.runtime_threshold if self.runtime_threshold is not None else 600)

        # Discover the evaluation function (also stores module as self._evaluate_module)
        self.evaluate_func = self._discover_evaluate_function()

        # Auto-discover target_name from the evaluate module if not provided
        if target_name is None:
            if hasattr(self._evaluate_module, 'TARGET_NAME'):
                target_name = self._evaluate_module.TARGET_NAME
            else:
                target_name = "search_algorithm"
        self.target_name = target_name

        # Set up the target code block for optimization
        self._setup_code_block()

    
    def _discover_evaluate_function(self):
        """Discover the evaluate function from the ADRS environment."""
        # Use custom evaluator file if specified, otherwise look for evaluator.py then evaluate.py
        if self.evaluator_file:
            evaluate_file = os.path.join(self.adrs_env_path, self.evaluator_file)
            if not os.path.exists(evaluate_file):
                raise FileNotFoundError(f"{self.evaluator_file} not found in {self.adrs_env_path}")
        else:
            evaluate_file = os.path.join(self.adrs_env_path, "evaluator.py")
            if not os.path.exists(evaluate_file):
                evaluate_file = os.path.join(self.adrs_env_path, "evaluate.py")

            if not os.path.exists(evaluate_file):
                raise FileNotFoundError(f"evaluate.py or evaluator.py not found in {self.adrs_env_path}")
        
        # Load the evaluate module
        # Use the actual filename (without .py) as the module name so that
        # ProcessPoolExecutor can pickle/unpickle top-level functions correctly.
        module_name = os.path.splitext(os.path.basename(evaluate_file))[0]
        spec = importlib.util.spec_from_file_location(module_name, evaluate_file)
        evaluate_module = importlib.util.module_from_spec(spec)

        # Register the module in sys.modules so worker processes can find it
        sys.modules[module_name] = evaluate_module

        # Add the environment directory to the path for imports
        if self.adrs_env_path not in sys.path:
            sys.path.insert(0, self.adrs_env_path)
        try:
            spec.loader.exec_module(evaluate_module)
        except (ImportError, AttributeError) as e:
            print(f"Error: Import error in {self.env_name}: {e}")
            print(f"Please install the required dependencies for {self.env_name}")
            # Clean up on failure
            sys.modules.pop(module_name, None)
            if self.adrs_env_path in sys.path:
                sys.path.remove(self.adrs_env_path)
            raise
        
        # Get the evaluate function
        if not hasattr(evaluate_module, "evaluate"):
            raise AttributeError(f"evaluate function not found in {evaluate_file}")

        self._evaluate_module = evaluate_module
        return evaluate_module.evaluate

    def discover_helper_code_and_evolvable_code(self, initial_program: str) -> Tuple[str, str]:
        """If there is something outside # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END block, return it."""
        helper_code = ""
        if "EVOLVE-BLOCK" not in initial_program:
            return helper_code, initial_program
        else:
            evolvable_code = ""
            found_evolve_block_end = False
            for line in initial_program.split("\n"):
                if found_evolve_block_end:
                    helper_code += line + "\n"
                else:
                    evolvable_code += line + "\n"
                # Lines after EVOLVE-BLOCK-END
                if "EVOLVE-BLOCK-END" in line:
                    found_evolve_block_end = True
            return helper_code, evolvable_code

    def _setup_code_block(self):
        """Set up the target code block for optimization."""
        initial_program = self._get_default_implementation()
        helper_code, evolvable_code = self.discover_helper_code_and_evolvable_code(initial_program)
        self._code_blocks[self.target_name] = CodeBlock(
            name=self.target_name,
            description=f"Optimizable code block for {self.env_name} environment",
            evolvable_code=evolvable_code,
            helper_code=helper_code,
        )

    def _get_default_implementation(self) -> str:
        """Get the default implementation for the target code block."""
        # Look for initial_program.py or similar files
        initial_files = ["initial_program.py", "baseline.py", "default.py"]
        
        for filename in initial_files:
            filepath = os.path.join(self.adrs_env_path, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    content = f.read()
                    return content

        raise ValueError(f"No implementation found for {self.target_name}")

    def _load_config(self) -> Dict[str, Any]:
        """Load config.yaml from the ADRS environment directory if it exists."""
        config_path = os.path.join(self.adrs_env_path, "config.yaml")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                return {}
        return {}

    def get_system_model(self) -> str:
        """Get the system model description.

        Resolution order:
        1. config.yaml: prompt.system_model (or prompt.system_message if no system_model)
        2. SYSTEM_MODEL.md, DESCRIPTION.md in the env directory
        3. Generic fallback description.
        """
        # 1. Prefer config.yaml in the env directory (automatically discovered)
        config = self._load_config()
        if config and "prompt" in config:
            prompt_cfg = config["prompt"]
            # Explicit system_model takes precedence; then system_message
            system_model = prompt_cfg.get("system_model") or prompt_cfg.get("system_message")
            if system_model:
                return system_model.strip()

        # 2. Look for documentation files in the env directory
        readme_files = ["SYSTEM_MODEL.md", "DESCRIPTION.md"]
        for filename in readme_files:
            filepath = os.path.join(self.adrs_env_path, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return f.read()

        # 3. Fallback to a generic description
        return f"""
This is an ADRS simulation environment that requires optimization of the 
{self.target_name} code block. The environment evaluates algorithms
based on performance metrics and returns a combined score.
"""

    def _run_evaluation_with_timeout(self, program_path: str) -> Dict[str, Any]:
        """
        Run the evaluation function in a separate process with timeout.

        Args:
            program_path: Path to the program file to evaluate

        Returns:
            Dictionary with evaluation results

        Raises:
            TimeoutError: If evaluation exceeds the timeout threshold
        """
        # Create a queue to receive results from the subprocess
        result_queue = multiprocessing.Queue()

        # Pass adrs_env_path and evaluator_file as arguments since nested functions can't access self in multiprocessing
        adrs_env_path = self.adrs_env_path
        evaluator_file_name = self.evaluator_file

        def _evaluate_wrapper(path: str, env_path: str, eval_file_name: str, queue: multiprocessing.Queue):
            """Wrapper function to run evaluation in subprocess."""
            import io
            from contextlib import redirect_stdout, redirect_stderr

            # Capture stdout and stderr
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            try:
                with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                    # Re-discover the evaluate function in the subprocess
                    # (functions can't be pickled, so we need to re-import)
                    if eval_file_name:
                        evaluate_file = os.path.join(env_path, eval_file_name)
                    else:
                        evaluate_file = os.path.join(env_path, "evaluator.py")
                        if not os.path.exists(evaluate_file):
                            evaluate_file = os.path.join(env_path, "evaluate.py")

                    module_name = os.path.splitext(os.path.basename(evaluate_file))[0]
                    spec = importlib.util.spec_from_file_location(module_name, evaluate_file)
                    evaluate_module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = evaluate_module
                    if env_path not in sys.path:
                        sys.path.insert(0, env_path)
                    spec.loader.exec_module(evaluate_module)

                    results = evaluate_module.evaluate(path)

                # Add captured output to results
                results["stdout"] = stdout_buffer.getvalue()
                results["stderr"] = stderr_buffer.getvalue()
                queue.put(("success", results))
            except Exception as e:
                import traceback
                error_data = {
                    "error": f"{str(e)}\n{traceback.format_exc()}",
                    "stdout": stdout_buffer.getvalue(),
                    "stderr": stderr_buffer.getvalue(),
                }
                queue.put(("error", error_data))

        # Start the evaluation in a separate process
        process = multiprocessing.Process(target=_evaluate_wrapper, args=(program_path, adrs_env_path, evaluator_file_name, result_queue))
        process.start()

        # Wait for the process to complete, with timeout
        process.join(timeout=self._timeout_seconds)

        # Check if process is still running (timed out)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)  # Give it a moment to clean up
            if process.is_alive():
                process.kill()  # Force kill if it doesn't terminate
            raise TimeoutError(f"Evaluation timed out after {self._timeout_seconds} seconds")

        # Get results from queue
        if not result_queue.empty():
            status, data = result_queue.get()
            if status == "success":
                return data
            else:
                # Error case: data is now a dict with error, stdout, stderr
                error_msg = data.get("error", str(data)) if isinstance(data, dict) else str(data)
                stdout = data.get("stdout", "") if isinstance(data, dict) else ""
                stderr = data.get("stderr", "") if isinstance(data, dict) else ""
                # Include captured output in error message
                full_error = error_msg
                if stdout:
                    full_error += f"\n\n=== STDOUT ===\n{stdout}"
                if stderr:
                    full_error += f"\n\n=== STDERR ===\n{stderr}"
                raise RuntimeError(f"Evaluation failed: {full_error}")
        else:
            # Process completed but didn't put anything in queue (shouldn't happen)
            raise RuntimeError("Evaluation process completed without returning results")

    def run_simulation(self, design_config: DesignConfig, scenario: Scenario) -> Dict[str, Any]:
        """Run simulation with the given design configuration."""
        try:
            # Extract the algorithm code from the design config code blocks
            algorithm_code = None
            for cb in getattr(design_config, 'code_blocks', []) or []:
                if cb.name == self.target_name:
                    algorithm_code = str(cb.implementation)
                    break
            
            if algorithm_code is None:
                raise ValueError(f"No implementation found for {self.target_name}")
            
            # Generate unique filename to avoid race conditions when multiple evaluations run concurrently
            unique_id = str(uuid.uuid4())[:8]
            evolved_file_path = os.path.join(self.adrs_env_path, f"_evolved_program_{unique_id}.py")
            
            # Write to a real file in the environment directory instead of temp file
            # This allows proper module importing and avoids pickling issues with ProcessPoolExecutor
            # Generate unique filename to avoid race conditions when multiple evaluations run concurrently
            unique_id = str(uuid.uuid4())[:8]
            evolved_file_path = os.path.join(self.adrs_env_path, f"_evolved_program_{unique_id}.py")
            with open(evolved_file_path, 'w') as f:
                f.write(algorithm_code)
            temp_file_path = evolved_file_path
            
            try:
                # Run the evaluation with timeout protection
                try:
                    results = self._run_evaluation_with_timeout(temp_file_path)
                except TimeoutError as e:
                    # Handle timeout specifically
                    print(f"Evaluation timed out after {self._timeout_seconds} seconds")
                    return {
                        "success": False,
                        "score": float("-inf"),
                        "metrics": {},
                        "info": {"timeout_duration": self._timeout_seconds},
                        "error": str(e),
                        "error_type": "timeout",
                        "sim_dir": None
                    }

                # Validate that the results have the required structure
                if not isinstance(results, dict):
                    raise ValueError("Evaluation function must return a dictionary")
                
                if "combined_score" not in results:
                    raise ValueError("Evaluation results must include 'combined_score'")
                
                # Extract metrics
                metrics = {
                    "combined_score": results.get("combined_score", 0.0),
                    "runs_successfully": results.get("runs_successfully", 0.0),
                    "successful_configs": results.get("successful_configs", 0),
                    "failed_configs": results.get("failed_configs", 0),
                    "success_rate": results.get("success_rate", 0.0),
                }
                
                # Calculate success status
                # Check for error message - it might be present as a key with empty or non-empty value
                error_msg = results.get("error", "")
                has_error = bool(error_msg and error_msg.strip())
                # Some evaluators (e.g. txn_scheduling) return validity/combined_score but not
                # runs_successfully. Treat as success when: no error and (runs_successfully>0 OR
                # validity>0 OR combined_score>0 with no explicit error).
                runs_ok = results.get("runs_successfully", None)
                if runs_ok is not None:
                    success = runs_ok > 0.0 and not has_error
                else:
                    validity = results.get("validity", 0.0)
                    combined = results.get("combined_score", 0.0)
                    success = not has_error and (validity > 0.0 or combined > 0.0)
                return {
                    "success": success,
                    "score": results.get("combined_score", 0.0),
                    "metrics": metrics,
                    "info": {"raw_results": results},
                    "error": error_msg if has_error else "",  # Preserve error message if present
                    "error_type": "evaluation_error" if not success else "",
                    "sim_dir": results.get("sim_dir", None),
                    "stdout": results.get("stdout", ""),
                    "stderr": results.get("stderr", ""),
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            return {
                "success": False,
                "score": float("-inf"),
                "metrics": {},
                "info": {},
                "error": str(e),
                "error_type": self._get_error_type(e),
                "sim_dir": None
            }
    
    def _get_error_type(self, e: Exception) -> str:
        """Helper to determine error type from exception."""
        if isinstance(e, TimeoutError):
            return "timeout"
        elif isinstance(e, SyntaxError):
            return "syntax"
        elif isinstance(e, (RuntimeError, AssertionError)):
            return "runtime"
        elif isinstance(e, FileNotFoundError):
            return "file_not_found"
        elif isinstance(e, AttributeError):
            return "attribute_error"
        return "unknown"
    
    def analyze_results(self, results: Dict[str, Any]) -> None:
        """Analyze and print the results of the simulation."""
        if not results["success"]:
            print(f"Evaluation failed: {results['error']}")
            return
        
        metrics = results["metrics"]
        
        print(f"\n=== {self.env_name.upper()} EVALUATION RESULTS ===")
        print(f"Combined Score: {metrics.get('combined_score', 0):.4f}")
        print(f"Success Rate: {metrics.get('success_rate', 0):.2%}")
        print(f"Runs Successfully: {metrics.get('runs_successfully', 0):.2%}")
        
        if metrics.get('total_cost', 0) > 0:
            print(f"Total Cost: {metrics.get('total_cost', 0):.2f}")
            print(f"Average Cost: {metrics.get('avg_cost', 0):.2f}")
        
        if metrics.get('max_transfer_time', 0) > 0:
            print(f"Max Transfer Time: {metrics.get('max_transfer_time', 0):.2f}s")
        
        print(f"Successful Configs: {metrics.get('successful_configs', 0)}")
        print(f"Failed Configs: {metrics.get('failed_configs', 0)}")
        
        if metrics.get('time_score', 0) > 0:
            print(f"Time Score: {metrics.get('time_score', 0):.4f}")
        if metrics.get('cost_score', 0) > 0:
            print(f"Cost Score: {metrics.get('cost_score', 0):.4f}")
    
    def get_baselines(self) -> List[Tuple[str, str]]:
        """Get the baselines for the ADRS environment."""
        baselines = []
        
        # First, check if baselines.py exists and has a get_baselines function
        baselines_file = os.path.join(self.adrs_env_path, "baselines.py")
        if os.path.exists(baselines_file):
            try:
                # Load the baselines module
                spec = importlib.util.spec_from_file_location("baselines", baselines_file)
                baselines_module = importlib.util.module_from_spec(spec)

                # Add the environment directory to the path for imports
                sys.path.insert(0, self.adrs_env_path)
                try:
                    spec.loader.exec_module(baselines_module)
                finally:
                    # Remove the environment directory from path
                    if self.adrs_env_path in sys.path:
                        sys.path.remove(self.adrs_env_path)

                # Check if get_baselines function exists and call it
                if hasattr(baselines_module, "get_baselines") and callable(baselines_module.get_baselines):
                    baseline_results = baselines_module.get_baselines()
                    if isinstance(baseline_results, list):
                        baselines.extend(baseline_results)
            except Exception as e:
                print(f"Warning: Could not load get_baselines from baselines.py: {e}")

        # Fallback: Look for baseline implementations in files
        baseline_files = ["baseline.py", "initial_program.py"]
        for filename in baseline_files:
            filepath = os.path.join(self.adrs_env_path, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    content = f.read()
                    baselines.append((f"{filename} implementation", content))

        return baselines
