import hashlib
from typing import Dict, Any, List, Tuple, Optional
import sys
import os
import time

import numpy as np
import pandas as pd
from pathlib import Path
import glob
import logging
import atexit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from SystemBench.evaluator import Evaluator
from Architect.utils import print_table
from .baselines import baselines_codes
from .descriptions import system_model, get_class_dependencies
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.scheduler.global_scheduler.global_scheduler_registry import GlobalSchedulerRegistry
from vidur.types.global_scheduler_type import GlobalSchedulerType
from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds
from Architect.types import DesignConfig, Scenario, Class, ClassImplementation

from vidur.entities.request import Request
from vidur.entities.batch import Batch
from vidur.entities.batch_stage import BatchStage
from vidur.entities.cluster import Cluster
from vidur.entities.execution_time import ExecutionTime
from vidur.entities.replica import Replica

from vidur.events.batch_end_event import BatchEndEvent
from vidur.events.batch_stage_arrival_event import BatchStageArrivalEvent
from vidur.events.batch_stage_end_event import BatchStageEndEvent
from vidur.events.global_schedule_event import GlobalScheduleEvent
from vidur.events.replica_schedule_event import ReplicaScheduleEvent
from vidur.events.replica_stage_schedule_event import ReplicaStageScheduleEvent
from vidur.events.request_arrival_event import RequestArrivalEvent

class VidurEvaluator(Evaluator):
    """Implementation that executes Python code in a controlled environment for Vidur global scheduling"""

    def __init__(self, **kwargs):
        """Initialize the Vidur evaluator"""
        super().__init__(**kwargs)
        self.target_name = "CustomGlobalScheduler"
        self.algorithm_code = None
        self.random_hash = hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]

        self._classes = {
            "CustomGlobalScheduler": Class(
                name="CustomGlobalScheduler",
                description="Custom global scheduler for LLM inference",
                base_class="BaseGlobalScheduler",
                required_methods=["schedule"],
                implementation=ClassImplementation(
                    default=baselines_codes[0][1] if baselines_codes else None,  # Use first baseline as default
                    base_class="BaseGlobalScheduler",
                    required_methods=["schedule"]
                ),
                dependencies=get_class_dependencies("CustomGlobalScheduler")
            )
        }

        # Define output metrics
        self._output_metrics = [
            "prefill_e2e_p50",
            "prefill_e2e_p90",
            "decode_time_p50",
            "decode_time_p90",
            "throughput",
            "qps"
        ]

    def get_default_scenario(self) -> Scenario:
        return Scenario(name="sarathi_qps7.5", config={"replica_scheduler": "sarathi", "qps": 7.5})

    def get_system_model(self) -> str:
        return system_model

    def run_simulation_with_algorithm_code(self, algorithm_code: str, scenario: Scenario) -> Dict[str, Any]:
        """Run simulation with the given algorithm code"""
        self.algorithm_code = algorithm_code
        if self.algorithm_code is None:
                raise ValueError(f"No implementation found for {self.target_name}")
        try:
            # Use the unified set_code approach
            self.set_code(self.target_name, self.algorithm_code)

            # Execute the code in the namespace
            namespace = self._generate_namespace()
            logger.info("Executing algorithm code...")
            logger.debug(f"Code to execute:\n{self.algorithm_code}")

            try:
                # First compile the code to catch syntax errors
                compiled_code = compile(self.algorithm_code, '<string>', 'exec')
                # Then execute it
                exec(compiled_code, namespace)
            except Exception as e:
                logger.error(f"Failed to execute code: {e}")
                logger.error(f"Namespace keys: {list(namespace.keys())}")
                raise

            # Get and validate the class
            scheduler_class = namespace.get(self.target_name)
            if not scheduler_class:
                logger.error(f"Available classes in namespace: {[k for k in namespace.keys() if not k.startswith('_')]}")
                raise SyntaxError(f"Algorithm code must define a '{self.target_name}' class")

            # Verify the class is properly defined
            if not isinstance(scheduler_class, type):
                raise TypeError(f"{self.target_name} is not a class type")
            if not issubclass(scheduler_class, BaseGlobalScheduler):
                raise TypeError(f"{self.target_name} must inherit from BaseGlobalScheduler")

            # Unregister the existing CUSTOM type if it exists
            if GlobalSchedulerType.CUSTOM in GlobalSchedulerRegistry._registry:
                del GlobalSchedulerRegistry._registry[GlobalSchedulerType.CUSTOM]

            # Register the custom scheduler
            GlobalSchedulerRegistry.register(GlobalSchedulerType.CUSTOM, scheduler_class)
            logger.info(f"Successfully registered {self.target_name} as custom scheduler")
            replica_scheduler = scenario.config.get("replica_scheduler", "sarathi")
            qps = scenario.config.get("qps", 7.5)
            print(f"Replica scheduler: {replica_scheduler}, QPS: {qps}")
            # Run the simulation with the custom scheduler
            metrics, _ = self._run_simulation(replica_scheduler, qps)
            if not metrics:
                raise RuntimeError("Simulation produced no metrics")

            score = self._calculate_score(metrics)
            if not isinstance(score, (int, float)):
                raise ValueError(f"Invalid score type: {type(score)}")

            return {
                "success": True,
                "score": score,
                "metrics": {
                    "prefill_e2e_p50": metrics["prefill_e2e_p50"],
                    "prefill_e2e_p90": metrics["prefill_e2e_p90"],
                    "decode_time_p50": metrics["decode_time_p50"],
                    "decode_time_p90": metrics["decode_time_p90"],
                    "average_request_slowdown": metrics["request_e2e_slowdown"],
                    "request_slowdown_p50": metrics["request_slowdown_p50"],
                    "request_slowdown_p90": metrics["request_slowdown_p90"],
                    "throughput": metrics["throughput"],
                    "qps": metrics["qps"]
                },
                "info": {"metrics": metrics},
                "error": "",
                "error_type": ""
            }

        except Exception as e:
            logger.error(f"Error executing algorithm: {str(e)}", exc_info=True)
            return {
                "success": False,
                "score": float("-inf"),
                "metrics": {},
                "error": str(e),
                "error_type": self._get_error_type(e),
                "info": {}
            }

    def run_simulation(self, design_config: DesignConfig, scenario: Scenario) -> Dict[str, Any]:
        """Run simulation with the given design configuration"""
        try:
            print("-" * 100)
            print(f"Design config classes implementation: {design_config.classes[0].implementation}")
            print("-" * 100)
            # Apply design config - find the implementation
            self.algorithm_code = None
            for cls in design_config.classes:
                if cls.name == self.target_name:
                    self.algorithm_code = str(cls.implementation)

            if self.algorithm_code is None:
                raise ValueError(f"No implementation found for {self.target_name}")

            # Use the unified set_code approach
            self.set_code(self.target_name, self.algorithm_code)

            # Execute the code in the namespace
            namespace = self._generate_namespace()
            logger.info("Executing algorithm code...")
            logger.debug(f"Code to execute:\n{self.algorithm_code}")

            try:
                # First compile the code to catch syntax errors
                compiled_code = compile(self.algorithm_code, '<string>', 'exec')
                # Then execute it
                exec(compiled_code, namespace)
            except Exception as e:
                logger.error(f"Failed to execute code: {e}")
                logger.error(f"Namespace keys: {list(namespace.keys())}")
                raise

            # Get and validate the class
            scheduler_class = namespace.get(self.target_name)
            if not scheduler_class:
                logger.error(f"Available classes in namespace: {[k for k in namespace.keys() if not k.startswith('_')]}")
                raise SyntaxError(f"Algorithm code must define a '{self.target_name}' class")

            # Verify the class is properly defined
            if not isinstance(scheduler_class, type):
                raise TypeError(f"{self.target_name} is not a class type")
            if not issubclass(scheduler_class, BaseGlobalScheduler):
                raise TypeError(f"{self.target_name} must inherit from BaseGlobalScheduler")

            # Unregister the existing CUSTOM type if it exists
            if GlobalSchedulerType.CUSTOM in GlobalSchedulerRegistry._registry:
                del GlobalSchedulerRegistry._registry[GlobalSchedulerType.CUSTOM]

            # Register the custom scheduler
            GlobalSchedulerRegistry.register(GlobalSchedulerType.CUSTOM, scheduler_class)
            logger.info(f"Successfully registered {self.target_name} as custom scheduler")
            replica_scheduler = scenario.config.get("replica_scheduler", "sarathi")
            qps = scenario.config.get("qps", 7.5)
            print(f"Replica scheduler: {replica_scheduler}, QPS: {qps}")
            # Run the simulation with the custom scheduler
            metrics, sim_dir = self._run_simulation(replica_scheduler, qps)
            if not metrics:
                raise RuntimeError("Simulation produced no metrics")

            score = self._calculate_score(metrics)
            if not isinstance(score, (int, float)):
                raise ValueError(f"Invalid score type: {type(score)}")

            return {
                "success": True,
                "score": score,
                "metrics": {
                    "prefill_e2e_p50": metrics["prefill_e2e_p50"],
                    "prefill_e2e_p90": metrics["prefill_e2e_p90"],
                    "decode_time_p50": metrics["decode_time_p50"],
                    "decode_time_p90": metrics["decode_time_p90"],
                    "average_request_slowdown": metrics["average_request_slowdown"],
                    "request_slowdown_p50": metrics["request_slowdown_p50"],
                    "request_slowdown_p90": metrics["request_slowdown_p90"],
                    "throughput": metrics["throughput"],
                    "qps": metrics["qps"]
                },
                "sim_dir": sim_dir,
                "info": {"metrics": metrics},
                "error": "",
                "error_type": ""
            }

        except Exception as e:
            logger.error(f"Error executing algorithm: {str(e)}", exc_info=True)
            return {
                "success": False,
                "score": float("-inf"),
                "metrics": {},
                "error": str(e),
                "error_type": self._get_error_type(e),
                "info": {}
            }

    def _get_error_type(self, e: Exception) -> str:
        """Helper to determine error type from exception"""
        if isinstance(e, TimeoutError):
            return "timeout"
        elif isinstance(e, SyntaxError):
            return "syntax"
        elif isinstance(e, (RuntimeError, AssertionError)):
            return "runtime"
        return "unknown"

    def _run_simulation(self, replica_scheduler: str, qps: float=7.0) -> Tuple[Dict[str, Any], str]:
        """Run the simulation and collect metrics"""
        # Save original sys.argv
        original_argv = sys.argv.copy()

        # Get the absolute path to the vidur directory
        vidur_dir = os.path.dirname(os.path.abspath(__file__))

        # Create random hash for output folder
        output_path = os.path.join(vidur_dir, "vidur_repo", "simulator_output", self.random_hash)
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Writing output to {output_path}")
        if replica_scheduler == "vllm":
            replica_scheduler_config = [
                "--replica_scheduler_config_type", "vllm",
                "--vllm_scheduler_config_batch_size_cap", "128",
                "--vllm_scheduler_config_max_tokens_in_batch", "8192",
                "--vllm_scheduler_config_block_size", "16",
                "--vllm_scheduler_config_watermark_blocks_fraction", "0.01",
            ]
        elif replica_scheduler == "sarathi":
            replica_scheduler_config = [
                "--replica_scheduler_config_type", "sarathi",
                "--sarathi_scheduler_config_num_blocks", "2240",
                "--sarathi_scheduler_config_batch_size_cap", "128",
                "--sarathi_scheduler_config_chunk_size", "8192",
                "--sarathi_scheduler_config_block_size", "16",
                "--sarathi_scheduler_config_watermark_blocks_fraction", "0.01",
            ]
        else:
            raise ValueError(f"Invalid replica scheduler: {replica_scheduler}")
        # fmt: off
        # Set up our arguments
        sys.argv = [
            "vidur.main",
            "--replica_config_device", "a10",
            "--replica_config_memory_margin_fraction", "0.2445",
            "--replica_config_model_name", "meta-llama/Meta-Llama-3-8B",
            "--cluster_config_num_replicas", "4",
            "--global_scheduler_config_type", "custom",
            "--replica_config_tensor_parallel_size", "1",
            "--replica_config_num_pipeline_stages", "1",
            *replica_scheduler_config,

            "--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size", "8192",
            "--random_forrest_execution_time_predictor_config_prediction_max_batch_size", "128",
            "--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request", "8192",

            "--request_generator_config_type", "trace_replay",
            "--trace_request_generator_config_trace_file", f"{vidur_dir}/vidur_repo/data/processed_traces/sharegpt_{qps}.csv",
            "--trace_request_generator_config_max_tokens", "8192",
            "--length_generator_config_type", "trace",
            "--trace_request_length_generator_config_trace_file", f"{vidur_dir}/vidur_repo/data/processed_traces/sharegpt_{qps}.csv",
            "--trace_request_length_generator_config_max_tokens", "8192",
            "--interval_generator_config_type", "trace",
            "--trace_request_interval_generator_config_trace_file", f"{vidur_dir}/vidur_repo/data/processed_traces/sharegpt_{qps}.csv",

            "--metrics_config_write_metrics",
            "--no-metrics_config_store_global_scheduler_logs",
            "--no-metrics_config_write_json_trace",
            "--no-metrics_config_enable_chrome_trace",
            "--no-metrics_config_save_table_to_wandb",
            "--no-metrics_config_store_plots",
            "--no-metrics_config_store_operation_metrics",
            "--no-metrics_config_store_token_completion_metrics",
            "--metrics_config_store_request_metrics",
            "--no-metrics_config_store_batch_metrics",
            "--no-metrics_config_store_utilization_metrics",
            "--no-metrics_config_keep_individual_batch_metrics",
            "--metrics_config_output_dir", f"{output_path}",
        ]
        # fmt: on

        original_dir = os.getcwd()
        start_time = time.time()
        try:
            # Change to the vidur directory
            os.chdir(f"{vidur_dir}/vidur_repo")
            logger.info(f"Changed working directory to: {vidur_dir}/vidur_repo")

            # Reset class-level fields, otherwise simulation reuses indices from past results.
            Request._id = -1
            Batch._id = -1
            BatchStage._id = -1
            Cluster._id = -1
            ExecutionTime._id = -1
            Replica._id = -1
            # these were not necessary, but can't hurt
            BatchEndEvent._id = 0
            BatchStageArrivalEvent._id = 0
            BatchStageEndEvent._id = 0
            GlobalScheduleEvent._id = 0
            ReplicaScheduleEvent._id = 0
            ReplicaStageScheduleEvent._id = 0
            RequestArrivalEvent._id = 0
            # sanity check
            assert GlobalScheduleEvent.outstanding is False

            # Create simulation config with custom scheduler
            logger.info("Creating simulation config...")
            config = SimulationConfig.create_from_cli_args()
            logger.info("Simulation config created successfully")

            # Set random seeds
            set_seeds(config.seed)
            logger.info("Random seeds set")

            # Create and run simulator
            logger.info("Creating simulator...")
            simulator = Simulator(config)
            atexit.unregister(simulator._write_output)

            logger.info("Running simulation...")
            simulator.run()
            logger.info("Simulation completed")

            # Force metric write
            simulator._write_output()

            # Find the output directory
            run_output_dir = self._find_matching_simulator_output(output_path, start_time)
            if not run_output_dir:
                raise RuntimeError("Could not find simulation output directory")
            logger.info(f"Found output directory: {run_output_dir}")

            # Parse metrics
            metrics = self._parse_metrics(run_output_dir)
            if not metrics:
                raise RuntimeError("Could not parse metrics from output directory")
            logger.info("Metrics parsed successfully")

            return metrics, run_output_dir

        except Exception as e:
            logger.error(f"Error running simulation: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error running simulation: {str(e)}")
        finally:
            # Restore original directory and sys.argv
            os.chdir(original_dir)
            logger.info(f"Restored working directory to: {original_dir}")
            sys.argv = original_argv

    @staticmethod
    def _find_matching_simulator_output(output_path: str, start_time: float) -> Optional[str]:
        """Find the simulator output directory that matches our run"""
        output_dirs = glob.glob(f"{output_path}/*")
        if not output_dirs:
            return None

        output_dirs.sort(key=os.path.getmtime, reverse=True)

        for output_dir in output_dirs:
            dir_time = os.path.getmtime(output_dir)
            if dir_time >= start_time:
                return output_dir

        return None

    @staticmethod
    def _parse_metrics(output_dir: str) -> Dict[str, Any]:
        """Parse metrics from the simulator output directory"""
        metrics = {}

        request_metrics_file = Path(output_dir) / "request_metrics.csv"
        if not request_metrics_file.exists():
            return None

        df = pd.read_csv(request_metrics_file)
        metrics['output_path'] = request_metrics_file
        df["decode_e2e_time"] = df["request_e2e_time"] - df["prefill_e2e_time"]
        df["arrival_time"] = df["request_inter_arrival_delay"].cumsum()

        # Performance metrics
        metrics["prefill_e2e_p50"] = float(df["prefill_e2e_time"].quantile(0.5))  # Convert to ms
        metrics["prefill_e2e_p90"] = float(df["prefill_e2e_time"].quantile(0.9))
        metrics["decode_time_p50"] = float(df["decode_e2e_time"].quantile(0.5))
        metrics["decode_time_p90"] = float(df["decode_e2e_time"].quantile(0.9))
        metrics["average_request_slowdown"] = float(df["request_e2e_slowdown"].mean())
        metrics["request_slowdown_p50"] = float(df["request_e2e_slowdown"].quantile(0.5))
        metrics["request_slowdown_p90"] = float(df["request_e2e_slowdown"].quantile(0.9))

        total_time = float(df["request_e2e_time"].max())
        last_arr = float(df["arrival_time"].max())
        metrics["throughput"] = float(len(df) / total_time if total_time > 0 else 0)
        metrics["qps"] = float(len(df) / last_arr if last_arr > 0 else np.inf)

        # Raw metrics
        metrics["raw_metrics"] = {
            "num_requests": int(len(df)),
            # "total_tokens": int(df["request_num_tokens"].sum()),
            # "avg_tokens_per_request": float(df["request_num_tokens"].mean()),
            "restarts": int(df["request_num_restarts"].sum()),
            "avg_scheduling_delay": float(df["request_scheduling_delay"].mean()),
            "avg_execution_time": float(df["request_execution_time"].mean()),
            "avg_preemption_time": float(df["request_preemption_time"].mean()),
            "total_time": float(total_time),
            # "tokens_per_second": float(df["request_num_tokens"].sum() / total_time if total_time > 0 else 0),
            # "avg_prefill_tokens": float(df["request_num_prefill_tokens"].mean()),
            # "avg_decode_tokens": float(df["request_num_decode_tokens"].mean()),
            # "avg_pd_ratio": float(df["request_pd_ratio"].mean()),
            "avg_request_e2e_time": float(df["request_e2e_time"].mean()),
            "average_request_slowdown": float(df["request_e2e_slowdown"].mean()),
            "request_slowdown_p50": float(df["request_e2e_slowdown"].quantile(0.5)),
            "request_slowdown_p90": float(df["request_e2e_slowdown"].quantile(0.9)),
            "request_e2e_p90": float(df["request_e2e_time"].quantile(0.9)),
        }

        return metrics

    @staticmethod
    def _calculate_score(metrics: Dict[str, Any]) -> float:
        """Calculate a score based on the metrics"""
        # Higher throughput and lower latencies are better
        # We'll use a weighted combination of metrics
        # weights = {"throughput": 0.4, "prefill_e2e_p50": -0.2, "decode_time_p50": -0.2, "tokens_per_second": 0.2}
        weights = {"avg_request_e2e_time": (1.0, 20.0)}

        score = 0
        for metric, (weight, numer) in weights.items():
            if metric in metrics:
                score += numer / metrics[metric] * weight
            elif metric in metrics.get("raw_metrics", {}):
                score += numer / metrics["raw_metrics"][metric] * weight

        return max(0, score)  # Ensure score is non-negative

    def get_baselines(self) -> List[Tuple[str, str]]:
        """Get the baselines for the Vidur replica scheduler"""
        return baselines_codes

    def analyze_results(self, stats: Dict[str, Any]) -> None:
        """Analyze and print the results of the Vidur replica scheduler"""
        if not stats["success"]:
            print(f"Evaluation failed: {stats['error']}")
            return

        metrics = stats["info"]["metrics"]
        raw_metrics = metrics.get("raw_metrics", {})

        # Performance Metrics
        table = []
        table.append(["Performance Metrics"])
        table.append(
            [
                "Prefill E2E Latency",
                f"{metrics.get('prefill_e2e_p50', 0):.2f}s (P50) / {metrics.get('prefill_e2e_p90', 0):.2f}s (P90)",
            ]
        )
        table.append(
            [
                "Decode Time",
                f"{metrics.get('decode_time_p50', 0):.2f}s (P50) / {metrics.get('decode_time_p90', 0):.2f}s (P90)",
            ]
        )
        table.append(
            [
                "Request E2E Time",
                f"{raw_metrics.get('avg_request_e2e_time', 0):.2f}s (avg) / {raw_metrics.get('request_e2e_p90', 0):.2f}s (P90)",
            ]
        )
        table.append(["Throughput", f"{metrics.get('throughput', 0):.2f} requests/second"])
        table.append(["", f"{raw_metrics.get('tokens_per_second', 0):.2f} tokens/second"])

        # Request Statistics
        table.append([])
        table.append(["Request Statistics"])
        table.append([])
        table.append(["Total Requests", f"{raw_metrics.get('num_requests', 0):,}"])
        table.append(["Total Tokens", f"{raw_metrics.get('total_tokens', 0):,}"])
        table.append(["Avg Tokens/Request", f"{raw_metrics.get('avg_tokens_per_request', 0):.2f}"])
        table.append(["QPS", f"{metrics.get('qps', 0):.2f} requests/second"])
        table.append(["Total Restarts", f"{raw_metrics.get('restarts', 0)}"])

        # Timing Information
        table.append([])
        table.append(["Timing Information"])
        table.append([])
        table.append(["Avg Scheduling Delay", f"{raw_metrics.get('avg_scheduling_delay', 0):.2f}s"])
        table.append(["Avg Execution Time", f"{raw_metrics.get('avg_execution_time', 0):.2f}s"])
        table.append(["Avg Preemption Time", f"{raw_metrics.get('avg_preemption_time', 0):.2f}s"])

        # Token Distribution
        table.append([])
        table.append(["Token Distribution"])
        table.append([])
        table.append(["Avg Prefill Tokens", f"{raw_metrics.get('avg_prefill_tokens', 0):.2f}"])
        table.append(["Avg Decode Tokens", f"{raw_metrics.get('avg_decode_tokens', 0):.2f}"])
        table.append(["Avg P/D Ratio", f"{raw_metrics.get('avg_pd_ratio', 0):.2f}"])

        table.append([])
        table.append(["Final Score", f"{stats['score']:.2f}"])

        print_table(table)

    @staticmethod
    def _generate_namespace():
        """Generate the namespace for code execution"""
        return {
            # Type hints
            "List": List,
            "Tuple": Tuple,
            "Optional": Optional,
            "Dict": Dict,
            "Any": Any,

            # Required base classes and types
            "BaseGlobalScheduler": BaseGlobalScheduler,
            "GlobalSchedulerRegistry": GlobalSchedulerRegistry,
            "GlobalSchedulerType": GlobalSchedulerType,
            "Request": Request,

            # Math and numerical libraries
            "np": np,
            "math": __import__('math'),

            # Python builtins that might be needed
            "float": float,
            "int": int,
            "len": len,
            "print": print,

            # Special variables
            "__name__": "__main__",
            "__builtins__": __builtins__,
        }
