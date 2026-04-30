import functools
import importlib.util
import json
import os
import time
import traceback
from typing import TypedDict

import torch
import numpy as np

TARGET_NAME = "rebalance_experts"
import re
import inspect

# Get the directory where this evaluator file is located
EVALUATOR_DIR = os.path.dirname(os.path.abspath(__file__))
WORKLOAD_PATH = os.path.join(EVALUATOR_DIR, "expert-load.json")
REBALANCE_INTERVAL = 100

NUM_REPLICAS = 288
NUM_GROUPS = 8
NUM_GPUS = 32
NUM_NODES = 4

# Check if workload file exists
if not os.path.exists(WORKLOAD_PATH):
    raise FileNotFoundError(f"Workload file {WORKLOAD_PATH} not found. "
        "Please download the workload file as instructed in the `README.md` "
        "under the `eplb` directory."
    )

@functools.cache
def load_workloads(path: str) -> list[torch.Tensor]: 
    with open(path, "r") as f:
        data = json.load(f)

    total_len = len(data['load_history'])
    workloads = []
    for i in range(0, total_len, REBALANCE_INTERVAL):
        start = i
        end = min(start + REBALANCE_INTERVAL, total_len)

        load = torch.tensor([x['logical_expert_load'] for x in data['load_history'][start:end]]).sum(dim=0)
        workloads.append(load)

    return workloads

class EvaluationResult(TypedDict, total=False):
    balancedness_score: float
    speed_score: float
    combined_score: float
    error: str
    success: bool
    runs_successfully: float

def validate_rebalance_solution(
    phy2log: torch.Tensor,
    log2phy: torch.Tensor,
    logcnt: torch.Tensor,
    num_layers: int,
    num_logical_experts: int,
    num_replicas: int,
) -> tuple[bool, list[str], dict]:
    """
    Validate that the rebalance solution is correct and not cheating.

    Checks:
    1. Tensor shapes are correct
    2. All physical experts are used exactly once per layer
    3. All indices are within valid ranges
    4. expert_count sums to num_replicas for each layer
    5. logical_to_physical_map contains valid physical indices
    6. No duplicate or missing assignments
    7. Consistency between phy2log and log2phy

    Returns: (is_valid, error_messages, validation_stats)
    """
    errors = []
    stats = {
        'shape_errors': 0,
        'index_range_errors': 0,
        'missing_physical_experts': 0,
        'duplicate_physical_experts': 0,
        'expert_count_mismatch': 0,
        'invalid_log2phy_entries': 0,
        'inconsistency_errors': 0,
    }

    # Check tensor shapes
    expected_phy2log_shape = (num_layers, num_replicas)
    expected_logcnt_shape = (num_layers, num_logical_experts)

    if phy2log.shape != expected_phy2log_shape:
        errors.append(
            f"Invalid phy2log shape: expected {expected_phy2log_shape}, got {phy2log.shape}"
        )
        stats['shape_errors'] += 1
        return False, errors, stats

    if logcnt.shape != expected_logcnt_shape:
        errors.append(
            f"Invalid logcnt shape: expected {expected_logcnt_shape}, got {logcnt.shape}"
        )
        stats['shape_errors'] += 1
        return False, errors, stats

    # Check log2phy shape (third dimension can vary, but should be at least max_replicas)
    if len(log2phy.shape) != 3 or log2phy.shape[0] != num_layers or log2phy.shape[1] != num_logical_experts:
        errors.append(
            f"Invalid log2phy shape: expected (num_layers, num_logical_experts, max_replicas), got {log2phy.shape}"
        )
        stats['shape_errors'] += 1
        return False, errors, stats

    max_replicas = log2phy.shape[2]

    # Check each layer
    for layer_id in range(num_layers):
        # Check 1: expert_count sums to num_replicas
        expert_count_sum = logcnt[layer_id].sum().item()
        if expert_count_sum != num_replicas:
            errors.append(
                f"Layer {layer_id}: expert_count sums to {expert_count_sum}, expected {num_replicas}"
            )
            stats['expert_count_mismatch'] += 1

        # Check 2: All physical experts are assigned (0 to num_replicas-1)
        physical_experts_used = set()
        physical_experts_seen_in_phy2log = set()

        # Check phy2log: each physical expert should map to a valid logical expert
        for phys_id in range(num_replicas):
            logical_id = phy2log[layer_id, phys_id].item()

            # Check index range
            if logical_id < 0 or logical_id >= num_logical_experts:
                errors.append(
                    f"Layer {layer_id}, physical expert {phys_id}: invalid logical expert index {logical_id} "
                    f"(must be in [0, {num_logical_experts}))"
                )
                stats['index_range_errors'] += 1
                continue

            physical_experts_seen_in_phy2log.add(phys_id)

        # Check 3: All physical experts 0 to num_replicas-1 are present
        missing_physical = set(range(num_replicas)) - physical_experts_seen_in_phy2log
        if missing_physical:
            errors.append(
                f"Layer {layer_id}: Missing physical experts: {sorted(missing_physical)}"
            )
            stats['missing_physical_experts'] += len(missing_physical)

        # Check 4: No duplicates in phy2log (each physical expert should appear once)
        if len(physical_experts_seen_in_phy2log) != num_replicas:
            errors.append(
                f"Layer {layer_id}: Expected {num_replicas} unique physical experts in phy2log, "
                f"found {len(physical_experts_seen_in_phy2log)}"
            )
            stats['duplicate_physical_experts'] += 1

        # Check 5: logical_to_physical_map consistency
        for logical_id in range(num_logical_experts):
            num_reps = int(logcnt[layer_id, logical_id].item())

            if num_reps < 1:
                errors.append(
                    f"Layer {layer_id}, logical expert {logical_id}: expert_count is {num_reps}, must be >= 1"
                )
                stats['expert_count_mismatch'] += 1
                continue

            if num_reps > max_replicas:
                errors.append(
                    f"Layer {layer_id}, logical expert {logical_id}: expert_count ({num_reps}) exceeds "
                    f"max_replicas dimension ({max_replicas})"
                )
                stats['expert_count_mismatch'] += 1
                continue

            # Check log2phy entries for this logical expert
            physical_ids_in_log2phy = []
            for rep_rank in range(num_reps):
                phys_id = log2phy[layer_id, logical_id, rep_rank].item()

                # Check if it's a valid index (non-negative and within range)
                if phys_id < 0 or phys_id >= num_replicas:
                    errors.append(
                        f"Layer {layer_id}, logical expert {logical_id}, replica {rep_rank}: "
                        f"invalid physical expert index {phys_id} (must be in [0, {num_replicas}))"
                    )
                    stats['invalid_log2phy_entries'] += 1
                    continue

                physical_ids_in_log2phy.append(phys_id)
                physical_experts_used.add(phys_id)

            # Check for duplicates within a logical expert's replicas
            if len(physical_ids_in_log2phy) != len(set(physical_ids_in_log2phy)):
                errors.append(
                    f"Layer {layer_id}, logical expert {logical_id}: duplicate physical experts in log2phy"
                )
                stats['duplicate_physical_experts'] += 1

            # Check consistency: phy2log should match log2phy
            for rep_rank, phys_id in enumerate(physical_ids_in_log2phy):
                if phy2log[layer_id, phys_id].item() != logical_id:
                    errors.append(
                        f"Layer {layer_id}, logical expert {logical_id}, replica {rep_rank}: "
                        f"Inconsistency: log2phy says physical expert {phys_id} maps to logical {logical_id}, "
                        f"but phy2log says physical {phys_id} maps to logical {phy2log[layer_id, phys_id].item()}"
                    )
                    stats['inconsistency_errors'] += 1

        # Check 6: All physical experts should be used (in log2phy)
        unused_physical = set(range(num_replicas)) - physical_experts_used
        if unused_physical:
            errors.append(
                f"Layer {layer_id}: Physical experts not used in log2phy: {sorted(unused_physical)}"
            )
            stats['missing_physical_experts'] += len(unused_physical)

        # Check 7: Check that unused slots in log2phy are marked with -1
        for logical_id in range(num_logical_experts):
            num_reps = int(logcnt[layer_id, logical_id].item())
            for rep_rank in range(num_reps, max_replicas):
                if log2phy[layer_id, logical_id, rep_rank].item() != -1:
                    # This is a warning, not necessarily an error, but we should check
                    # Actually, if it's not -1 and not in the valid range, that's an error
                    val = log2phy[layer_id, logical_id, rep_rank].item()
                    if val >= 0:
                        errors.append(
                            f"Layer {layer_id}, logical expert {logical_id}, replica {rep_rank}: "
                            f"Expected -1 for unused slot, got {val}"
                        )
                        stats['invalid_log2phy_entries'] += 1

    is_valid = len(errors) == 0
    return is_valid, errors, stats

def simulate_inference(log2phy: torch.Tensor, logcnt: torch.Tensor, workload: torch.Tensor) -> float:
    '''
    Simulate a MoE inference with the given expert mapping, and return the balancedness factor.
    '''
    # workload 形状: (num_layers, num_logical_experts) - 每层每个逻辑专家的负载
    num_layers, num_logical_experts = workload.shape
    
    # 初始化物理专家负载累积器
    num_physical_experts = NUM_REPLICAS
    total_physical_load = torch.zeros(num_layers, num_physical_experts, dtype=torch.float, device=workload.device)
    
    # 对每个逻辑专家，分配负载到其物理副本
    for layer_id in range(num_layers):
        for logical_id in range(num_logical_experts):
            # 获取该逻辑专家的负载
            logical_load = workload[layer_id][logical_id].item()
            
            # 跳过零负载
            if logical_load <= 0:
                continue
                
            num_replicas = int(logcnt[layer_id][logical_id].item())

            # 跳过零副本
            if num_replicas <= 0:
                continue

            # 获取物理专家映射
            physical_ids = log2phy[layer_id][logical_id][:num_replicas]
                
            # 计算每个副本的负载（基于有效副本数量）
            replica_load = logical_load / num_replicas
            
            # 分配负载到有效的物理专家
            total_physical_load[layer_id, physical_ids] += replica_load
    
    # 计算 balancedness
    total_load = total_physical_load.sum()
    if total_load == 0:
        return 0.0
    
    # 计算每层的平均负载和最大负载，然后求和
    layer_avg = total_physical_load.mean(dim=1)  # (num_layers,)
    layer_max = total_physical_load.max(dim=1).values  # (num_layers,)
    
    avg_load = layer_avg.sum().item()
    max_load = layer_max.sum().item()
    
    # 计算 balancedness: avg_load / max_load
    balancedness = avg_load / max_load if max_load > 0 else 0.0
    
    print(f'balancedness: {balancedness}')
    
    return balancedness

def evaluate(program_path: str) -> EvaluationResult:
    workloads = load_workloads(WORKLOAD_PATH)

    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        assert spec is not None
        program = importlib.util.module_from_spec(spec)
        assert spec.loader is not None

        # Inject commonly used imports into the program namespace BEFORE execution
        # This allows type hints like torch.Tensor to work
        program.torch = torch
        program.np = np
        program.os = os
        program.re = re
        program.inspect = inspect

        # Now execute the module
        spec.loader.exec_module(program)

        if not hasattr(program, "rebalance_experts"):
            print('Error: program does not have `rebalance_experts` function')
            return {
                "balancedness_score": 0.0,
                "speed_score": 0.0,
                "combined_score": 0.0,
                "error": "Missing `rebalance_experts` function",
                "success": False,
                "runs_successfully": 0.0,
            }

        if not hasattr(program, "rebalance_experts"):
            raise ValueError("Program does not have rebalance_experts function")
        
        balancedness_scores = []
        times = []

        for i in range(len(workloads) - 1):
            start_time = time.perf_counter()
            try:
                phy2log, log2phy, logcnt = program.rebalance_experts(
                    workloads[i],
                    NUM_REPLICAS,
                    NUM_GROUPS,
                    NUM_NODES,
                    NUM_GPUS,
                )
            except (IndexError, RuntimeError) as e:
                error_msg = str(e)
                if "out of bounds" in error_msg or "IndexError" in str(type(e)):
                    raise ValueError(
                        f"IndexError during execution (workload {i}): {error_msg}. "
                        f"This usually means expert_count doesn't sum to {NUM_REPLICAS} or "
                        f"physical_idx exceeds valid range [0, {NUM_REPLICAS-1}]. "
                        f"Check that expert_count sums exactly to {NUM_REPLICAS} for each layer."
                    ) from e
                else:
                    raise
            end_time = time.perf_counter()

            # Validate the solution
            num_layers, num_logical_experts = workloads[i].shape
            is_valid, errors, stats = validate_rebalance_solution(
                phy2log, log2phy, logcnt,
                num_layers, num_logical_experts, NUM_REPLICAS
            )

            if not is_valid:
                error_msg = f"Validation failed for workload {i}: " + "; ".join(errors[:3])
                print(f"⚠️  VALIDATION FAILED for workload {i}:")
                print(f"  Shape errors: {stats['shape_errors']}")
                print(f"  Index range errors: {stats['index_range_errors']}")
                print(f"  Missing physical experts: {stats['missing_physical_experts']}")
                print(f"  Duplicate physical experts: {stats['duplicate_physical_experts']}")
                print(f"  Expert count mismatches: {stats['expert_count_mismatch']}")
                print(f"  Invalid log2phy entries: {stats['invalid_log2phy_entries']}")
                print(f"  Inconsistency errors: {stats['inconsistency_errors']}")
                print(f"  First few errors: {errors[:3]}")
                # Fail fast on validation errors
                raise ValueError(f"Solution validation failed: {error_msg}")

            balancedness_score = simulate_inference(log2phy, logcnt, workloads[i + 1])
            balancedness_scores.append(balancedness_score)
            times.append(end_time - start_time)
        avg_balancedness_score = sum(balancedness_scores) / len(balancedness_scores)
        avg_time = sum(times) / len(times)
        speed_score = 0.02 / avg_time
        print(f'avg_time: {avg_time}, speed_score: {speed_score}')
        print(f'✅ All solutions validated successfully')
        combined_score = avg_balancedness_score #(avg_balancedness_score + speed_score) / 2
        return {
            "balancedness_score": float(avg_balancedness_score),
            "speed_score": float(speed_score),
            "combined_score": float(combined_score),
            "success": True,
            "runs_successfully": 1.0,
        }
    except Exception as e:
        traceback.print_exc()
        print(f'Error during evaluation: {str(e)}')
        return {
            "balancedness_score": 0.0,
            "speed_score": 0.0,
            "combined_score": 0.0,
            "error": str(e),
            "success": False,
            "runs_successfully": 0.0,
        }
    
    return {
        "balancedness_score": 0.0,
        "speed_score": 0.0,
        "combined_score": 0.0,
        "error": "No error",
        "success": False,
        "runs_successfully": 0.0,
    }
    