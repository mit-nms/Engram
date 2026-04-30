from typing import List, Tuple
import inspect
import re

from vidur.entities.request import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class CustomGlobalSchedulerV0(BaseGlobalScheduler):
    def schedule(self) -> List[Tuple[int, Request]]:
        from random import randint

        self.sort_requests()

        request_mapping = []
        while self._request_queue:
            request = self._request_queue.pop(0)
            replica_id = randint(1, self._num_replicas) - 1
            request_mapping.append((replica_id, request))
        return request_mapping


class CustomGlobalSchedulerV1(BaseGlobalScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._request_counter = 0

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        while self._request_queue:
            request = self._request_queue.pop(0)
            replica_id = self._request_counter % self._num_replicas
            self._request_counter += 1
            request_mapping.append((replica_id, request))

        return request_mapping


class CustomGlobalSchedulerV2(BaseGlobalScheduler):
    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        # keep a map of replica_id -> replica_scheduler
        # this is used to find the replica with the least loaded queue
        pending_requests_map = {
            replica_scheduler.replica_id: replica_scheduler.num_pending_requests + replica_scheduler.num_active_requests
            for replica_scheduler in self._replica_schedulers.values()
        }

        # using a very simple implementation here, to keep wiring simple
        while self._request_queue:
            request = self._request_queue.pop(0)
            replica_id = min(pending_requests_map.items(), key=lambda x: x[1])[0]
            pending_requests_map[replica_id] += 1
            request_mapping.append((replica_id, request))

        return request_mapping


class CustomGlobalSchedulerV3(BaseGlobalScheduler):
    @staticmethod
    def calculate_max_committed_blocks(request_queue, active_queue, block_size) -> int:
        from math import ceil

        max_active_blocks = 0
        for request in active_queue + request_queue:
            tokens_so_far = max(request._num_prefill_tokens, request.num_processed_tokens)
            blocks_so_far = ceil(tokens_so_far / block_size)
            max_active_blocks += blocks_so_far
        return max_active_blocks

    def schedule(self) -> List[Tuple[int, Request]]:
        from math import ceil
        from random import choice

        self.sort_requests()

        mem_usage = {}
        pending = {}
        active = {}
        max_committed_blocks = {}
        block_size = {}

        for rs_id, rs in self._replica_schedulers.items():
            mem_usage[rs_id] = rs.memory_usage_percent
            pending[rs_id] = rs.num_pending_requests
            active[rs_id] = rs.num_active_requests
            block_size[rs_id] = rs._config.block_size
            max_committed_blocks[rs_id] = self.calculate_max_committed_blocks(
                rs._request_queue, rs._active_queue, block_size[rs_id]
            )

        request_mapping = []
        self._request_queue.sort(key=lambda x: x.num_prefill_tokens)

        threshold = 2100

        while self._request_queue:
            if min(max_committed_blocks.values()) > threshold:
                break
            request = self._request_queue.pop(0)

            eligible_ids = [
                rs_id for rs_id in self._replica_schedulers.keys() if max_committed_blocks[rs_id] <= threshold
            ]
            assert len(eligible_ids) > 0
            costs = {rs_id: pending[rs_id] + active[rs_id] for rs_id in eligible_ids}
            min_cost = min(costs.values())
            candidates = [rs_id for rs_id, total in costs.items() if total == min_cost]
            min_replica_id = choice(candidates)

            max_committed_blocks[min_replica_id] += ceil(request._num_prefill_tokens / block_size[min_replica_id])
            pending[min_replica_id] += 1

            request_mapping.append((min_replica_id, request))

        return request_mapping


class CustomGlobalSchedulerV4(BaseGlobalScheduler):
    """
    CustomGlobalScheduler that prioritizes decode-phase to avoid evictions,
    balances load via composite scoring of memory, queue length, predicted load,
    free capacity, and decode concentration.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Weights for scoring: must sum appropriately to balance objectives
        self._w_mem = 0.3  # memory usage weight
        self._w_queue = 0.2  # queue length weight
        self._w_load = 0.3  # predicted load weight
        self._w_free = 0.1  # free capacity weight (negative in score)
        self._w_decode = 0.1  # decode-phase concentration weight

    def schedule(self) -> List[Tuple[int, Request]]:
        import math
        import time

        # Current time for arrival fallback
        now = time.time()
        # Determine block size (assumed consistent across replicas)
        block_size = 1
        for r in self._replica_schedulers.values():
            block_size = r.block_size
            break

        # Sort requests: decode-phase first, more restarts, older arrival, shorter remaining prefill
        def req_key(req: Request):
            phase = 0 if req.num_processed_tokens >= req.num_prefill_tokens else 1
            restarts = getattr(req, "num_restarts", 0)
            arrival = getattr(req, "_arrived_at", getattr(req, "arrived_at", now))
            remaining = max(0, req.num_prefill_tokens - req.num_processed_tokens)
            return (phase, -restarts, arrival, remaining)

        self._request_queue.sort(key=req_key)
        assignments: List[Tuple[int, Request]] = []

        # Try to route each pending request
        for req in list(self._request_queue):
            # Compute blocks needed: prefill + estimated decode
            remaining_prefill = max(0, req.num_prefill_tokens - req.num_processed_tokens)
            estimated_decode = getattr(req, "estimated_decode_tokens", req.num_prefill_tokens)
            total_tokens = remaining_prefill + estimated_decode
            needed_blocks = math.ceil(total_tokens / block_size)

            best_replica = None
            best_score = float("inf")

            for replica in self._replica_schedulers.values():
                free_blocks = replica.num_blocks - replica.num_allocated_blocks
                # Skip if not enough free memory blocks
                if free_blocks < needed_blocks:
                    continue
                # Immediately choose idle replicas
                if not replica.pending_queue and not replica.active_queue:
                    best_replica = replica
                    break

                # Compute metrics
                mem_frac = replica.memory_usage_percent / 100.0
                queue_frac = (len(replica.pending_queue) + len(replica.active_queue)) / max(1, replica.num_blocks)
                load_frac = (replica.num_allocated_blocks + needed_blocks) / replica.num_blocks
                free_frac = free_blocks / replica.num_blocks
                # Decode-phase concentration among active requests
                active = replica.active_queue
                if active:
                    decode_cnt = sum(1 for r2 in active if r2.num_processed_tokens >= r2.num_prefill_tokens)
                    decode_ratio = decode_cnt / len(active)
                else:
                    decode_ratio = 0.0

                # Composite score: lower is better
                score = (
                    self._w_mem * mem_frac
                    + self._w_queue * queue_frac
                    + self._w_load * load_frac
                    - self._w_free * free_frac
                    + self._w_decode * decode_ratio
                )

                if score < best_score:
                    best_score = score
                    best_replica = replica

            # Assign if a suitable replica is found
            if best_replica is not None:
                assignments.append((best_replica.replica_id, req))
                self._request_queue.remove(req)

        return assignments


class CustomGlobalSchedulerV5(BaseGlobalScheduler):
    """
    CustomGlobalScheduler balances the load to minimize average completion time by:
      1) Prioritizing decode-phase requests (to reduce evictions)
      2) Favoring previously evicted requests (higher num_restarts)
      3) Scheduling older requests sooner (reduce latency)
      4) Shortest remaining prefill first (reduce blocking)
    It assigns to idle replicas immediately, otherwise scores replicas by a composite metric
    of predicted load, queue length, memory usage, backlog, decode ratio, and boosts for age and restarts.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def schedule(self) -> List[Tuple[int, Request]]:
        import time, math

        now = time.time()
        assignments: List[Tuple[int, Request]] = []

        # Nothing to do if no replicas available
        if not self._replica_schedulers:
            return assignments

        # Sort pending requests by:
        #   1) decode-phase first
        #   2) more restarts first
        #   3) older arrival first
        #   4) shorter remaining prefill first
        def _req_key(r: Request):
            phase = 0 if r.num_processed_tokens >= r.num_prefill_tokens else 1
            restarts = getattr(r, "num_restarts", 0)
            arrival = getattr(r, "_arrived_at", getattr(r, "arrived_at", now))
            remaining = max(0, r.num_prefill_tokens - r.num_processed_tokens)
            return (phase, -restarts, arrival, remaining)

        self._request_queue.sort(key=_req_key)

        # Scoring weights
        w_load, w_queue, w_mem = 0.3, 0.2, 0.2
        w_backlog, w_decode = 0.1, 0.05
        w_age, w_restart = 0.1, 0.05
        max_age = 60.0
        max_restarts = 5.0

        # Attempt to assign each request
        for req in list(self._request_queue):
            remaining_prefill = max(0, req.num_prefill_tokens - req.num_processed_tokens)
            est_decode = req.num_prefill_tokens
            total_tokens = remaining_prefill + est_decode

            best_replica = None
            best_score = float("inf")

            for replica in self._replica_schedulers.values():
                free_blocks = replica.num_blocks - replica.num_allocated_blocks
                needed_blocks = math.ceil(total_tokens / max(1, replica.block_size))
                if free_blocks < needed_blocks:
                    continue

                # Immediate assignment to idle replicas
                if not replica.pending_queue and not replica.active_queue:
                    best_replica = replica
                    break

                # Compute composite metrics
                load_frac = (replica.num_allocated_blocks + needed_blocks) / max(1, replica.num_blocks)
                total_q = len(replica.pending_queue) + len(replica.active_queue)
                queue_frac = total_q / max(1, replica.num_blocks)
                mem_frac = replica.memory_usage_percent / 100.0

                backlog_tokens = sum(
                    max(0, r2.num_prefill_tokens - r2.num_processed_tokens)
                    for r2 in (replica.pending_queue + replica.active_queue)
                )
                backlog_frac = backlog_tokens / max(1, replica.num_blocks * replica.block_size)

                act = replica.active_queue
                if act:
                    dec_cnt = sum(1 for r2 in act if r2.num_processed_tokens >= r2.num_prefill_tokens)
                    decode_ratio = dec_cnt / len(act)
                else:
                    decode_ratio = 0.0

                arrival = getattr(req, "_arrived_at", getattr(req, "arrived_at", now))
                age_norm = min((now - arrival) / max_age, 1.0)
                rest_norm = min(getattr(req, "num_restarts", 0) / max_restarts, 1.0)

                score = (
                    w_load * load_frac
                    + w_queue * queue_frac
                    + w_mem * mem_frac
                    + w_backlog * backlog_frac
                    + w_decode * decode_ratio
                    - w_age * age_norm
                    - w_restart * rest_norm
                )
                if score < best_score:
                    best_score = score
                    best_replica = replica

            if best_replica:
                assignments.append((best_replica.replica_id, req))
                self._request_queue.remove(req)

        return assignments


class CustomGlobalSchedulerV6(BaseGlobalScheduler):
    def schedule(self) -> List[Tuple[int, Request]]:
        self._request_queue.sort(key=lambda r: (r.arrived_at, r.num_prefill_tokens))

        from math import ceil

        assignments: List[Tuple[int, Request]] = []

        # Attempt to assign each request
        for req in list(self._request_queue):
            total_tokens = 2 * req.num_prefill_tokens

            replica_score = {}

            for replica in self._replica_schedulers.values():
                free_blocks = replica.num_blocks - replica.num_allocated_blocks
                needed_blocks = ceil(total_tokens / max(1, replica.block_size))
                if free_blocks < needed_blocks:
                    continue

                # Compute allocated memory load + new request load
                load_frac = (replica.num_allocated_blocks + needed_blocks) / max(1, replica.num_blocks)

                # Compute memory backlog
                backlog_tokens = sum(
                    max(0, r2.num_prefill_tokens - r2.num_processed_tokens)
                    for r2 in (replica.pending_queue + replica.active_queue)
                )
                backlog_frac = backlog_tokens / max(1, replica.num_blocks * replica.block_size)

                # Compute queueing
                total_q = len(replica.pending_queue) + len(replica.active_queue)
                queue_frac = total_q / max(1, replica.num_blocks)

                # Compute decode ratio
                if replica.active_queue:
                    dec_cnt = sum(1 for r2 in replica.active_queue if r2.num_processed_tokens >= r2.num_prefill_tokens)
                    decode_ratio = dec_cnt / len(replica.active_queue)
                else:
                    decode_ratio = 0.0

                score = 0.5 * load_frac + 0.2 * queue_frac + 0.1 * backlog_frac + 0.05 * decode_ratio
                replica_score[replica.replica_id] = score

            if replica_score:
                best_rid = min(replica_score.keys(), key=lambda r: replica_score[r])
                assignments.append((best_rid, req))
                self._request_queue.remove(req)

        return assignments


_baselines_list = [
    # ("Random", CustomGlobalSchedulerV0),
    # ("Round Robin", CustomGlobalSchedulerV1),
    # ("Least Loaded Queue", CustomGlobalSchedulerV2),
    # ("Mohammad's Scheduler", CustomGlobalSchedulerV3),
    # ("o4-mini best shot iteration 39", CustomGlobalSchedulerV4),
    # ("o4-mini best shot", CustomGlobalSchedulerV5),
    # ("o4-mini best shot - cleaned", CustomGlobalSchedulerV6),
]
baselines_codes = [(lbl, inspect.getsource(foo)) for lbl, foo in _baselines_list]
baselines_codes = [
    (lbl, re.sub(r"class CustomGlobalSchedulerV[\d]{1,}\(", r"class CustomGlobalScheduler(", code))
    for lbl, code in baselines_codes
]
