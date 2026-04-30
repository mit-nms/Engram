class CustomGlobalScheduler(BaseGlobalScheduler):
    """
    Least Loaded Queue (LLQ) global scheduler.
    """

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
