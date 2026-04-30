def get_class_dependencies(class_name: str) -> str:
    """Get the description for a class"""
    if class_name == "CustomGlobalScheduler":
        return '''
Your task is to implement a custom load balancer by inheriting from BaseGlobalScheduler.
To achieve this, you will implement the `schedule` function of this class.
This function is called everytime 1) a new request has arrived, or 2) a replica has finished a request.
To return the routing decisions, this function should return a list of tuples. Each tuple consists of 1) the id number of the replica to route to and 2) the request to be routed `(replica_id, request)`.
Note that routed requests should be popped from `self._request_queue`.
The signature of this class and the function can be observed from an example implementation of a round-robin scheduler.
Do not change the properties of requests or replicas.

```python
class CustomGlobalScheduler(BaseGlobalScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Optional initialization code

        # round-robin counter
        self._request_counter = 0

    def schedule(self) -> List[Tuple[int, Request]]:
        # You can sort unrouted requests in `self._request_queue` by `request._arrived_at`.
        self.sort_requests()

        # You must return a list of tuples to describe the mapping of requests.
        request_mapping = []
        while self._request_queue:
            request = self._request_queue.pop(0)

            # increment counter and correct overflows
            replica_id = self._request_counter % self._num_replicas
            self._request_counter += 1

            # assign request to replica at counter
            request_mapping.append((replica_id, request))

        return request_mapping
```

Here is how you can access some helpful metrics to guide the decision-making process.

1. The CustomGlobalScheduler you will implement can see the following elements from a parent class (BaseGlobalScheduler):
   ```python
   _request_queue: List[Request]                               # queue of pending requests waiting to be routed
   _num_replicas: int                                          # number of available replicas with IDs `0` to `self._num_replicas-1`
   _replica_schedulers: Dict[int, ReplicaScheduler]            # maps replica IDs to replica objects
   ```

2. Each replica in `_replica_schedulers` is a `ReplicaScheduler` object, and has the following READ-ONLY properties:
   ```python
   replica_id: int                                             # unique replica id
   memory_usage_percent: float                                 # percentage of memory allocated to requests currently being processed
   num_blocks: int                                             # number of total blocks available for allocation
   num_allocated_blocks: int                                   # number of blocks allocated to currently running requests
   block_size: int                                             # Number of tokens per each allocated block
   pending_queue: List[Request]                                # list of requests that have not yet begun processing
   active_queue: List[Request]                                 # list of requests that are currently being processed
   ```

3. Each request has the following READ-ONLY properties:
   ```python
   arrived_at: float                                           # arrival time
   num_prefill_tokens: int                                     # number of prefill tokens
   num_processed_tokens: int                                   # number of tokens processed so far
   num_restarts: int                                           # number of times request was evicted
   ```

Now, implement the load balancer, i.e., CustomGlobalScheduler. Only output the full code for the CustomGlobalScheduler class.
'''

system_model = '''
Here is how the load balance works:

1. The load balancer manages a number (e.g., 16) of LLM serving node called `replica_scheduler`s.
   - The load balancer routes requests to any of these replicas.
   - The load balancer must eventually route all requests.

2. The load balancer makes routing decisions per each request. The load balancer knows these three key properties per request:
   - `_arrived_at`: When the request was received at the load balancer.
   - `_num_prefill_tokens`: number of tokens to prefill.
   - `num_processed_tokens`: number of tokens that have been processed so far.
   - A request has some number of decode tokens but this is not known until the request is completed.

3. Each replica maintains two queues: `pending_queue` and `active_queue`.
   - `pending_queue` contains requests where the prefill has not started. `num_processed_tokens` is 0 and there is no memory allocated in the GPU for these requests.
   - `active_queue` contains requests that are currently being processed. `num_processed_tokens > 0` and some memory is allocated in the GPU for these requests. If `num_processed_tokens < num_prefill_tokens`, the request is still in the prefill phase. Otherwise, the request is in the decode phase.
   - A request cannot be in both queues at the same time.

4. Each replica has to allocate memories for requests currently being processed, in the `active_queue`, in blocks (16 tokens at a time).
   - In case there is no memery left at a replica for continuing decoding of active requests, replicas will evict newer requests to free memory for earlier requests.
   - This removes the request from the 'active_queue', frees its allocated memory, resets its state, and adds it to the 'pending_queue'.
   - If the evicted request was in the decode phase, the 'num_prefill_tokens' is updated to 'num_processed_tokens'. 

5. The load balancer can observe the current state of all replicas, meaning all requests in `pending_queue` and `active_queue` of each replica.
'''