from globus_compute_sdk import Executor

# First, define the function ...
def add_func(a, b):
    return a + b

tutorial_endpoint_id = '4b116d3c-1703-4f8f-9f6f-39921e5864df' # Public tutorial endpoint
# ... then create the executor, ...
with Executor(endpoint_id=tutorial_endpoint_id) as gce:
    # ... then submit for execution, ...
    future = gce.submit(add_func, 5, 10)

    # ... and finally, wait for the result
    print(future.result())