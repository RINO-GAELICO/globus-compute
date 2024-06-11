
from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk import Client

c= Client(code_serialization_strategy=CombinedCode())
from globus_compute_sdk import Executor

def add_func(a, b):
    return a * b

tutorial_endpoint_id = '199d0b45-5243-4301-a28c-118b3f3f0e6d' # Public tutorial endpoint
# ... then create the executor, ...
with Executor(endpoint_id=tutorial_endpoint_id, funcx_client=c) as gce:
    # ... then submit for execution, ...
    future = gce.submit(add_func, 5, 10)

    # ... and finally, wait for the result
    print(future.result())