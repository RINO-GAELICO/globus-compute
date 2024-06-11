from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk import Client
from globus_compute_sdk import Executor

c= Client(code_serialization_strategy=CombinedCode())

def add_func(a, b):
    return a * b

perlmutter_endpoint = 'a0d768b4-994b-46b3-a6ba-3fd62bcb66cb' 
# ... then create the executor, ...
with Executor(endpoint_id=perlmutter_endpoint, funcx_client=c) as gce:
    # ... then submit for execution, ...
    future = gce.submit(add_func, 5, 10)

    # ... and finally, wait for the result
    print(future.result())
