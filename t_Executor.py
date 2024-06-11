
from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk import Client
from globus_compute_sdk import Executor
from dotenv import load_dotenv
import os

ENV_PATH = "./globus_torch.env"
load_dotenv(dotenv_path=ENV_PATH)
perlmutter_endpoint = os.getenv("ENDPOINT_ID")


c= Client(code_serialization_strategy=CombinedCode())



# test function to add two numbers
def add_func(a, b):
    return a * b

tutorial_endpoint_id = '4b116d3c-1703-4f8f-9f6f-39921e5864df' # Public tutorial endpoint
# ... then create the executor, ...
with Executor(endpoint_id=perlmutter_endpoint, funcx_client=c) as gce:
    # ... then submit for execution, ...
    future = gce.submit(add_func, 5, 10)

    # ... and finally, wait for the result
    print(future.result())