from globus_compute_sdk import Executor
from dotenv import load_dotenv
import os

# env variables
ENV_PATH = "./globus_torch.env"
load_dotenv(dotenv_path=ENV_PATH)
perlmutter_endpoint = os.getenv("ENDPOINT_ID")

# test function
def double():
    return 9

# ... then create the executor, ...
with Executor(endpoint_id=perlmutter_endpoint) as gce:
    fut = gce.submit(double)

    print(fut.result())