from globus_compute_sdk import Client
from dotenv import load_dotenv'
import os

gcc = Client()

ENV_PATH = "./globus_torch.env"
load_dotenv(dotenv_path=ENV_PATH)
perlmutter_endpoint = os.getenv("ENDPOINT_ID")


def platform_func():
  import platform
  return platform.platform()

func_uuid = gcc.register_function(platform_func)

tutorial_endpoint = '4b116d3c-1703-4f8f-9f6f-39921e5864df' # Public tutorial endpoint
task_id = gcc.run(endpoint_id=tutorial_endpoint, function_id=func_uuid)

try:
  print(gcc.get_result(task_id))
except Exception as e:
  print("Exception: {}".format(e))
