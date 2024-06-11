from globus_compute_sdk import Client
gcc = Client()

def platform_func():
  import platform
  return platform.platform()

func_uuid = gcc.register_function(platform_func)

tutorial_endpoint = '199d0b45-5243-4301-a28c-118b3f3f0e6d'
task_id = gcc.run(endpoint_id=tutorial_endpoint, function_id=func_uuid)

try:
  print(gcc.get_result(task_id))
except Exception as e:
  print("Exception: {}".format(e))
