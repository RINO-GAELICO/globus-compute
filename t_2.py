from globus_compute_sdk import Executor


from globus_compute_sdk import Executor

def double():
    return 9

tutorial_endpoint_id = '199d0b45-5243-4301-a28c-118b3f3f0e6d'

with Executor(endpoint_id=tutorial_endpoint_id) as gce:
    fut = gce.submit(double)

    print(fut.result())