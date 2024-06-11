from globus_compute_sdk import Executor

def double(x):
    return x * 2

tutorial_endpoint_id = '199d0b45-5243-4301-a28c-118b3f3f0e6d'
# Instantiate the Executor with amqp_port set to 443
with Executor(endpoint_id=tutorial_endpoint_id, amqp_port=443) as gce:
    fut = gce.submit(double, 7)

    print(fut.result())

