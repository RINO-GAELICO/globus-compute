import os
import json
import pickle
import base64
import globus_sdk
from globus_sdk.scopes import AuthScopes
from dotenv import load_dotenv

from globus_compute_sdk import Client, Executor
from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk.sdk.login_manager import AuthorizerLoginManager
from globus_compute_sdk.sdk.login_manager.manager import ComputeScopeBuilder

ENV_PATH = "./globus_torch.env"
load_dotenv(dotenv_path=ENV_PATH)
endpoint_id_globus_torch = os.getenv("ENDPOINT_ID")
gc= Client(code_serialization_strategy=CombinedCode())
gce = Executor(endpoint_id=endpoint_id_globus_torch, client=gc)

import time

# function that estimates pi by placing points in a box
def pi(num_points):
    from random import random
    inside = 0   
    
    for i in range(num_points):
        x, y = random(), random()  # Drop a point randomly within the box.
        if x**2 + y**2 < 1:        # Count points within the circle.
            inside += 1  
    return (inside*4 / num_points)


# execute the function 100 times 
estimates = []
for i in range(100):
    estimates.append(gce.submit(pi, 
                               10**5))

# get the results and calculate the total
total = [future.result() for future in estimates]

# print the results
print("Estimates: {}".format(total))
print("Average: {:.5f}".format(sum(total)/len(estimates)))