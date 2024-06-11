from globus_compute_sdk import Client
from dotenv import load_dotenv
import json
import os


gc = Client()

ENV_PATH = "./resnet.env"
load_dotenv(dotenv_path=ENV_PATH)


def run_batch(function, nbatch=1):
   
    function_id = os.getenv(function)
    endpoint_id = os.getenv("ENDPOINT_ID")

    batch = gc.create_batch()

    for _i in range(nbatch):
        batch.add(function_id=function_id)
 
    batch_ret = gc.batch_run(endpoint_id,batch=batch)
    with open(f"resnet_batch.json","w") as f:
        json.dump(batch_ret,f)
    
    
    return batch_ret


    
if __name__ == '__main__':

    run_batch("RESNET")
    

