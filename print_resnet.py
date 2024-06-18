from globus_compute_sdk import Client
from dotenv import load_dotenv
import json
import os

gc = Client()

ENV_PATH = "./globus_torch.env"
load_dotenv(dotenv_path=ENV_PATH)
perlmutter_endpoint = os.getenv("ENDPOINT_ID")

def print_results(outfiles=["./resnet_batch.json"]):

    for outfile in outfiles:
        with open(outfile,"r") as f:
            
            batch_ret = json.load(f)

            function_id = os.getenv(f"RESNET")
            
            results_batch = gc.get_batch_result(batch_ret['tasks'][function_id])
            try:
                completion_times = [float(results_batch[tid]["completion_t"]) for tid in results_batch]
                # get_result(task_id)
                task_id = [tid for tid in results_batch][0]
                result_task = gc.get_result(task_id)
                print(result_task)
                print(gc.get_endpoint_status(perlmutter_endpoint))
            except KeyError:
                print(f"Functions in {outfile} have not completed")
                # print error message
                print(gc.get_endpoint_status(perlmutter_endpoint))
                continue
    print("Results batch: ", results_batch)
    print("Status: ", gc.get_endpoint_status(perlmutter_endpoint))
        

if __name__ == '__main__':
    
    print_results()
