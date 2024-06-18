from globus_compute_sdk import Client
from dotenv import load_dotenv
import json
import os
from PIL import Image
import concurrent.futures
import datetime
import torch
import time
import sys

gcc = Client()

ENV_PATH = "./globus_torch_container.env"
load_dotenv(dotenv_path=ENV_PATH)

NUMBER_OF_TASKS = int(sys.argv[1])

image_path = 'dog.jpg'
input_image = Image.open(image_path)
    
# Read categories file to string
categories_file_path = 'imagenet_classes.txt'
with open(categories_file_path, "r") as file:
    categories_str = file.read()


futures_addresses = []
submission_times = {}
completion_times = {}
results = []

def run_batch(function, nbatch=NUMBER_OF_TASKS):
   
    function_id = os.getenv(function)
    endpoint_id = os.getenv("ENDPOINT_ID")

    batch = gcc.create_batch()

    for i in range(nbatch):
        batch.add(function_id=function_id, args=[input_image, i])
 
    batch_ret = gcc.batch_run(endpoint_id,batch=batch)
    with open(f"resnet_batch.json","w") as f:
        json.dump(batch_ret,f)
    
    submission_time = datetime.datetime.now()
    
    print(f"Batch submitted at {submission_time}")

    # THIS PART SHOULD EXECUTE ONLY AFTER ALL THE TASKS ARE COMPLETED SO WE SHOULD WAIT FOR ALL THE TASKS TO BE COMPLETED
    # check if the get_batch_result is returning the 'pending'=True or False, if it is False, then we can proceed with the next steps
    # if it is True, then we should wait for all the futures to be completed, therefore in this case we should 
    all_completed = False
    while True:
        results_batch = gcc.get_batch_result(batch_ret['tasks'][function_id])
        all_completed = all([results_batch[tid]["pending"] == False for tid in results_batch])
        if all_completed:
            break
        print("Waiting for all the tasks to be completed")
        time.sleep(7)
        

    completion_time = datetime.datetime.now()

    # Use the categories string to get the categories list
    categories = [s.strip() for s in categories_str.splitlines()]
    formatted_results = []
    dict_results = {}
    
    for tid in results_batch:
        result_task = gcc.get_result(tid)
        results.append(result_task)
        completion_times.update({result_task['func_id']: completion_time})
        submission_times.update({result_task['func_id']: submission_time})
        probabilities = torch.tensor(result_task['probabilities'])
        top3_prob, top3_catid = torch.topk(probabilities, 3)
        top3_results = [(categories[top3_catid[i]], top3_prob[i].item()) for i in range(top3_prob.size(0))]
        formatted_result = f"Function ID: {result_task['func_id']} \n Results: {top3_results} \n Execution Time: {result_task['time_execution']}\nEnvironment: {result_task['environment']} \n"
        diff_time = completion_times[result_task['func_id']] - submission_times[result_task['func_id']]
        dict_results[result_task['func_id']] = {
            "result": top3_results,
            "time_execution_function": result_task['time_execution'],
            "start_time": str(result_task['start_time']),
            "end_time": str(result_task['end_time']),
            "submission_time": str(submission_times[result_task['func_id']]),
            "completion_time": str(completion_times[result_task['func_id']]),
            "time_difference": str(diff_time),
            "environment": result_task['environment']
        }
        formatted_results.append(formatted_result)
        

    # Save the dictionary to a json file called results_pytorch_globus_compute_container_NUMBER_OF_FUNCTIONS.json
    with open("results_pytorch_globus_compute_container_batch_submission_" + str(nbatch) + ".json", "w") as outfile:
        json.dump(dict_results, outfile)
        
        return batch_ret


    
if __name__ == '__main__':

    run_batch("RESNET")
    
    
    

