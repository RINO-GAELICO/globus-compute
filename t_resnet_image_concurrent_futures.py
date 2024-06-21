from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk import Client
from globus_compute_sdk import Executor
from dotenv import load_dotenv
from PIL import Image
import concurrent.futures
import json
import datetime
import os
import torch
import sys

# Number of functions to run
NUMBER_OF_FUNCTIONS = int(sys.argv[1])

# if no number of functions is provided, it will raise an error
if NUMBER_OF_FUNCTIONS is None:
    raise ValueError("Please provide the number of functions to run")

# name of the endpoint from second argument
ENDPOINT_NAME = sys.argv[2]

# env path is "./{name from second argument}.env"
ENV_PATH = "./" + ENDPOINT_NAME + ".env"

# if the path is not correct, it will raise an error
if not os.path.exists(ENV_PATH):
    raise FileNotFoundError(f"File {ENV_PATH} not found")
load_dotenv(dotenv_path=ENV_PATH)


c = Client(code_serialization_strategy=CombinedCode())

def infer_image(input_image, func_id):
    import time
    # Start timing
    start_time = time.time()
    from torchvision import transforms
    import torch

    # THIS CODE IS JUST TEMPORARY TO CHECK IN WHICH NODE THE JOB IS RUNNING
    # # # # # # # # # # # # # # # #
    import os
    node_name = os.getenv('SLURMD_NODENAME')
    # # # # # # # # # # # # # # # #
    
    # Load the model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.eval()

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Convert to probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # End timing
    end_time = time.time()
    execution_time = end_time - start_time

    # Gather environment information
    environment = {
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_name": "resnet18"
    }

    # Return the raw output and execution metadata
    return {
        "probabilities": probabilities.tolist(),
        "time_execution": execution_time,
        "start_time": start_time,
        "end_time": end_time,
        "environment": environment,
        "func_id": func_id,
        "node_name": node_name
    }


def read_file_to_string(file_path):
    with open(file_path, "r") as file:
        return file.read()
    
perlmutter_endpoint = os.getenv("ENDPOINT_ID")

# Read categories file to string
categories_file_path = 'imagenet_classes.txt'
categories_str = read_file_to_string(categories_file_path)

with Executor(endpoint_id=perlmutter_endpoint, funcx_client=c) as gce:
    
    image_path = 'dog.jpg'
    input_image = Image.open(image_path)
    
    futures_addresses = []
    submission_times = {}
    completion_times = {}
    results = []

    for i in range(NUMBER_OF_FUNCTIONS):
        submission_time = datetime.datetime.now()
        future = gce.submit(infer_image, input_image, i)
        futures_addresses.append(future)
        submission_times.update({i: submission_time})
    
    # Get the results and record completion times
    for future in concurrent.futures.as_completed(futures_addresses):
        result = future.result()
        completion_time = datetime.datetime.now()
        results.append(result)
        print(f"Future {result['func_id']} completed at {completion_time}")
        completion_times.update({result['func_id']: completion_time})
        
    # Use the categories string to get the categories list
    categories = [s.strip() for s in categories_str.splitlines()]

    # Format the results
    formatted_results = []
    dict_results = {}
    for result in results:
        probabilities = torch.tensor(result['probabilities'])
        top3_prob, top3_catid = torch.topk(probabilities, 3)
        top3_results = [(categories[top3_catid[i]], top3_prob[i].item()) for i in range(top3_prob.size(0))]
        
        formatted_result = f"Function ID: {result['func_id']} \n Results: {top3_results} \n Execution Time: {result['time_execution']}\nEnvironment: {result['environment']} \n"
        
        # Calculate the time difference between submission and completion
        diff_time = completion_times[result['func_id']] - submission_times[result['func_id']]

        dict_results[result['func_id']] = {
            "result": top3_results,
            "time_execution_function": result['time_execution'],
            "start_time": str(result['start_time']),
            "end_time": str(result['end_time']),
            "submission_time": str(submission_times[result['func_id']]),
            "completion_time": str(completion_times[result['func_id']]),
            "duration_completion": str(diff_time),
            "environment": result['environment']
            
        }
        formatted_results.append(formatted_result)
    
    # Print all results
    for i in range(NUMBER_OF_FUNCTIONS):
        print(f"Future {i+1}: Submitted at {submission_times[i]}, Completed at {completion_times[i]}, Result: {formatted_results[i]}")
    
    # Save the dictionary to a json file called results_pytorch_globus_compute_container_NUMBER_OF_FUNCTIONS.json
    with open("results_pytorch_globus_compute_container_concurrent_" + str(NUMBER_OF_FUNCTIONS) + ".json", "w") as outfile:
        json.dump(dict_results, outfile)