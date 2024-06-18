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

NUMBER_OF_FUNCTIONS = 2
ENV_PATH = "./globus_torch_container.env"
load_dotenv(dotenv_path=ENV_PATH)
gcc = Client(code_serialization_strategy=CombinedCode())

def infer_image(input_image, func_id):
    import time
    # Start timing
    start_time = time.time()
    from torchvision import transforms
    import torch
    import sys
    
    
    
    print(f"Size of the image sys.getsizeof(input_image): {sys.getsizeof(input_image)}",  flush=True)

    
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
        "func_id": func_id
    }

def image_to_bytes(image_path):
    with open(image_path, "rb") as image_file:
        return image_file.read()

def read_file_to_string(file_path):
    with open(file_path, "r") as file:
        return file.read()
    
perlmutter_endpoint = os.getenv("ENDPOINT_ID")

# Read categories file to string
categories_file_path = 'imagenet_classes.txt'
categories_str = read_file_to_string(categories_file_path)


image_path = 'dog.jpg'
input_image = Image.open(image_path)

futures_addresses = []
submission_times = {}
completion_times = {}
results = []

batch = gcc.create_batch()



for x in range(0, NUMBER_OF_FUNCTIONS):
    batch.add(args=[input_image, x], function_id=gcc.register_function(infer_image))
    
# print the batch using the batch.prepare() method
print(batch.prepare())


# batch_run returns a list task ids
batch_res = gcc.batch_run(endpoint_id=perlmutter_endpoint)
submission_time = datetime.datetime.now()


for task_id in batch_res:
    future = gcc.get_result(task_id)
    futures_addresses.append(future)
    submission_times.append(datetime.datetime.now())




for future in concurrent.futures.as_completed(futures_addresses):
    result = future.result()
    completion_time = datetime.datetime.now()
    results.append(result)
    print(f"Future {result['func_id']} completed at {completion_time}")
    completion_times.update({result['func_id']: completion_time})
    submission_times.update({result['func_id']: submission_time})
    
    
# Use the categories string to get the categories list
categories = [s.strip() for s in categories_str.splitlines()]

# Process results in the main thread
# create a dictionary with key is the function id and value is the result with all the information on time, environment, etc. contained as values
formatted_results = []
dict_results = {}
for result in results:
    probabilities = torch.tensor(result['probabilities'])
    top3_prob, top3_catid = torch.topk(probabilities, 3)
    top3_results = [(categories[top3_catid[i]], top3_prob[i].item()) for i in range(top3_prob.size(0))]
    
    formatted_result = f"Function ID: {result['func_id']} \n Results: {top3_results} \n Execution Time: {result['time_execution']}\nEnvironment: {result['environment']} \n"
    # save the result, time execution, time submission, time completion, environment to a dictionary
    # calculate the difference between the submission time and completion time
    diff_time = completion_times[result['func_id']] - submission_times[result['func_id']]

    # csave the times as strings
    dict_results[result['func_id']] = {
        "result": top3_results,
        "time_execution_function": result['time_execution'],
        "start_time": str(result['start_time']),
        "end_time": str(result['end_time']),
        "submission_time": str(submission_times[result['func_id']]),
        "completion_time": str(completion_times[result['func_id']]),
        "time_difference": str(diff_time),
        "environment": result['environment']
        
    }
    formatted_results.append(formatted_result)

# Print all results
for i in range(NUMBER_OF_FUNCTIONS):
    print(f"Future {i+1}: Submitted at {submission_times[i]}, Completed at {completion_times[i]}, Result: {formatted_results[i]}")

# Save the dictionary to a json file called results_pytorch_globus_compute_container_NUMBER_OF_FUNCTIONS.json
with open("results_pytorch_globus_compute_container_batch_submission" + str(NUMBER_OF_FUNCTIONS) + ".json", "w") as outfile:
    json.dump(dict_results, outfile)