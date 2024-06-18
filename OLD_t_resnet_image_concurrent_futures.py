from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk import Client
from globus_compute_sdk import Executor
from dotenv import load_dotenv
from PIL import Image
import concurrent.futures
import json
import datetime
import os


NUMBER_OF_FUNCTIONS = 10

# ENV_PATH = "./globus_torch.env"
ENV_PATH = "./globus_torch_container.env"
# ENV_PATH = "./resnet_venv.env"
load_dotenv(dotenv_path=ENV_PATH)

c= Client(code_serialization_strategy=CombinedCode())

# TODO time function


def infer_image(input_image, categories_str, func_id):
    
    import time

    # Start timing
    start_time = time.time()

    import torch
    from torchvision import transforms

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
    
    # input_image = Image.open(io.BytesIO(image_bytes))
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

   # Use the categories string to get the categories list
    categories = [s.strip() for s in categories_str.splitlines()]

    # Get top 5 categories
    top3_prob, top3_catid = torch.topk(probabilities, 3)
    
    # We should do this part once we are back in the main thread
    results = [(categories[top3_catid[i]], top3_prob[i].item()) for i in range(top3_prob.size(0))]

    # End timing
    end_time = time.time()
    execution_time = end_time - start_time

    # Gather environment information
    environment = {
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_name": "resnet18"
    }

    return {
        "results": results,
        "time_execution": execution_time,
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
# # # ... then create the executor, ...

# Read categories file to string
categories_file_path = 'imagenet_classes.txt'
categories_str = read_file_to_string(categories_file_path)

with Executor(endpoint_id=perlmutter_endpoint, funcx_client=c) as gce:
    
    image_path = 'dog.jpg'
    
    input_image = Image.open(image_path)
    
    
    # an attempt to run the function multiple times
    futures_addresses = []
    submission_times = []

    for i in range(NUMBER_OF_FUNCTIONS):
        submission_time = datetime.datetime.now()
        future = gce.submit(infer_image, input_image, categories_str, i)
        futures_addresses.append(future)
        submission_times.append(submission_time)
    
    # Get the results and record completion times
    completion_times = []
    results = []

    for future in concurrent.futures.as_completed(futures_addresses):
        result = future.result()
        completion_time = datetime.datetime.now()
        results.append(result)
        completion_times.append(completion_time)    
        
    # Print the submission and completion times
    for i in range(NUMBER_OF_FUNCTIONS):
        print(f"Future {i+1}: Submitted at {submission_times[i]}, Completed at {completion_times[i]}, Result: {results[i]}")
        
    # get the results and calculate the total
    total_results = results

    # format the results
    formatted_results = []
    for result in total_results:
        formatted_results.append(f"Function ID: {result['func_id']} \n Results: {result['results']} \n Execution Time: {result['time_execution']}\nEnvironment: {result['environment']} \n ")
        
    # Print all results
    print("All results: {}".format(formatted_results))
   


