from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk import Client
from globus_compute_sdk import Executor
from dotenv import load_dotenv
import concurrent.futures
import json
import datetime
import os
import torch
import sys
from time import perf_counter

NUM_ITERATIONS = 5

# If no arguments are provided, it will raise an error
if len(sys.argv) < 3:
    raise ValueError("Please provide the number of functions to run and the endpoint name")

# Number of functions to run
NUMBER_OF_FUNCTIONS = int(sys.argv[1])

# if no number of functions is provided, it will raise an error
if NUMBER_OF_FUNCTIONS is None:
    raise ValueError("Please provide the number of functions to run")

ENDPOINT_NAME = sys.argv[2]

ENV_PATH = "./" + ENDPOINT_NAME + ".env"

# if the path is not correct, it will raise an error
if not os.path.exists(ENV_PATH):
    raise FileNotFoundError(f"File {ENV_PATH} not found")
load_dotenv(dotenv_path=ENV_PATH)

c = Client(code_serialization_strategy=CombinedCode())


# FUNCTION TO RUN
def infer_image(image_path, func_id):
    
    from time import perf_counter
    import torch
    # Start timing
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t1_start = perf_counter()
    starter.record()
    
 
    from PIL import Image
    from torchvision import transforms
    from pathlib import Path

    
    # THIS CODE IS JUST TEMPORARY TO CHECK IN WHICH NODE THE JOB IS RUNNING
    # # # # # # # # # # # # # # # #
    import os
    node_name = os.getenv('SLURMD_NODENAME')
    # # # # # # # # # # # # # # # #

    
    # check if a file exists at the path using pathlib
    if not Path(image_path).is_file():
        return f"File {image_path} not found"
    
    input_image = Image.open(image_path)

    
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
    t1_stop = perf_counter()
    ender.record()
    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    execution_time = t1_stop-t1_start
    

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
        "start_time": t1_start,
        "end_time": t1_stop,
        "environment": environment,
        "func_id": func_id,
        "node_name": node_name,
        "time_execution_cuda": curr_time
    }
    
    
    
        
    
def read_file_to_string(file_path):
    with open(file_path, "r") as file:
        return file.read()

perlmutter_endpoint = os.getenv("ENDPOINT_ID")
# Read categories file to string
categories_file_path = 'imagenet_classes.txt'
categories_str = read_file_to_string(categories_file_path)

# # # ... then create the executor, ...
with Executor(endpoint_id=perlmutter_endpoint, funcx_client=c) as gce:
     
    futures_addresses = []
    submission_times = {}
    completion_times = {}
    results = []
    
    
    default_image_path = "images/0.jpg"
    # start with a warm up function
    warm_up_future = gce.submit(infer_image, default_image_path, -1)
    warm_up_result = warm_up_future.result()
    print(f"First warm up function completed at {datetime.datetime.now()}")
    
    all_results = {}
    all_throughputs_results = {}
    
    for iteration in range(NUM_ITERATIONS):
        # start timing for throughput
        t_0 = perf_counter()
        for i in range(NUMBER_OF_FUNCTIONS):
            image_path = f"images/{i}.jpg"
            submission_time = perf_counter()
            future = gce.submit(infer_image, image_path, i)
            futures_addresses.append(future)
            submission_times.update({i: submission_time})
        
        # Get the results and record completion times
        for future in concurrent.futures.as_completed(futures_addresses):
            result = future.result()
            completion_time = perf_counter()
            results.append(result)
            # print(f"Result: {result}")
            # print(f"Future {result['func_id']} completed at {completion_time}")``
            completion_times.update({result['func_id']: completion_time})
        print(f"Iteration {iteration+1} completed at {datetime.datetime.now()}")
        # stop timing for throughput
        t_n = perf_counter()
        
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
                "submission_time": submission_times[result['func_id']],
                "completion_time": completion_times[result['func_id']],
                "duration_completion": diff_time,
                "node_name": result['node_name'],
                "time_execution_cuda": result['time_execution_cuda'],
                
            }
            formatted_results.append(formatted_result)
            
        print("Saving results of iteration {}".format(iteration+1))
        # Store the results for this iteration
        all_results[iteration] = dict_results
        
        throughput = NUMBER_OF_FUNCTIONS / (t_n - t_0)
        print(f"Throughput: {throughput} functions per second")
        throughputs_results = {
            "throughput": throughput,
            "start_time": t_0,
            "end_time": t_n
        }
        
        all_throughputs_results[iteration] = throughputs_results
        
        # # Print all results
        # for i in range(NUMBER_OF_FUNCTIONS):
        #     print(f"Future {i+1}: Submitted at {submission_times[i]}, Completed at {completion_times[i]}, Node used: {results[i]['node_name']}, Result: {formatted_results[i]}")
        

    output_file_name_functions_resutls = "./results_throughput/4_node_results_pytorch_concurrent_{}_{}_64_proc.json".format(NUMBER_OF_FUNCTIONS, ENDPOINT_NAME)
    with open(output_file_name_functions_resutls, "w") as f:
        json.dump(all_results, f)
    output_file_name_throughput = "./results_throughput/4_node_throughput_pytorch_concurrent_{}_{}_64_proc.json".format(NUMBER_OF_FUNCTIONS, ENDPOINT_NAME)
    with open(output_file_name_throughput, "w") as f:
        json.dump(all_throughputs_results, f)
    
    print("All results saved to file: {}".format(output_file_name_functions_resutls))
    print("All throughputs saved to file: {}".format(output_file_name_throughput))


