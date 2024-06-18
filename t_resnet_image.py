from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk import Client
from globus_compute_sdk import Executor
from dotenv import load_dotenv
import concurrent.futures
import json
import datetime
import os

# ENV_PATH = "./globus_torch.env"
ENV_PATH = "./globus_torch_container.env"
# ENV_PATH = "./resnet_venv.env"
load_dotenv(dotenv_path=ENV_PATH)

c= Client(code_serialization_strategy=CombinedCode())


# TODO time function


def infer_image(image_path):
    
    # normally I would do :
    # import torch
    # from PIL import Image
    # from torchvision import transforms
    # But given the difficulty of installing torch and torchvision in the environment of execution, I will install them through the script calling a subproces
    
    
    # try:
    #     __import__('torch')
    # except ImportError:
    #     print(f"Installing 'torch'...")
    #     subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
    #     __import__('torch')
        
    # # let's do the same thing for torchvision and PIL
    # try:
    #     __import__('torchvision')
    # except ImportError:
    #     print(f"Installing 'torchvision'...")
    #     subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvision"])
    #     __import__('torchvision')
    
    # try:
    #     __import__('PIL')
    # except ImportError:
    #     print(f"Installing 'PIL'...")
    #     subprocess.check_call([sys.executable, "-m", "pip", "install", "PIL"])
    #     __import__('PIL')

    
    import sys
    import subprocess
    import os
    
    try:
        __import__('pathlib2')
    except ImportError:
        print(f"Installing 'pathlib2'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pathlib2"])
        __import__('pathlib2')
        
        
    
    
    import torch
    from PIL import Image
    from torchvision import transforms
    from pathlib import Path
    
    # # check if a file exists at the path
    # try:
    #     with open(image_path) as f:
    #         pass
    # except FileNotFoundError:
    #     # return the path of the pwd
    #     return f"File not found in this working directory. Current working directory: {os.getcwd()}"
    

    p = Path('.')
    
    results = []

    results.append(f"Current working directory: {p.cwd()}")
    # list subdirectories
    results.append(f"Subdirectories: {list(p.iterdir())}")
    # list python files
    # results.append(f"Python files: {list(p.glob('**/*.py'))}")
    
    return results
    

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
    input_image = Image.open(image_path)
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

    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # Get top 5 categories
    top3_prob, top5_catid = torch.topk(probabilities, 3)
    results = [(categories[top5_catid[i]], top3_prob[i].item()) for i in range(top3_prob.size(0))]

    return results


perlmutter_endpoint = os.getenv("ENDPOINT_ID")
# # # ... then create the executor, ...
with Executor(endpoint_id=perlmutter_endpoint, funcx_client=c) as gce:
    image_path = 'dog.jpg'
    
    
    # original single execution
    # future = gce.submit(infer_image, image_path)
    # print(gce.get_worker_hardware_details())
    # result = future.result()
    
    # print(result)
    
    # an attempt to run the function multiple times
    estimates = []
    submission_times = []

    for i in range(1):
        submission_time = datetime.datetime.now()
        future = gce.submit(infer_image, image_path)
        estimates.append(future)
        submission_times.append(submission_time)
    
    # Get the results and record completion times
    completion_times = []
    results = []

    for future in estimates:
        result = future.result()
        completion_time = datetime.datetime.now()
        results.append(result)
        completion_times.append(completion_time)    
        
    # Print the submission and completion times
    for i in range(1):
        print(f"Future {i+1}: Submitted at {submission_times[i]}, Completed at {completion_times[i]}, Result: {results[i]}")
        
    # get the results and calculate the total
    total_results = results

    # Print all results
    print("All results: {}".format(total_results))
   


