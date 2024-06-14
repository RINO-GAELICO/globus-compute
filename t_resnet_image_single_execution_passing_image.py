from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk import Client
from globus_compute_sdk import Executor
from PIL import Image
import json
import datetime
from dotenv import load_dotenv
import os

# ENV_PATH = "./globus_torch.env"
ENV_PATH = "./globus_torch_container.env"
# ENV_PATH = "./resnet_venv.env"
load_dotenv(dotenv_path=ENV_PATH)

c= Client(code_serialization_strategy=CombinedCode())


# TODO time function


def infer_image(input_image, categories_str):

    import torch
    from torchvision import transforms
    import io
    from PIL import Image

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
    top3_prob, top5_catid = torch.topk(probabilities, 3)
    results = [(categories[top5_catid[i]], top3_prob[i].item()) for i in range(top3_prob.size(0))]

    return results



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
    image_bytes = image_to_bytes(image_path)
    
    input_image = Image.open(image_path)
    # original single execution
    future = gce.submit(infer_image, input_image, categories_str)

    result = future.result()
        
    print(result)
   


