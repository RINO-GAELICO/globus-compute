from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk import Client
from globus_compute_sdk import Executor
import json
from dotenv import load_dotenv
import os

ENV_PATH = "./globus_torch.env"
load_dotenv(dotenv_path=ENV_PATH)

c= Client(code_serialization_strategy=CombinedCode())


def infer_image(image_path):
    
    import torch
    from PIL import Image
    from torchvision import transforms
    
    return image_path
    
    # # check if a file exists at the path
    # try:
    #     with open(image_path) as f:
    #         pass
    # except FileNotFoundError:
    #     return f"File not found at path: {image_path}"
    
    # # if the file exists, return success
    # return f"File found at path: {image_path}"

    # # Load the model
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # model.eval()

    # # Preprocess the image
    # preprocess = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # input_image = Image.open(image_path)
    # input_tensor = preprocess(input_image)
    # input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # # Move the input and model to GPU for speed if available
    # if torch.cuda.is_available():
    #     input_batch = input_batch.to('cuda')
    #     model.to('cuda')

    # # Perform inference
    # with torch.no_grad():
    #     output = model(input_batch)

    # # Convert to probabilities
    # probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # # Read the categories
    # with open("imagenet_classes.txt", "r") as f:
    #     categories = [s.strip() for s in f.readlines()]

    # # Get top 5 categories
    # top5_prob, top5_catid = torch.topk(probabilities, 5)
    # results = [(categories[top5_catid[i]], top5_prob[i].item()) for i in range(top5_prob.size(0))]

    # return results


perlmutter_endpoint = os.getenv("ENDPOINT_ID")
# ... then create the executor, ...
with Executor(endpoint_id=perlmutter_endpoint, funcx_client=c) as gce:
    image_path = 'dog.jpg'  # Change to your image path
    # ... then submit for execution, ...
    future = gce.submit(infer_image, image_path)
    result = future.result()
    
    print(result)
    
    

