from globus_compute_sdk import Client
from dotenv import load_dotenv

gc = Client()

ENV_PATH = "./globus_torch_container.env"
load_dotenv(dotenv_path=ENV_PATH)

def infer_image(input_image, func_id):
    import time
    # Start timing
    start_time = time.time()
    from torchvision import transforms
    import torch

    
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


def register_functions(functions={"RESNET":infer_image}):
    function_ids = {}
    for envvar in functions:
        func = gc.register_function(functions[envvar])
        
        # delete the function if it already exists before writing to the file
        with open(ENV_PATH, "r") as f:
            lines = f.readlines()
        with open(ENV_PATH, "w") as f:
            for line in lines:
                if envvar not in line:
                    f.write(line)
               
        
        with open(ENV_PATH, "a") as f:
            f.write(f"\n{envvar}={func}\n")
        print(f"{envvar}={func}")
        function_ids[envvar] = func

    return function_ids

if __name__ == '__main__':
    
    register_functions()
