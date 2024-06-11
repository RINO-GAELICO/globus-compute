from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk import Client
from globus_compute_sdk import Executor
import json
from dotenv import load_dotenv
import os

ENV_PATH = "./globus_torch.env"
load_dotenv(dotenv_path=ENV_PATH)

c= Client(code_serialization_strategy=CombinedCode())

def check_environment():
    import sys
    import subprocess
    import os

    environment_details = {}

    # Get Python version
    python_version = sys.version
    environment_details["python_version"] = python_version

    # Get Conda environment details
    try:
        conda_list = subprocess.check_output(['conda', 'list']).decode('utf-8')
        environment_details["conda_list"] = conda_list
    except Exception as e:
        environment_details["conda_list_error"] = str(e)

    # Get pip installed packages
    try:
        pip_list = subprocess.check_output([sys.executable, '-m', 'pip', 'list']).decode('utf-8')
        environment_details["pip_list"] = pip_list
    except Exception as e:
        environment_details["pip_list_error"] = str(e)

    # Get system PATH
    path = sys.path
    environment_details["system_path"] = path
    
    # Get the system environment variables
    environment_details["system_env"] = dict(os.environ)
    
    # Get the current working directory
    environment_details["cwd"] = os.getcwd()
    

    return environment_details


def format_environment_details(env_details):
    formatted_details = []

    formatted_details.append(f"Worker SDK Version: 2.21.0")
    formatted_details.append(f"Worker OS: Linux-5.14.21-150400.24.81_12.0.87-cray_shasta_c-x86_64-with-glibc2.31")
    formatted_details.append(f"Python Version:\n{env_details['python_version']}\n")

    conda_list = env_details.get("conda_list", "No conda environment details found.")
    formatted_details.append(f"Conda Environment Packages:\n{conda_list}\n")

    pip_list = env_details.get("pip_list", "No pip installed packages found.")
    formatted_details.append(f"Pip Installed Packages:\n{pip_list}\n")

    system_path = env_details.get("system_path", "No system path found.")
    formatted_details.append(f"System PATH:\n{json.dumps(system_path, indent=2)}\n")
    
    environment_variables = env_details.get("system_env", "No system environment variables found.")
    formatted_details.append(f"System Environment Variables:\n{json.dumps(environment_variables, indent=2)}\n")
    
    cwd = env_details.get("cwd", "No current working directory found.")
    formatted_details.append(f"Current Working Directory:\n{cwd}\n")

    return "\n".join(formatted_details)

def save_environment_details_to_file(env_details, filename):
    formatted_env_details = format_environment_details(env_details)
    
    with open(filename, 'w') as file:
        file.write(formatted_env_details)
    
    print(f"Environment details saved to {filename}")


perlmutter_endpoint = os.getenv("ENDPOINT_ID")
# ... then create the executor, ...
with Executor(endpoint_id=perlmutter_endpoint, funcx_client=c) as gce:
    # ... then submit for execution, ...
    future = gce.submit(check_environment)
    
    save_environment_details_to_file(future.result(), "environment_details.txt")

