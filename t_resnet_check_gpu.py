from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk import Client
from globus_compute_sdk import Executor
import json
from dotenv import load_dotenv
import os
import sys
import concurrent.futures


ENDPOINT_NAME = sys.argv[1]

ENV_PATH = "./" + ENDPOINT_NAME + ".env"

# if the path is not correct, it will raise an error
if not os.path.exists(ENV_PATH):
    raise FileNotFoundError(f"File {ENV_PATH} not found")
load_dotenv(dotenv_path=ENV_PATH)

c= Client(code_serialization_strategy=CombinedCode())

def check_gpu():
    
   
    print("Checking GPU...", flush=True)
    import subprocess
    
    try:
        result = subprocess.run(['ps', '-ef'], stdout=subprocess.PIPE, text=True, check=True)
        mps_check = subprocess.run(['grep', 'mps'], input=result.stdout, stdout=subprocess.PIPE, text=True, check=True)
        
        mps_output = mps_check.stdout.strip()
        if mps_output:
            header = "MPS Server Processes:\n"
            formatted_output = header + mps_output.replace('\n', '\n')
        else:
            formatted_output = "No MPS server processes found."
        
        print(formatted_output, flush=True)
        return formatted_output
    except subprocess.CalledProcessError as e:
        error_message = f"Error checking MPS: {e}"
        print(error_message, flush=True)
        return error_message
    
#  run directory /pscratch/sd/d/duccio/ionorb/batch_shot_163303/100
def ionorb_wrapper(run_directory, bin_path, config_path="ionorb_stl2d_boris.config", outfile="out.hits.els.txt"):
    import subprocess, os, time, shutil, glob

    start = time.time()
    os.chdir(run_directory)

    if len(glob.glob("*.stl")+glob.glob("*.STL")) == 0:
        stl_files = glob.glob(os.path.join(bin_path,"*.stl"))+glob.glob(os.path.join(bin_path,"*.STL"))
        for stl_file in stl_files:
            stl_file_name = stl_file.split("/")[-1]
            os.symlink(stl_file,os.path.join(run_directory,stl_file_name))

    command = f"ionorb_stl_boris2d {config_path}"
    res = subprocess.run(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    end = time.time()
    runtime = end - start

    if res.returncode != 0:
        raise Exception(f"Application failed with non-zero return code: {res.returncode} stdout='{res.stdout.decode('utf-8')}' stderr='{res.stderr.decode('utf-8')}' runtime={runtime}")
    else:
        try:
            shutil.copyfile(outfile,os.path.join(run_directory,"outputs",outfile))
        except:
            os.makedirs(os.path.join(run_directory,"outputs"))
            shutil.copyfile(outfile,os.path.join(run_directory,"outputs",outfile))
        return res.returncode, res.stdout.decode("utf-8"), res.stderr.decode("utf-8"), runtime


perlmutter_endpoint = os.getenv("ENDPOINT_ID")
# ... then create the executor, ...
with Executor(endpoint_id=perlmutter_endpoint, funcx_client=c) as gce:
    futures_addresses = []
    for i in range(1):
        future = gce.submit(ionorb_wrapper, "/pscratch/sd/d/duccio/ionorb/batch_shot_163303/100")
        futures_addresses.append(future)
    results = []
    for future in concurrent.futures.as_completed(futures_addresses):
        result = future.result()
        results.append(result)

    print(results)

