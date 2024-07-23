from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk import Client
from globus_compute_sdk import Executor
import json
from dotenv import load_dotenv
import os
import sys
import concurrent.futures
from time import perf_counter

ENDPOINT_NAME = sys.argv[1]

ENV_PATH = "./" + ENDPOINT_NAME + ".env"

# Number of functions to run from the second argument of the command line
NUM_FUNCTIONS = int(sys.argv[2])
NUM_ITERATIONS = 1

# if the path is not correct, it will raise an error
if not os.path.exists(ENV_PATH):
    raise FileNotFoundError(f"File {ENV_PATH} not found")
load_dotenv(dotenv_path=ENV_PATH)

c= Client(code_serialization_strategy=CombinedCode())
    
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

    command = f"/pscratch/sd/d/duccio/ionorb/ionorb_stl_boris2d {config_path}"
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
    all_throughputs_results = {}
    all_results = {}
    
    print("Starting warm up", flush=True)
    # warm up
    warm_up_future = gce.submit(ionorb_wrapper, "/pscratch/sd/d/duccio/ionorb/batch_shot_163303/0000", '/pscratch/sd/d/duccio/ionorb/ionorb_stl_boris2d')
    warm_up_result = warm_up_future.result()
    
    print("Warm up completed", flush=True)
    
    for iteration in range(NUM_ITERATIONS):
    
        print(f"Submitting functions for iteration {iteration}", flush=True)
        futures_addresses = []
        bin_path = '/pscratch/sd/d/duccio/ionorb/ionorb_stl_boris2d'
        # start timing for throughput
        t_0 = perf_counter()
        for i in range(NUM_FUNCTIONS+1):
            directory_path = f"/pscratch/sd/d/duccio/ionorb/batch_shot_163303/{str(i).zfill(4)}"
            future = gce.submit(ionorb_wrapper, directory_path, bin_path)
            futures_addresses.append(future)
        results = []
        for future in concurrent.futures.as_completed(futures_addresses):
            result = future.result()
            results.append(result)
        t_n = perf_counter()
        
        all_results[iteration] = results
        
        # THROUGHPUT CALC
        throughput = NUM_FUNCTIONS / (t_n - t_0)
        print(f"Throughput: {throughput} functions per second", flush=True)
        throughputs_results = {
            "throughput": throughput,
            "start_time": t_0,
            "end_time": t_n
        }
        all_throughputs_results[iteration] = throughputs_results
        print(f"Iteration {iteration+1} completed")
        
        
    # save the results in a file
    output_file_name_functions_results = "./results_ionorb/2_node_results_ionorb_{}_{}_4_proc.json".format(NUM_FUNCTIONS, ENDPOINT_NAME)
    with open(output_file_name_functions_results, "w") as f:
        json.dump(all_results, f)
    # save the throughput results in a file
    output_file_name_throughput = "./results_ionorb/throughput/2_node_throughput_ionorb_{}_{}_4_proc.json".format(NUM_FUNCTIONS, ENDPOINT_NAME)
    with open(output_file_name_throughput, "w") as f:
        json.dump(all_throughputs_results, f)
    print("Results saved in {}".format(output_file_name_functions_results))
    print("Throughput results saved in {}".format(output_file_name_throughput))