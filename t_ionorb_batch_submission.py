from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk import Client
from globus_compute_sdk import Executor
import concurrent.futures
import json
import os
import sys
from time import perf_counter
from dotenv import load_dotenv


ENDPOINT_NAME = sys.argv[1]

ENV_PATH = "./" + ENDPOINT_NAME + ".env"

# if the path is not correct, it will raise an error
if not os.path.exists(ENV_PATH):
    raise FileNotFoundError(f"File {ENV_PATH} not found")
load_dotenv(dotenv_path=ENV_PATH)

# Number of functions to run from the second argument of the command line
NUM_FUNCTIONS = int(sys.argv[2])
NUM_ITERATIONS = 1

gcc = Client(code_serialization_strategy=CombinedCode())

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

print(f"Endpoint: {perlmutter_endpoint}", flush=True)

all_throughputs_results = {}
all_results = {}

# with Executor(endpoint_id=perlmutter_endpoint, funcx_client=gcc) as gce:
#     print("Starting warm up", flush=True)
#     # warm up
#     warm_up_future = gce.submit(ionorb_wrapper, "/pscratch/sd/d/duccio/ionorb/batch_shot_163303/0000", '/pscratch/sd/d/duccio/ionorb/ionorb_stl_boris2d')
#     warm_up_result = warm_up_future.result()
    
#     print("Warm up completed", flush=True)


# # warm up batch of one function
# batch = gcc.create_batch()
# bin_path = '/pscratch/sd/d/duccio/ionorb/ionorb_stl_boris2d'
# directory_path = f"/pscratch/sd/d/duccio/ionorb/batch_shot_163303/0000"
# batch.add(args=[directory_path, bin_path], function_id=gcc.register_function(ionorb_wrapper))
# # print the batch using the batch.prepare() method
# print(batch.prepare())
# batch_res = gcc.batch_run(endpoint_id=perlmutter_endpoint)

# futures_addresses_warm_up = []
# for task_id in batch_res:
#     future = gcc.get_result(task_id)
#     futures_addresses_warm_up.append(future)

# results_warm_up = []
# for future in concurrent.futures.as_completed(futures_addresses_warm_up):
#     result = future.result()
#     results_warm_up.append(result)

# print("Warm up completed", flush=True)
    
for iteration in range(NUM_ITERATIONS):

    batch = gcc.create_batch()
    bin_path = '/pscratch/sd/d/duccio/ionorb/ionorb_stl_boris2d'
    for x in range(NUM_FUNCTIONS+1):
        directory_path = f"/pscratch/sd/d/duccio/ionorb/batch_shot_163303/{str(x).zfill(4)}"
        batch.add(args=[directory_path, bin_path], function_id=gcc.register_function(ionorb_wrapper))
        
    # print the batch using the batch.prepare() method
    print(batch.prepare())

    futures_addresses = []
    results = []
    
    # start timing for throughput
    t_0 = perf_counter()
    # batch_run returns a list task ids
    batch_res = gcc.batch_run(endpoint_id=perlmutter_endpoint)

    for task_id in batch_res:
        future = gcc.get_result(task_id)
        futures_addresses.append(future)

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
    print(f"Iteration {0+1} completed")


# save the results in a file
output_file_name_functions_results = "./results_ionorb_batch/8_node_results_ionorb_{}_{}_64_proc.json".format(NUM_FUNCTIONS, ENDPOINT_NAME)
with open(output_file_name_functions_results, "w") as f:
    json.dump(all_results, f)
# save the throughput results in a file
output_file_name_throughput = "./results_ionorb_batch/throughput/8_node_throughput_ionorb_{}_{}_64_proc.json".format(NUM_FUNCTIONS, ENDPOINT_NAME)
with open(output_file_name_throughput, "w") as f:
    json.dump(all_throughputs_results, f)
print("Results saved in {}".format(output_file_name_functions_results))
print("Throughput results saved in {}".format(output_file_name_throughput))