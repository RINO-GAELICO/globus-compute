from globus_compute_sdk import Client
from dotenv import load_dotenv

gc = Client()

ENV_PATH = "./resnet.env"
load_dotenv(dotenv_path=ENV_PATH)

def resnet_python():
    import subprocess
    command = f"python $SCRATCH/conda/globus-torch/resnet.py"
    res = subprocess.run(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return res


def register_functions(functions={"RESNET":resnet_python}):
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
