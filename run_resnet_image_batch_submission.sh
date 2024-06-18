#!/bin/bash

# first run the register_functions.py script to register the functions
python3 register_func_resnet.py

# Loop through the arguments 4, 8, 12
for num_functions in 4 8 12
do
    # Run the python script with the argument
    python3 launch_resnet.py $num_functions
    
    # Wait for the command to complete before proceeding to the next one
    wait
done
