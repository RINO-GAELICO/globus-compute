#!/bin/bash

# Loop through the arguments 4, 8, 12
for num_functions in 4 8 
do
    # Run the python script with the argument
    python3 t_resnet_image_concurrent_futures.py $num_functions
    
    # Wait for the command to complete before proceeding to the next one
    wait
done
