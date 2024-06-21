#!/bin/bash

# Loop from 1 to 5
for i in {1..5}
do
  # Run the command with the current value of i
  python3 t_resnet_image_concurrent_futures_passing_path.py $i globus-torch
done
