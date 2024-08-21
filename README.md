# Globus Compute at NERSC

This repo contains a series of experiments that have been executed during the summer of 2024.

Mainly concerning runtime and throughput, the experiments have deployed Python functions to an endpoint installed in the NERSC system.

Initial experiments were conducted using a ResNet PyTorch inference model that makes use of GPUs to classify images provided to it.

A second stage of experiments was conducted by executing a function that runs Ion Orbiter executables. This code simulates particle trajectories and determines their hit locations on the tokamak wall.

[View the full report](docs/report.pdf).

More information on how to use Globus Compute at NERSC can be found in NERSC docs.



