#!/bin/bash

# Loop over each line in params_homo.txt
while IFS= read -r params_benchmark; do
  # Run the training script with the current parameter set
  python benchmark.py $params_benchmark
done < params_benchmark.txt