#!/bin/bash

# Loop over each line in params_homo.txt
while IFS= read -r params_homo; do
  # Run the training script with the current parameter set
  python train.py $params_homo
done < params_homo.txt

echo "All runs completed."