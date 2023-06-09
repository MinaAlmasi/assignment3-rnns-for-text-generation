#!/bin/bash

# activate virtual environment 
source ./env/bin/activate

echo -e "[INFO:] TRAINING MODEL"

# run script 
python3 src/train_model.py -n 1000 -e 2

# deactivate env 
deactivate

echo -e "[INFO:] MODEL TRAINED AND SAVED"