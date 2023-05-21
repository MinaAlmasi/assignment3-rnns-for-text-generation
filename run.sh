#!/bin/bash

# activate virtual environment 
source ./env/bin/activate

# run training 
echo -e "[INFO:] TRAINING MODEL"
python3 src/train_model.py -n 1000
echo -e "[INFO:] MODEL TRAINED AND SAVED"

# run generation
echo -e "[INFO:] RUNNING TEXT GENERATION"
python3 src/generate_text.py

# deactivate env 
deactivate

