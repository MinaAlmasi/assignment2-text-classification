#!/bin/bash

# activate virtual environment (works only if setup.sh has been run only !)
source ./env/bin/activate

# run logistic regression
echo -e "[INFO:] Running classification pipeline with LR ..." # user msg 
python3 src/classify_LR.py

# deactivate virtual environment
deactivate

# celebratory user msg ! 
echo -e "[INFO:] Classifications complete!"