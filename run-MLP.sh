#!/bin/bash

# activate virtual environment (works only if setup.sh has been run only !)
source ./env/bin/activate

# run MLP
echo -e "[INFO:] Running classification pipeline with MLP ..." # user msg 
python3 src/classify_MLP.py

# deactivate virtual environment
deactivate

# celebratory user msg ! 
echo -e "[INFO:] Classifications complete!"