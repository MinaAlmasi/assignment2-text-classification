#!/bin/bash

# activate virtual environment (works only if setup.sh has been run only !)
source ./classifier_env/bin/activate

# run logistic regression
echo -e "[INFO:] Running classification pipeline with LR ..." # user msg 
python src/classify_LR.py

# run MLP
echo -e "[INFO:] Running classification pipeline with MLP ..." # user msg 
python src/classify_MLP.py

# deactivate virtual environment
deactivate

# celebratory user msg ! 
echo -e "[INFO:] Classifications complete!"