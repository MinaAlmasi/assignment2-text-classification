#!/bin/bash

# activate virtual environment (works only if setup.sh has been run only !)
source ./env/bin/activate

# run vectorization
echo -e "[INFO:] Running Vectorization ..." # user msg 
python3 src/vectorize.py

# run logistic regression
echo -e "[INFO:] Running classification pipeline with LR ..." # user msg 
python3 src/classify_LR.py

# run MLP
echo -e "[INFO:] Running classification pipeline with MLP ..." # user msg 
python3 src/classify_MLP.py

# deactivate virtual environment
deactivate

# celebratory user msg ! 
echo -e "[INFO:] Classifications complete!"