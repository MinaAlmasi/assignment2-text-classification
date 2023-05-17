#!/bin/bash

# create virtual environment
python3 -m venv classifier_env

# activate virtual environment 
source ./classifier_env/bin/activate

echo -e "[INFO:] Installing necessary requirements..."

# install reqs
python3 -m pip install -r requirements.txt

# deactivate env 
deactivate

echo -e "[INFO:] Setup complete!"
