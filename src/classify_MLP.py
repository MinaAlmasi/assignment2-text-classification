'''
Script for Assignment 2, Language Analytics, Cultural Data Science, F2023

This script is made to train a simple neural network (MLPClassifier()) on a vectorized version of the "fake_or_real_news.csv" data.
The hyperparameters are set following a grid search upon a defined parameter space. 

Run the script by typing in the command line: 
    python src/classify_MLP.py -d {VECTORIZED_DATAFILE}

Where:
-d refers to datafile (vectorized). Defaults to tfid500f_data.npz

@MinaAlmasi
'''

# system tools
import pathlib
import time

# custom utils 
import sys
sys.path.append(str(pathlib.Path(__file__).parents[1]))
from utils.custom_logging import custom_logger
from utils.classify_dataload import load_npz_vec_data, input_parse

# machine learning
from classify_pipeline import clf_pipeline
from sklearn.neural_network import MLPClassifier

def main():
    start_time = time.time()
    
    # logging, args
    logging = custom_logger("MLP_logger")
    args = input_parse()

    # define paths
    path = pathlib.Path(__file__) # path to current file
    input_dir = path.parents[1] / "in" 
    datafile = args.data

    output_dir = path.parents[1] / "out"
    model_path = path.parents[1] / "models"

    # load data
    logging.info("Loading data ...")
    X_train_feats, X_test_feats, y_train, y_test = load_npz_vec_data(input_dir, datafile)

    # initialize classifier 
    logging.info(f"Intializing classifier")
    classifier = MLPClassifier(random_state=129, max_iter=5000) # max_iter is set to high val to avoid grid search convergences issues
        
    # define parameter space for grid search 
    grid={'hidden_layer_sizes':[(10,),(20,), (30,)],
          'activation':['logistic'],
          'solver':["adam"],
          }

    # run classifier pipeline (steps -> 1: grid search/fitting, 2: model eval, 3: save results)
    classifier = clf_pipeline(classifier = classifier, 
                 X_train_feats = X_train_feats, 
                 X_test_feats = X_test_feats, 
                 y_train = y_train, 
                 y_test = y_test,
                 output_dir = output_dir,
                 save_model = True,
                 model_dir = model_path,
                 grid_search = True,
                 param_grid = grid,
                 cv = 3
                 )

    # print elapsed time
    elapsed = round(time.time() - start_time, 2)

    logging.info(f"Classification finished. Classifier metrics saved to 'out' directory. \n Time elapsed: {elapsed} seconds.")

# run classifier
if __name__ == "__main__":
    main()
