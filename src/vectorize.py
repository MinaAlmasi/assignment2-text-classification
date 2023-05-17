'''
Script for Assignment 2, Language Analytics, Cultural Data Science, F2023

This script is intended for machine learning purposes with vectorized data. 
It will read a CSV file with a 'text' column and 'label' column, split the data into a test and training dataset and vectorize the dataset using either a Bag-of-Words or TF-IDF textual representation. 
The vectorizer and vectorized data will be saved in the "in" and "models" folder, respectively. It futhermore contains helper 

Run the script by typing in the command line: 
    python src/vectorize.py

Additional arguments for running the script (if not specified, the defaults 'vec = tfid', 'n = 500' and 'd = fake_or_real_news.csv' will run)
    -vec (for the type of vectorizer), -n (for N features to keep in the vectorization) and -d (for datafile)

@MinaAlmasi
'''

# system tools
import pathlib
import argparse
import time

# data wrangling
import pandas as pd 
import numpy as np

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from joblib import dump

# custom logger
import sys
sys.path.append(str(pathlib.Path(__file__).parents[1]))
from utils.custom_logging import custom_logger # custom logger function


## helper functions ##
def split_data(datafile:str, input_dir:pathlib.Path, X_col:str, Y_col:str, test_size=0.2, random_state=129):
    '''
    Read data file and split data into train and test using scikit-learn's train_test_split

    Args:
        - datafile: name of datafile (CSV)
        - input_dir: directory where datafile is located
        - X_col: name of column in datafile that serve as X-values (data features)
        - Y_col: name of column in datafile that serve as the Y-values (labels)
        - test_size: ratio of split of data. Defaults to 0.2, creating an 80/20 split. 
        - random_state: to ensure reproducibility of split. Defaults to 129. 
    
    Returns: 
        - X_train, X_test, y_train, y_test: data split into X train/test features and Y train/test labels

    '''
    data_path = input_dir / datafile
    data = pd.read_csv(data_path)

    # define X and Y 
    X = data[X_col]
    y = data[Y_col]

    X_train, X_test, y_train, y_test = train_test_split(X,           # texts for the model
                                                    y,          # classification labels
                                                    test_size=test_size,   # create an 80/20 split
                                                    random_state=random_state) # random state for reproducibility
    
    return X_train, X_test, y_train, y_test


def vectorize_data(X_data:list, vectorizer:str = "tfid", 
                   ngram_range:tuple=(1,2), lowercase:bool=True, 
                   max_min_df:list=[0.95, 0.05], max_features:int=500):
    '''
    Vectorize data with either bag-of-words (BOW) or TF-IDF textual representation. 
    
    Args:
        - X_data: list containing X vector for train data and an X vector test data. 
        - vectorizer: bag-of-words or TF-IDF (specified as either 'tfid' or 'bow'). Defaults to 'tfid'
        - ngram_range: range of continous sequences (unigrams, bigrams, trigrams). Defaults to containing both unigrams and bigrams.
        - lowercase: whether the text should be lowercased or not. Defaults to True. 
        - max_min_df: range of words to be removed from vectorized data (max = common words, min = rare words). Defaults to 0.95 and 0.05.
        - max_features: Amount of features to be kept in vectorized data. Defaults to 500. 
   
    Returns:
        - vectorizer: vectorizer object
        - X_train_feats, X_test feats: vectorized arrays
    '''
    
    # initialize either BOW of TFID vectorizer   
    if vectorizer == "tfid":
        vec = TfidfVectorizer(
            ngram_range = ngram_range,    # define ngram range,
            lowercase =  lowercase,       # lowercase or not
            max_df = max_min_df[0],           # rm super common words
            min_df = max_min_df[1],           # rm super rare words
            max_features = max_features)
    
    elif vectorizer == "bow":
        vec = CountVectorizer(
            ngram_range = ngram_range,    # define ngram range,
            lowercase =  lowercase,       # lowercase or not
            max_df = max_min_df[0],           # rm super common words
            min_df = max_min_df[1],           # rm super rare words
            max_features = max_features)
    
    # fit vectorizer to training data
    X_train_feats = vec.fit_transform(X_data[0])

    # transform test data
    X_test_feats = vec.transform(X_data[1])

    return vectorizer, X_train_feats, X_test_feats


## scripting functions ##
def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-vec", "--vectorizer", help="choose either 'tfid' or 'bow' for text representation", default="tfid")
    parser.add_argument("-n", "--n_features", help="amount of features", default=500, type=int)
    parser.add_argument("-d", "--data", help = "data you want to vectorize", default = "fake_or_real_news.csv")

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def main():
    start_time = time.time()

    # config logging, args
    logging = custom_logger("vectorize_logger")
    args = input_parse()

    # define paths
    path = pathlib.Path(__file__) # path to current file
    input_dir = path.parents[1] / "in"
    datafile = args.data
    model_outpath =  path.parents[1] / "models"
    
    # split and vectorize data
    logging.info(f'Loading data: splitting and vectorizing with {args.vectorizer}')

    X_train, X_test, y_train, y_test = split_data(datafile, input_dir, "text", "label")
    vectorizer, X_train_feats, X_test_feats = vectorize_data([X_train, X_test], vectorizer=args.vectorizer, max_features=args.n_features)

    # save data 
    logging.info(f"Saving {args.vectorizer} vectorizer & vectorized arrays")
    np.savez(input_dir/f"{args.vectorizer}{args.n_features}f_data.npz", X_train_feats = X_train_feats, X_test_feats = X_test_feats, y_train = y_train, y_test = y_test) 

    # save vectorizer
    dump(vectorizer, model_outpath/f"{args.vectorizer}{args.n_features}f_vectorizer.joblib")

    # print elapsed time
    elapsed = round(time.time() - start_time, 2)

    logging.info(f'Vectorization finished. Time elapsed: {elapsed} seconds')


## run vectorization ##
if __name__ == "__main__":
    main()