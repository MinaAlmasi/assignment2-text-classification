'''
Script for Assignment 2, Language Analytics, Cultural Data Science, F2023

This script comprises several functions which make up a pipeline for fitting and evaluating a scikit-learn classifier. 
The classification pipeline also includes a grid search function (relying on scikit-learn's GridSearchCV) which can optionally be set to run.

Import clf_pipeline into your script to run the entire pipeline or pick and choose ! 

@MinaAlmasi
'''

# system tools
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parents[1]))
from utils.custom_logging import custom_logger # custom logger function
from datetime import datetime

# machine learning
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from joblib import dump

## functions ##
def clf_grid_search_fit(classifier, param_grid, cv, X_train_feats, y_train):
    '''
    Performs grid search for an instantiated classifier for a specified parameter space. Returns best model on training data. 

    Args: 
        - classifier: instantiated classifier (e.g., LogisticRegression() or MLPClassifier())
        - grid_search: whether the pipeline should include a grid search to find the best params. Defaults to False
        - param_grid: parameter space to perform grid search with if grid_search = True. Defaults to None
        - cv: the number of (Stratified)Kfold. Defaults to None. If None and grid_search = True, cv will be 5-fold

        - X_train_feats: train data array
        - y_train: train data labels 
    '''

    # create grid search instance
    grid_search = GridSearchCV(classifier, param_grid = param_grid, cv = cv, verbose=1)
    
    # perform grid search
    grid_search.fit(X_train_feats, y_train)

    return grid_search.best_estimator_, grid_search.best_params_ 


def clf_evaluate(classifier, X_test_feats, y_test):
    '''
    Evaluates fitted classifier on test data, returning a classification report. 

    Args: 
        - classifier: classifier already fitted on training data
        - X_test_feats: test data array
        - y_test: test data labels

    Returns: 
        - clf_metrics: classification report containing information such as accuracy, F1, precision and recall 
    '''

    # make predictions
    y_pred = classifier.predict(X_test_feats)

    # evaluate predictions 
    clf_metrics = metrics.classification_report(y_test, y_pred)

    return clf_metrics 


def clf_get_name(classifier): 
    '''
    Retrieves the name of an instantiated classifier. Useful for logging and saving models or classification metrics. 

    Args: 
        - classifier: instantiated classifier

    Returns: 
        - classifier_name: name of instantiated classifier (abbreviation for Logistic Regression and MLPClassifier, full names for other sklearn models)
    '''

    if classifier.__class__.__name__ == "LogisticRegression":
        classifier_name = "LR"
    elif classifier.__class__.__name__ == "MLPClassifier":
        classifier_name = "MLP"
    else: 
        classifier_name = classifier.__class__.__name__
    
    return classifier_name


def clf_metrics_to_txt(txt_name, output_dir, clf_metrics, best_params=None):
    '''
    Write classifier metrics report to .txt file. 

    Args:
        - clf_metrics: metrics report (sklearn.metrics.classification_report() or returned from clf_evaluate)
        - best_params: best parameters (if model is run with grid_search). Defaults to None. 
        - txtname: filename for .txt report
        - output_dir: directory where the text file should be stored. 

    Outputs: 
        - .txt file in specified output_dir
    '''
    # define filepath
    filepath = output_dir / txt_name

    #if best_params is None, write the following:
    if best_params == None: 
        with open(f"{filepath}.txt", "w") as file: 
            file.write(f"Results from model run at {datetime.now()} \n {clf_metrics}")

    #if best_params is specified (i.e., grid_search is performed, write the following)
    else: 
        with open(f"{filepath}.txt", "w") as file: 
            file.write(f"Results from model with parameters: {best_params}. Run at {datetime.now()} \n {clf_metrics}")


def clf_pipeline(classifier, X_train_feats, y_train, X_test_feats, y_test, output_dir:pathlib.Path, 
                 save_model:bool=False, model_dir:pathlib.Path=None, grid_search:bool=False, param_grid=None, cv=None):
    '''
    Classifier pipeline which does model fitting and model evaluation of an instantiated classifier. 
    Additionally, it can perform grid search of a parameter space (param_grid) if grid_search is set to True. 
    It will save model metrics to as a .txt file in a specified diretory (output_dir) with an option to also save the model.

    Args: 
        - classifier: instantiated classifier (e.g., LogisticRegression() or MLPClassifier())
        - X_train_feats, y_train, X_test_feats, y_test: data arrays for model fitting and evaluation
        - output_dir: directory where classifier metrics should be saved after evaluation
        - save_model: whether the model should be saved. Defaults to false
        - model_dir: directory where model is saved if save_model = True. Defaults to None
        
        - grid_search: whether the pipeline should include a grid search to find the best params. Defaults to False
        - param_grid: parameter space to perform grid search with if grid_search = True. Defaults to None
        - cv: the number of (Stratified)Kfold. Defaults to None. If None and grid_search = True, cv will be 5-fold by default as stated in https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html. 
    
    Returns: 
        - classifier: fitted and evaluated classifier

    Output:
        - model metrics as .txt file 
        - model as .joblib (if save_model = True)
    '''
    # instantiate logger, define classifier_name for logging and txtfile
    logging = custom_logger("pipeline")
    classifier_name = clf_get_name(classifier)

    # create path if they do not exist
    output_dir.mkdir(exist_ok=True, parents=True)
    model_dir.mkdir(exist_ok=True, parents=True)

    # if grid_search is set to True, do grid search to find the best model and return that model ! 
    if grid_search == True: 
        logging.info(f"Commencing grid search for {classifier_name} with {cv} cv folds")
        classifier, best_params = clf_grid_search_fit(classifier, param_grid, cv, X_train_feats, y_train)

    else: # if False, fit the classifier normally
        logging.info(f"Fitting {classifier_name}")
        classifier = classifier.fit(X_train_feats, y_train)

    # evaluate classifier, save metrics report 
    logging.info(f"Evaluating ... ")
    clf_metrics = clf_evaluate(classifier, X_test_feats, y_test)

    # write metrics report to txt
    txtname = f"{classifier_name}_{X_test_feats.shape[1]}f_metrics"  #X_test_feats.shape[1] gives the amount of features trained on  
    # depending on whether grid_search has been performed, save file with or without best_params: 

    logging.info("Saving ...")    
    if grid_search == True: 
        clf_metrics_to_txt(txtname, output_dir, clf_metrics, best_params)
    else: 
        clf_metrics_to_txt(txtname, output_dir, clf_metrics)

    # save model only if specified
    if save_model == True:
        filepath = model_dir /  f"{classifier_name}_{X_test_feats.shape[1]}f_classifier"
        dump(classifier, f"{filepath}.joblib")
    
    return classifier
