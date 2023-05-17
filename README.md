# Text Classification Benchmarks with scikit-learn
This repository forms *assignment 2* in the subject *Language Analytics*, *Cultural Data Science*, F2023. The assignment description can be found [here](https://github.com/AU-CDS/assignment-2---text-classification-MinaAlmasi/blob/main/assignment-desc.md). All code is written by Mina Almasi (202005465).

The repository contains code for training and evaluating a logistic regression and a neural network classifier using ```scikit-learn``` (see [results](https://github.com/AU-CDS/assignment-2---text-classification-MinaAlmasi#results) for their final performance).

Concretely, the classifiers are trained for a binary classification task on the **Fake News Dataset** ([fake_or_real_news.csv](https://github.com/AU-CDS/assignment-2---text-classification-MinaAlmasi/blob/main/in/fake_or_real_news.csv)). By default, the training also involves a short grid search to optimise the hyperparameters for each model. For greater detail on the repository structure and workflow, please refer to the the [*Project Structure*](https://github.com/AU-CDS/assignment-2---text-classification-MinaAlmasi#project-structure) section. 


## Reproducibility
To reproduce the results, follow the instructions in the [*Pipeline*](https://github.com/AU-CDS/assignment-2---text-classification-MinaAlmasi#pipeline) section. 

**NB! Be aware that the grid search is computationally heavy and may take several minutes. Cloud computing (e.g., UCloud) is encouraged.**


## Project Structure 
The repository is structured as such:

```
├── README.md
├── assignment-desc.md
├── in
│   ├── fake_or_real_news.csv           <---    initial data
│   └── tfid500f_data.npz               <---    vectorized data which models train & test on
├── models                              <---    classifiers & vectorizer saved here
│   ├── LR_500f_classifier.joblib
│   ├── MLP_500f_classifier.joblib
│   └── tfid_500f_vectorizer.joblib
├── out                                 <---    classifier eval metrics saved here
│   ├── LR_500f_metrics.txt
│   └── MLP_500f_metrics.txt
├── requirements.txt
├── run.sh                              <---    to run classifier pipeline for both classifiers
├── setup.sh                            <---    to create venv & install reqs
├── src
│   ├── classify_LR.py                  <---    to run logistic regression (LR)
│   ├── classify_MLP.py                 <---    to run neural network (MLP)
│   ├── classify_pipeline.py            <---    contains classification pipeline (functions)
│   └── vectorize.py                    <---    to vectorize data
└── utils
    ├── classify_dataload.py            <---    helper functions to load .npz vec data + args for input
    └── custom_logging.py               <---    custom logger to display user msg
```

### Workflow
The overall workflow is detailed as such 

1. Prior to training, the data was vectorized with TF-IDF representation using the script [vectorize.py](https://github.com/AU-CDS/assignment-2---text-classification-MinaAlmasi/blob/main/src/vectorize.py). For custom vectorization, see [*Running Vectorization*](https://github.com/AU-CDS/assignment-2---text-classification-MinaAlmasi#running-vectorization-custom-vectorization)
2. Two scripts ([*classify_LR.py*](https://github.com/AU-CDS/assignment-2---text-classification-MinaAlmasi/blob/main/src/classify_LR.py) & [*classify_MLP.py*](https://github.com/AU-CDS/assignment-2---text-classification-MinaAlmasi/blob/main/src/classify_MLP.py)) run a classification pipeline that is defined in [classify_pipeline.py](https://github.com/AU-CDS/assignment-2---text-classification-MinaAlmasi/blob/main/src/classify_pipeline.py)
3. Models and classifier metrics are saved

## Pipeline
The pipeline was tested on Ubuntu (Via UCloud) and on macOS Ventura (13.0.1). Whether it will work on Windows is not confirmed.


### Setup
Before running the classification, please run ```setup.sh``` in the terminal. This will create a Python virtual environment called ```classifier-env``` 

```
bash setup.sh
```


### Running the Analysis
To run the classification of both models (in the virtual env), type the following in the terminal: 
```
bash run.sh
```
**NB! Be aware that the grid search is computationally heavy and may take several minutes. Cloud computing (e.g., UCloud) is encouraged.**


Alternatively, you can run each script seperately (with ```classifier-env``` already activated):

**Logistic regression**
```
 python src/classify_LR.py
```

**Neural network**
```
 python src/classify_MLP.py
```

### Running Vectorization (Custom Vectorization)
If you wish to run the vectorization with a bag-of-words representation or with more/less features in the vectorized data, you can run ```vectorize.py``` with additional arguments ```-vec``` (*bow* or *tfid*) and ```-n``` (N features to keep in vectorization):
```
 source ./classifier_env/bin/activate
 python src/vectorize.py -vec bow -n 300
 deactivate
```
The code above will return a file called "*bow_300f_data.npz*" containing a bag-of-words vectorized dataset with 300 features. 

If you wish to vectorize a different dataset, this can be specified with an additional argument ```-d```:

```
 python src/vectorize.py -vec bow -n 300 -d datafile.csv
```


The ```-d``` can also be used when training classifiers to run them on this custom vectorization:
```
 source ./classifier_env/bin/activate
 python src/classify_LR.py -d bow300f_data.npz
 deactivate
```

## Results
The results for each classifier are given below. Interestingly, both classifiers are identical in performance. 

### Logistic regression
The best model was with the parameters {'C': 10.0, 'penalty': 'l2', 'solver': 'saga'}.
```
               precision    recall  f1-score   support

        FAKE       0.90      0.91      0.90       655
        REAL       0.90      0.89      0.89       612

    accuracy                           0.90      1267
   macro avg       0.90      0.90      0.90      1267
weighted avg       0.90      0.90      0.90      1267
```
### Neural Network (MLP)
The best model was with the parameters {'activation': 'logistic', 'hidden_layer_sizes': (10,), 'solver': 'adam'}
```
               precision    recall  f1-score   support

        FAKE       0.90      0.91      0.90       655
        REAL       0.90      0.89      0.89       612

    accuracy                           0.90      1267
   macro avg       0.90      0.90      0.90      1267
weighted avg       0.90      0.90      0.90      1267
```

## Author 
All code is made by Mina Almasi:
- github user: @MinaAlmasi
- student no: 202005465, AUID: au675000
- mail: mina.almasi@post.au.dk 
