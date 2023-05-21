# Text Classification Benchmarks with scikit-learn
Repository link: https://github.com/MinaAlmasi/assignment2-text-classification

This repository forms *assignment 2* by Mina Almasi (202005465) in the subject *Language Analytics*, *Cultural Data Science*, F2023. The assignment description can be found [here](https://github.com/MinaAlmasi/assignment2-text-classification/blob/main/assignment-desc.md). 

The repository contains code for training and evaluating a logistic regression and a neural network classifier using ```scikit-learn``` in a binary classification task (see [Results](https://github.com/MinaAlmasi/assignment2-text-classification/tree/main#results) for their final performance). As a bonus, the training also involves a short grid search to optimise the hyperparameters for each model. 

## Data
The classifiers are trained on the [Fake or Real News](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news) dataset. The dataset contains the title (headline), text and label (```FAKE``` or ```REAL```) of 7796 articles. The classifcation task is to predict whether the ```text``` column comes from a ```FAKE``` or ```REAL``` news article. The ```text``` column is vectorized prior to this. The original dataset and the vectorized data are located in the ```in``` folder. 
 
## Reproducibility
To reproduce the results, follow the instructions in the [*Pipeline*](https://github.com/MinaAlmasi/assignment2-text-classification/tree/main#pipeline) section. 

NB! Be aware that the grid search is computationally heavy and may take several minutes. Cloud computing (e.g., [UCloud](https://cloud.sdu.dk/)) is encouraged.


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
├── run.sh                              <---    run entire pipeline 
├── run-LR.sh                           <---    run grid search, train + eval for LR 
├── run-MLP.sh                          <---    run grid search, train + eval for MLP 
├── setup.sh                            <---    create venv & install reqs
├── src
│   ├── classify_LR.py                  <---    run logistic regression (LR)
│   ├── classify_MLP.py                 <---    run neural network (MLP)
│   ├── classify_pipeline.py            <---    contains classification pipeline (functions)
│   └── vectorize.py                    <---    run vectorization of data
└── utils
    ├── classify_dataload.py            <---    helper functions load .npz vec data + input args
    └── custom_logging.py               <---    custom logger to display user msg
```

## Pipeline
The pipeline has been tested on Ubuntu v22.10, Python v3.10.7 ([UCloud](https://cloud.sdu.dk/), Coder Python 1.77.3). Python's [venv](https://docs.python.org/3/library/venv.html) needs to be installed for the pipeline to work.


### Setup
Prior to running the classification pipeline,  run ```setup.sh``` to create a virtual environment (```env```) and install the necessary requirements within it:
```
bash setup.sh
```

### Running the Classification Pipeline
To run the entire pipeline, type ```run.sh``` in the terminal:
```
bash run.sh
```

Alternatively, you can run only the training and evaluation of the models seperately by running the ```run-X.sh``` scripts. For instance: 
```
bash run-LR.sh
```

### Running Vectorization (Custom Vectorization)
If you wish to run the vectorization with a bag-of-words representation or with more/less features in the vectorized data, you can run ```vectorize.py``` with additional arguments:
```
python src/vectorize.py -vec {VEC_TYPE} -n {NUMBER_OF_FEATURES} -d {CSV_DATAFILE}
```

| Arg        | Description                                         | Default               |
| :---       |:---                                                 |:---                   |
| ```-vec``` | choose either "bow" or "tfid"                       | tfid                  |
| ```-n```   | N features to keep                                  | 500                   |
| ```-d```   | datafile                                            | fake_or_real_news.csv |

NB! Remember to activate the ```env``` first (by running ```source ./env/bin/activate```)


The ```-d``` argument can also be used when training classifiers to run them on this custom vectorization:
```
 python src/classify_LR.py -d bow300f_data.npz
```

## Results

### Logistic Regression (LR)
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

### Remarks on the Results
Interestingly, both classifiers are identical in performance. Both models have high weighted F1 scores (```0.90```). These scores indicate that there is a clear difference in the vocabulary used in ```FAKE``` versus ```REAL``` news. While it is not certain that this difference is noticable for humans, the machines are at least able to pick up on it with training.

## Author 
This repository was created by Mina Almasi:

- github user: @MinaAlmasi
- student no: 202005465, AUID: au675000
- mail: mina.almasi@post.au.dk 