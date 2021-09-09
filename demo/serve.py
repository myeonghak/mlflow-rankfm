import os
import numpy as np
import pandas as pd

from rankfm._rankfm import _fit, _predict, _recommend
from rankfm.utils import get_data

import mlflow
from mlflow import log_metric, log_param, log_artifacts

import warnings
# warnings.filterwarnings("ignore", category=nb.NumbaPerformanceWarning)


path = "../dataset/"


import pickle
item_names=pickle.load(open(path+"item_names.pkl","rb"))


from serve_module import RankFM_MLflow

from preprocess import preprocess_data
train_data, valid_data = preprocess_data()

    
if __name__ == "__main__":

    
    mlflow.pyfunc.save_model(
            path="example_mlflow_model", python_model=RankFM_MLflow(), artifacts="")

    num_factors=20

    log_param("factor",num_factors)

    model = RankFM_MLflow(factors=num_factors, loss='warp', max_samples=20, alpha=0.01, sigma=0.1, learning_rate=0.10, learning_schedule='invscaling')

    model.fit(train_data, epochs=20, verbose=True)

    mlflow.pyfunc.log_model(artifact_path="example_mlflow_model", python_model=model)
    
    print("Done! mlflow serve let's go~")

