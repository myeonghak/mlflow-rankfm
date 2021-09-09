import numpy as np
import pandas as pd

import mlflow
from mlflow import log_metric, log_param, log_artifacts

from model import RankFM_MLflow

from rankfm.evaluation import hit_rate, reciprocal_rank, discounted_cumulative_gain, precision, recall, diversity


# optuna + visulization
import optuna
from optuna.integration.mlflow import MLflowCallback

from preprocess import preprocess_data
train_data, valid_data = preprocess_data()


class model_config:
    num_factors=3
    k=10
    num_epochs=3

def objective(trial):
    num_factors=model_config.num_factors
    k=model_config.k
    
    # parameters to optimize
    
    max_samples = trial.suggest_int('max_samples', 5, 20, 5)
    learning_schedule = trial.suggest_categorical('bootstrap', ['constant', 'invscaling'])
    sigma = trial.suggest_discrete_uniform('sigma', 0.1, 0.9, 0.1)
    alpha = trial.suggest_discrete_uniform('alpha', 0.01, 0.05, 0.01)
    
    # our ML pipeline
    model = RankFM_MLflow(factors=num_factors, loss='warp', max_samples=max_samples, alpha=alpha, 
                          sigma=sigma, learning_rate=0.10, learning_schedule=learning_schedule)
    
    
    with mlflow.start_run():
        mlflow.log_param("max_samples", max_samples)
        mlflow.log_param("learning_schedule", learning_schedule)
        mlflow.log_param("sigma", sigma)
        mlflow.log_param("alpha", alpha)
    
        model.fit(train_data, epochs=model_config.num_epochs, verbose=True)
    

        model_hit_rate = hit_rate(model, valid_data, k=k)
        model_reciprocal_rank = reciprocal_rank(model, valid_data, k=k)
        model_dcg = discounted_cumulative_gain(model, valid_data, k=k)
        model_recall = recall(model, valid_data, k=k)
        model_precision = precision(model, valid_data, k=k)

        mlflow.log_metric("hit_rate", model_hit_rate)
        mlflow.log_metric("MRR", model_reciprocal_rank)
        mlflow.log_metric("DCG", model_dcg)
        mlflow.log_metric("recall", model_recall)
        mlflow.log_metric("precision", model_precision)
    
    
    # mlflow.pyfunc.log_model(artifact_path="example_mlflow_model", python_model=model)
    
    return model_precision

def run():
    
    mlflow_cb = MLflowCallback(
        tracking_uri='mlruns',
        metric_name='precision'
    )

    hd_study = optuna.create_study(
        study_name='FM_movie',
        direction='maximize',
        pruner=optuna.pruners.HyperbandPruner(max_resource="auto")
    )

    hd_study.optimize(objective, n_trials=10, callbacks=[mlflow_cb])

    print("model created!")
    
if __name__ == "__main__":
    run()
