import numpy as np
import pandas as pd
import os

#zenml config
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

#import steps
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

docker_settings = DockerSettings(
    required_integrations=[MLFLOW]
)

class DeploymentTriggerConfig(BaseParameters):
    min_accuracy: float = 0.92

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig
):
    return accuracy >= config.min_accuracy

@pipeline(enable_cache=True, settings={"docker_settings": docker_settings})
def continuous_deployment_pipeline(
    min_accuracy : float = 0.92,
    workers : int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    df = ingest_df()
    X_train, X_test, y_train, y_test = clean_df(df)
    trained_model = train_model(X_train, X_test, y_train, y_test)
    mse, r2 = evaluate_model(trained_model, X_test, y_test)

    deployment_decision = deployment_trigger(r2, min_accuracy=min_accuracy)
    if deployment_decision:
        mlflow_model_deployer = mlflow_model_deployer_step(
            model=trained_model
        )
        mlflow_model_deployer.deploy(
            service=MLFlowDeploymentService,
            workers=workers,
            timeout= timeout)
        
 