from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline, inference_pipeline
)

import click
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService

DEPLOY="deploy"
PREDICT="predict"
DEPLOY_AND_PREDICT="deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice(DEPLOY, PREDICT, DEPLOY_AND_PREDICT),
    default=DEPLOY_AND_PREDICT,
    help="The type of deployment : "
    "Optionally you can choose deploy, predict or deploy_and_predict"
)

@click.option(
        "--min-accuracy",
        default= 0.92,
        help="The minimum accuracy to deploy the model"
)

def run_deployment(config: str, min_accuracy: float):
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    if deploy:
        continuous_deployment_pipeline(
            min_accuracy = min_accuracy,
            workers = 3,
            timeout=50
        )
    if predict:
        inference_pipeline()
