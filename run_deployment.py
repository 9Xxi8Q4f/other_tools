from pipelines.deployment_pipeline import (
    deployment_pipeline, inference_pipeline
)

import click
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
    if config == DEPLOY:
        deployment_pipeline(min_accuracy)
    elif config == PREDICT:
        inference_pipeline()
    elif config == DEPLOY_AND_PREDICT:
        deployment_pipeline(min_accuracy)
        inference_pipeline()