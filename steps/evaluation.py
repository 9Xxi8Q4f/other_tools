import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE, R2
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from typing import Tuple

import mlflow

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model:RegressorMixin,
    test_data:pd.DataFrame,
    y_test:pd.Series
    ) -> Tuple[
        Annotated[float, "MSE"],
        Annotated[float, "R2"]
    ]:
    """
    Evaluate the model on the test data

    Args:
        model: The model to evaluate
        test_data: The test data to evaluate the model on,
    Returns:

    """
    try:
        y_pred = model.predict(test_data)
        mse = MSE().calculate_score(y_test, y_pred)
        mlflow.log_metric("mse", mse)
        r2 = R2().calculate_score(y_test, y_pred)
        mlflow.log_metric("r2", r2)

        return mse, r2
    
    except Exception as e:
        logging.error(e)
        raise e

