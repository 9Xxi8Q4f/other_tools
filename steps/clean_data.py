import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataPreProcessStrategy, DataDivideStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df:pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Cleans the data and divides it into train and test data

    Args:
        df (pd.DataFrame): The input data
    Returns:
        X_train: The training data
        X_test: The test data
        y_train: The training labels
        y_test: The test labels
    """
    try:
        df = DataCleaning(df, DataPreProcessStrategy()).handle_data()
        X_train, X_test, y_train, y_test = DataDivideStrategy().handle_data(df)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e
    