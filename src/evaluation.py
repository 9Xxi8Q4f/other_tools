import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class evaluation(ABC):
    """
    Abstract class for model evaluation
    """
    @abstractmethod
    def calculate_score(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        Calculates the score

        Args:
            y_true: The true values
            y_pred: The predicted values

        Returns:
            None
        """

class MSE(evaluation):
    """
    Class for calculating the Mean Squared Error
    """
    def calculate_score(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        Calculates the score

        Args:
            y_true: The true values
            y_pred: The predicted values

        Returns:
            None
        """
        try:
            logging.info("Calculating the Mean Squared Error")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"Mean Squared Error: {mse}")
            return mse
        except Exception as e:
            logging.error(e)
            raise e

class R2(evaluation):
    """
    Class for calculating the R2 Score
    """
    def calculate_score(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        Calculates the score

        Args:
            y_true: The true values
            y_pred: The predicted values

        Returns:
            None
        """
        try:
            logging.info("Calculating the R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 Score: {r2}")
            return r2
        except Exception as e:
            logging.error(e)
            raise e
        
