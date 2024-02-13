import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for model development
    """
    @abstractmethod
    def train(self):
        """
        Train the model
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear regression model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Train the model
        """
        try: 
            model = LinearRegression(**kwargs)
            model.fit(X_train, y_train)
            logging.info(f"Model trained successfully")
            return model
        except Exception as e:
            logging.error(e)