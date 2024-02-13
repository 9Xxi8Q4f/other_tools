import logging
import pandas as pd

from zenml import step

class Ingest_data():
    """
    Ingest data from a given path
    """
    def __init__ (self, path):
        self.path = path

    def get_data(self):
        logging.info(f"Ingesting data from {self.path}")
        df = pd.read_csv(self.path)
        return df  
     
@step
def ingest_df(path: str) -> pd.DataFrame:
    """
    Ingest data from a given path
    
    Args:
        path: path to the data
    Returns:
        pd.DataFrame
    Example:
        >>> ingest_data("data.csv")
        Ingesting data from data.csv 
    """

    try:
        return Ingest_data(path).get_data()
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e
    
