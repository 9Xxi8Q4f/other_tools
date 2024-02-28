from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    # uri for mlflow tracking server
    print(Client().active_stack.experiment_tracker.get_tracking_uri())

    train_pipeline(path="data/olist_customers_dataset.csv")


    # mlflow ui --backend-store-uri "file: /home/o/.config/zenml/local_stores/45a17ebf-fa83-4b13-9a58-f69bed7b452b/mlruns"