# check_runs.py
import mlflow

experiment = mlflow.get_experiment_by_name("Churn_Prediction_Models")
experiment_id = experiment.experiment_id

runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"])
print(runs.columns.tolist())
