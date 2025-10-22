"""
forest_script.py
-------------------
Modèle : Random Forest
Stratégies : None | class_weight | SMOTE
"""

import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/dataset.csv")
df = df[df["Age"] <= 80]
X = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"], errors="ignore")
y = df["Exited"]

X_train, X_test = joblib.load("data/X_train.pkl"), joblib.load("data/X_test.pkl")
y_train, y_test = joblib.load("data/y_train.pkl"), joblib.load("data/y_test.pkl")
preprocessor = joblib.load("models/preprocessing_pipeline.pkl")

strategies = ["none", "class_weight", "smote"]
mlflow.set_experiment("Churn_Prediction_Models")

for strategy in strategies:
    with mlflow.start_run(run_name=f"RandomForest_{strategy}"):
        mlflow.set_tag("model_name", "RandomForestClassifier")
        mlflow.set_tag("imbalance_strategy", strategy)

        if strategy == "smote":
            sm = SMOTE(random_state=42)
            X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
        else:
            X_train_bal, y_train_bal = X_train, y_train

        clf_params = {"n_estimators": 200, "random_state": 42}
        if strategy == "class_weight":
            clf_params["class_weight"] = "balanced"

        model = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("clf", RandomForestClassifier(**clf_params))
        ])

        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("strategy", strategy)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"✅ RandomForest ({strategy}) - F1 = {f1:.3f}")
