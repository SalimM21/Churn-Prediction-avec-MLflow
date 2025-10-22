"""
logistic_script.py
-------------------
Modèle : Régression Logistique
Stratégies : None | class_weight | SMOTE
Logging complet avec MLflow.
"""

import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

# Chargement des données et du pipeline
df = pd.read_csv("data/dataset.csv")
df = df[df["Age"] <= 80]
X = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"], errors="ignore")
y = df["Exited"]

X_train, X_test = joblib.load("data/X_train.pkl"), joblib.load("data/X_test.pkl")
y_train, y_test = joblib.load("data/y_train.pkl"), joblib.load("data/y_test.pkl")

preprocessor = joblib.load("models/preprocessing_pipeline.pkl")

strategies = {
    "none": (X_train, y_train, {}),
    "class_weight": (X_train, y_train, {"class_weight": "balanced"}),
    "smote": (None, None, {})  # à remplir dynamiquement
}

mlflow.set_experiment("Churn_Prediction_Models")

for strategy, (X_tr, y_tr, params) in strategies.items():
    with mlflow.start_run(run_name=f"LogReg_{strategy}"):
        mlflow.set_tag("model_name", "LogisticRegression")
        mlflow.set_tag("imbalance_strategy", strategy)

        # Appliquer SMOTE si besoin
        if strategy == "smote":
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_train, y_train)
            X_tr, y_tr = X_res, y_res

        # Pipeline complet
        model = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(**params, max_iter=1000, random_state=42))
        ])

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_test)

        # Calcul des métriques
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("strategy", strategy)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")
        print(f"✅ LogReg ({strategy}) - F1 = {f1:.3f}")
