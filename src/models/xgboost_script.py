"""
xgboost_script_adapted.py
--------------------------
Modèle : XGBoost
Stratégies : None | class_weight | SMOTE
Logging complet dans MLflow avec métriques F1 et matrice de confusion.
"""

import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# -------------------------
# Chargement des données
# -------------------------
df = pd.read_csv("data/dataset.csv")
df = df[df["Age"] <= 80]
X = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"], errors="ignore")
y = df["Exited"]

# -------------------------
# Chargement des splits et du pipeline
# -------------------------
X_train, X_test = joblib.load("data/X_train.pkl"), joblib.load("data/X_test.pkl")
y_train, y_test = joblib.load("data/y_train.pkl"), joblib.load("data/y_test.pkl")
preprocessor = joblib.load("models/preprocessing_pipeline.pkl")

# -------------------------
# Stratégies et expérience MLflow
# -------------------------
strategies = ["none", "class_weight", "smote"]
mlflow.set_experiment("Churn_Prediction_Models")

for strategy in strategies:
    with mlflow.start_run(run_name=f"XGBoost_{strategy}"):

        # -------------------------
        # Tags MLflow
        # -------------------------
        mlflow.set_tag("model_name", "XGBoost")
        mlflow.set_tag("imbalance_strategy", strategy)

        # -------------------------
        # Gestion du déséquilibre
        # -------------------------
        if strategy == "smote":
            sm = SMOTE(random_state=42)
            X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
        else:
            X_train_bal, y_train_bal = X_train, y_train

        # -------------------------
        # Paramètres du modèle
        # -------------------------
        clf_params = {"n_estimators": 200, "learning_rate": 0.05, "random_state": 42}
        if strategy == "class_weight":
            # Ajuster le poids pour le déséquilibre
            clf_params["scale_pos_weight"] = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

        # -------------------------
        # Pipeline complet
        # -------------------------
        model = Pipeline([
            ("preprocess", preprocessor),
            ("clf", XGBClassifier(**clf_params, use_label_encoder=False, eval_metric="logloss"))
        ])

        # -------------------------
        # Entraînement
        # -------------------------
        model.fit(X_train_bal, y_train_bal)

        # -------------------------
        # Prédictions et métriques
        # -------------------------
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")  # ou "weighted"

        # -------------------------
        # Logging MLflow
        # -------------------------
        mlflow.log_param("strategy", strategy)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # -------------------------
        # Matrice de confusion
        # -------------------------
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n✅ XGBoost ({strategy}) - F1 = {f1:.3f} - Accuracy = {acc:.3f}")
        print(f"Matrice de confusion ({strategy}):\n{cm}")

        # Visualisation et sauvegarde de la matrice
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title(f"Matrice de confusion - {strategy}")
        plt.savefig("confusion_matrix.png")
        plt.close(fig)

        # Log artefact dans MLflow
        mlflow.log_artifact("confusion_matrix.png")
