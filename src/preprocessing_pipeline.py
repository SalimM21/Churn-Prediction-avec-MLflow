"""
preprocessing_pipeline.py
--------------------------
Ce script prépare les données clients pour la prédiction du churn :
- Chargement et nettoyage du dataset.
- Suppression des colonnes inutiles.
- Traitement des valeurs manquantes et aberrantes.
- Construction d’un pipeline de prétraitement (numérique + catégoriel).
- Sauvegarde du pipeline prêt pour l’entraînement des modèles ML.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# =========================
# 1️⃣ Chargement des données
# =========================
def load_dataset(path: str) -> pd.DataFrame:
    """Charge le dataset depuis un fichier CSV."""
    df = pd.read_csv(path)
    print(f"✅ Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


# =========================
# 2️⃣ Nettoyage et préparation
# =========================
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les données selon les règles définies."""
    # Supprimer les colonnes inutiles
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors="ignore")

    # Supprimer les valeurs aberrantes (Age > 80)
    df = df[df["Age"] <= 80]

    print(f"🧹 Données nettoyées : {df.shape[0]} lignes restantes après filtrage.")
    return df


# =========================
# 3️⃣ Séparation features / target
# =========================
def split_features_target(df: pd.DataFrame, target_col: str = "Exited"):
    """Sépare les features et la target."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"🎯 Variable cible : '{target_col}' - Classe positive = {y.sum()}/{len(y)}")
    return X, y


# =========================
# 4️⃣ Construction du pipeline
# =========================
def build_preprocessing_pipeline(X: pd.DataFrame):
    """Construit un pipeline pour le prétraitement des données."""
    # Identifier les colonnes par type
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"🔢 Variables numériques : {numeric_features}")
    print(f"🔠 Variables catégorielles : {categorical_features}")

    # Pipelines individuels
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combinaison des deux pipelines
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    print("⚙️ Pipeline de prétraitement construit avec succès.")
    return preprocessor


# =========================
# 5️⃣ Split train/test
# =========================
def split_train_test(X, y, test_size=0.2, random_state=42):
    """Effectue une séparation train/test stratifiée."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"📊 Split : {X_train.shape[0]} train / {X_test.shape[0]} test")
    return X_train, X_test, y_train, y_test


# =========================
# 6️⃣ Sauvegarde du pipeline
# =========================
def save_pipeline(pipeline, path="models/preprocessing_pipeline.pkl"):
    """Sauvegarde le pipeline sur disque."""
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"💾 Pipeline sauvegardé : {path}")


# =========================
# 7️⃣ Main
# =========================
if __name__ == "__main__":
    # Charger et préparer le dataset
    df = load_dataset("data/dataset.csv")
    df = clean_dataset(df)
    X, y = split_features_target(df)

    # Construire le pipeline
    preprocessor = build_preprocessing_pipeline(X)

    # Split train/test
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Sauvegarder le pipeline
    save_pipeline(preprocessor)

    print("✅ Prétraitement terminé avec succès !")
