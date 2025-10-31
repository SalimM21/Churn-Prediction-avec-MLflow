## main.py
# -------------------------------
# Pipeline complet pour churn prediction avec RandomForest, gestion du déséquilibre et MLflow
# -------------------------------

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion

# Simple DataFrame selector transformer for scikit-learn pipelines
class DataFrameSelector:
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Expecting a pandas DataFrame; return numpy array for downstream transformers
        return X[self.attribute_names].values

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

warnings.filterwarnings('ignore')
# --------------------- Modeling ----------------------------


def train_model(X_train, y_train, X_test, y_test, plot_name, n_estimators=200, max_depth=None, class_weight=None):
    """
    Entraîne un RandomForestClassifier, log les métriques et la matrice de confusion sur MLflow
    """
    mlflow.set_experiment('churn-detection')

    with mlflow.start_run():
        mlflow.set_tag('clf', 'RandomForest')

        # Initialiser le modèle
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=45,
            class_weight=class_weight
        )
        clf.fit(X_train, y_train)

        # Prédictions sur le test set
        y_pred_test = clf.predict(X_test)

        # Calcul des métriques
        f1_test = f1_score(y_test, y_pred_test)
        acc_test = accuracy_score(y_test, y_pred_test)

        # Logging MLflow
        mlflow.log_params({'n_estimators': n_estimators, 'max_depth': max_depth})
        mlflow.log_metrics({'accuracy': acc_test, 'f1_score': f1_test})
        mlflow.sklearn.log_model(clf, f'{clf.__class__.__name__}_{plot_name}')

        # Matrice de confusion
        plt.figure(figsize=(10, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, cbar=False, fmt='.2f', cmap='Blues')
        plt.title(f'{plot_name}')
        plt.xticks(ticks=np.arange(2) + 0.5, labels=[False, True])
        plt.yticks(ticks=np.arange(2) + 0.5, labels=[False, True])

        # Sauvegarde de la figure dans MLflow
        conf_matrix_fig = plt.gcf()
        mlflow.log_figure(conf_matrix_fig, artifact_file=f'{plot_name}_conf_matrix.png')

        print(f"✅ {plot_name} - F1 = {f1_test:.3f}, Accuracy = {acc_test:.3f}")


# --------------------- Entrée principale ----------------------------
if __name__ == "__main__":
    # --------------------- Data Preparation ----------------------------

    # Chargement du dataset
    TRAIN_PATH = os.path.join(os.getcwd(), 'dataset.csv')
    df = pd.read_csv(TRAIN_PATH)

    # Supprimer les colonnes non pertinentes
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

    # Filtrage par âge
    df = df[df['Age'] <= 80]

    # Séparer features et target
    X = df.drop(columns=['Exited'])
    y = df['Exited']

    # Split train/test avec stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=45, shuffle=True, stratify=y
    )


    # --------------------- Data Processing ----------------------------

    # Définition des colonnes numériques et catégorielles
    num_cols = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
    categ_cols = ['Gender', 'Geography']
    ready_cols = list(set(X_train.columns) - set(num_cols) - set(categ_cols))

    # Pipeline pour les numériques
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_cols)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ], memory=None)

    # Pipeline pour les catégorielles
    categ_pipeline = Pipeline([
        ('selector', DataFrameSelector(categ_cols)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False))
    ], memory=None)

    # Pipeline pour les colonnes prêtes
    ready_pipeline = Pipeline([
        ('selector', DataFrameSelector(ready_cols)),
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ], memory=None)

    # Combinaison de tous les pipelines
    all_pipeline = FeatureUnion([
        ('numerical', num_pipeline),
        ('categorical', categ_pipeline),
        ('ready', ready_pipeline)
    ])

    # Transformation des données
    X_train_final = all_pipeline.fit_transform(X_train)
    X_test_final = all_pipeline.transform(X_test)


    # --------------------- Handling Imbalance ----------------------------

    # Calcul des poids pour class_weight
    vals_count = 1 - (np.bincount(y_train) / len(y_train))
    vals_count = vals_count / np.sum(vals_count)
    dict_weights = {i: vals_count[i] for i in range(2)}

    # Oversampling avec SMOTE
    smote = SMOTE(sampling_strategy=0.7, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_final, y_train)


    # --------------------- Exemples d'utilisation ----------------------------

    # Exemple d'entraînement sans pondération
    train_model(X_train_final, y_train, X_test_final, y_test, plot_name='RF_no_weight')

    # Exemple avec class_weight
    train_model(X_train_final, y_train, X_test_final, y_test, plot_name='RF_class_weight', class_weight=dict_weights)

    # Exemple avec SMOTE
    train_model(X_train_resampled, y_train_resampled, X_test_final, y_test, plot_name='RF_SMOTE')
