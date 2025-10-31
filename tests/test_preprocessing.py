import os
import sys
import pandas as pd

# Ensure repository root is on sys.path so "src" imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_build_preprocessing_pipeline_returns_transformer():
	"""Smoke test: build_preprocessing_pipeline doit retourner un transformer (ColumnTransformer)."""
	from src.preprocessing_pipeline import build_preprocessing_pipeline

	# DataFrame minimal avec colonnes numériques et catégorielles
	X = pd.DataFrame({
		"num1": [1, 2, 3],
		"num2": [0.1, 0.2, 0.3],
		"cat": ["a", "b", "a"]
	})

	preproc = build_preprocessing_pipeline(X)

	# ColumnTransformer expose fit_transform (et transform)
	assert hasattr(preproc, "fit_transform") or hasattr(preproc, "transform")


def test_clean_dataset_removes_unwanted_columns_and_filters_age():
	"""Vérifie que clean_dataset supprime les colonnes inutiles et filtre les âges > 80."""
	from src.preprocessing_pipeline import clean_dataset

	df = pd.DataFrame({
		"RowNumber": [1, 2, 3, 4],
		"CustomerId": [10, 11, 12, 13],
		"Surname": ["A", "B", "C", "D"],
		"Age": [25, 85, 40, 90],
		"Exited": [0, 1, 0, 1]
	})

	cleaned = clean_dataset(df)

	# Les colonnes inutiles doivent être supprimées
	for col in ["RowNumber", "CustomerId", "Surname"]:
		assert col not in cleaned.columns

	# Les lignes avec Age > 80 doivent être filtrées
	assert cleaned["Age"].max() <= 80


def test_split_train_test_stratified_and_shapes():
	"""Vérifie que split_train_test effectue une séparation stratifiée et renvoie des tailles correctes."""
	from src.preprocessing_pipeline import split_train_test

	# Construire X et y (10 échantillons, 4 positives -> proportion 0.4)
	X = pd.DataFrame({"f": list(range(10))})
	y = pd.Series([0] * 6 + [1] * 4)

	X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.5, random_state=42)

	# Tailles
	assert len(X_train) + len(X_test) == len(X)
	assert len(y_train) == len(X_train) and len(y_test) == len(X_test)

	# Avec test_size=0.5 et proportion 0.4 on attend exactement 2 positives dans train et 2 dans test
	assert y_test.sum() == 2
	assert y_train.sum() == 2
