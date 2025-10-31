import importlib.util
import sys
import types
import os
import sys
import pandas as pd
import numpy as np
import matplotlib

# Ensure repository root is on sys.path so "src" imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Use non-interactive backend for tests to avoid GUI/Tk errors
matplotlib.use("Agg")


def _make_fake_mlflow():
    """Crée un module mlflow factice avec les API minimales utilisées par les scripts."""
    fake = types.SimpleNamespace()

    def set_experiment(name):
        return None

    class _DummyRun:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def start_run(run_name=None):
        return _DummyRun()

    def set_tag(k, v):
        return None

    def log_param(k, v):
        return None

    def log_metric(k, v):
        return None

    # plural variants used in some scripts
    def log_params(d):
        return None

    def log_metrics(d):
        return None

    def log_artifact(p):
        return None

    fake.set_experiment = set_experiment

    def log_figure(fig, artifact_file=None):
        return None

    fake.log_figure = log_figure
    fake.start_run = start_run
    fake.set_tag = set_tag
    fake.log_param = log_param
    fake.log_metric = log_metric
    fake.log_params = log_params
    fake.log_metrics = log_metrics
    fake.log_artifact = log_artifact

    # Submodule sklearn with stubbed log_model
    fake.sklearn = types.SimpleNamespace()

    def log_model(model, name):
        return None

    fake.sklearn.log_model = log_model

    return fake


def _register_fake_mlflow_submodules(fake_mlflow):
    """Registere des entrées dans sys.modules pour permettre 'import mlflow.sklearn'."""
    msk = types.ModuleType("mlflow.sklearn")
    msk.log_model = lambda *a, **k: None
    sys.modules["mlflow.sklearn"] = msk
    # Also ensure mlflow itself is importable as a module with attribute sklearn
    sys.modules["mlflow"] = fake_mlflow


def _register_fake_xgboost_and_imblearn():
    # Fake xgboost with XGBClassifier
    xgb_mod = types.ModuleType("xgboost")

    class XGBClassifier:
        def __init__(self, *a, **k):
            pass

        def fit(self, X, y):
            return self

        def predict(self, X):
            # return zeros of appropriate length
            import numpy as _np

            return _np.zeros(len(X), dtype=int)

    xgb_mod.XGBClassifier = XGBClassifier
    sys.modules["xgboost"] = xgb_mod

    # Fake imblearn.over_sampling.SMOTE
    imblearn_os = types.ModuleType("imblearn.over_sampling")

    class SMOTE:
        def __init__(self, *a, **k):
            pass

        def fit_resample(self, X, y):
            return X, y

    imblearn_os.SMOTE = SMOTE
    sys.modules["imblearn.over_sampling"] = imblearn_os


def _import_script(path):
    """Importe un script Python depuis un chemin de fichier sans l'ajouter au package."""
    spec = importlib.util.spec_from_file_location("temp_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_import_logistic_script_monkeypatched(tmp_path, monkeypatch):
    """Importe `logistic_script.py` en patchant les dépendances externes pour éviter les IO lourds."""
    # Fake mlflow
    fake_mlflow = _make_fake_mlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    # register fake mlflow and its sklearn submodule
    _register_fake_mlflow_submodules(fake_mlflow)
    _register_fake_xgboost_and_imblearn()

    # Patch pandas.read_csv to return a small dataframe
    # Make a slightly larger dataframe so stratified split works
    df = pd.DataFrame({
        "Age": [30, 40, 25, 60, 50, 35, 45, 29, 55, 38],
        "RowNumber": list(range(1, 11)),
        "CustomerId": list(range(10, 20)),
        "Surname": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        "Exited": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "CreditScore": [600 + i for i in range(10)],
        "Balance": [1000.0 * i for i in range(10)],
        "EstimatedSalary": [30000 + 1000 * i for i in range(10)],
        "Feature1": [0.1 * i for i in range(1, 11)]
    })
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: df)

    # Patch joblib.load to return small train/test splits and a preprocessor that returns numpy arrays
    def fake_joblib_load(path):
        if "X_train" in path:
            return pd.DataFrame({"Feature1": [0.1, 0.2]})
        if "X_test" in path:
            return pd.DataFrame({"Feature1": [0.15]})
        if "y_train" in path:
            return pd.Series([0, 1])
        if "y_test" in path:
            return pd.Series([1])
        if "preprocessing_pipeline" in path:
            # Return a simple transformer with fit/transform that returns numpy array
            class SimplePreproc:
                def fit(self, X, y=None):
                    return self

                def transform(self, X):
                    return np.array(X)

                def fit_transform(self, X, y=None):
                    return self.transform(X)

            return SimplePreproc()
        return None

    monkeypatch.setattr("joblib.load", fake_joblib_load)

    # Patch plt.savefig to avoid file writes
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)

    # Now import the script (it will execute top-level code but with patched dependencies)
    path = "src/models/logistic_script.py"
    _import_script(path)


def test_import_xgboost_script_monkeypatched(tmp_path, monkeypatch):
    """Importe `xgboost_script.py` avec les mêmes patchs pour s'assurer qu'il s'exécute rapidement."""
    fake_mlflow = _make_fake_mlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    _register_fake_mlflow_submodules(fake_mlflow)
    _register_fake_xgboost_and_imblearn()

    df = pd.DataFrame({
        "Age": [30, 40, 25, 60, 50, 35, 45, 29, 55, 38],
        "RowNumber": list(range(1, 11)),
        "CustomerId": list(range(10, 20)),
        "Surname": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        "Exited": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "CreditScore": [600 + i for i in range(10)],
        "Balance": [1000.0 * i for i in range(10)],
        "EstimatedSalary": [30000 + 1000 * i for i in range(10)],
        "Feature1": [0.1 * i for i in range(1, 11)]
    })
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: df)

    def fake_joblib_load(path):
        if "X_train" in path:
            return pd.DataFrame({"Feature1": [0.1, 0.2]})
        if "X_test" in path:
            return pd.DataFrame({"Feature1": [0.15]})
        if "y_train" in path:
            return pd.Series([0, 1])
        if "y_test" in path:
            return pd.Series([1])
        if "preprocessing_pipeline" in path:
            class SimplePreproc:
                def fit(self, X, y=None):
                    return self

                def transform(self, X):
                    return np.array(X)

                def fit_transform(self, X, y=None):
                    return self.transform(X)

            return SimplePreproc()
        return None

    monkeypatch.setattr("joblib.load", fake_joblib_load)

    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)

    path = "src/models/xgboost_script.py"
    _import_script(path)


def test_import_forest_script_monkeypatched(tmp_path, monkeypatch):
    """Importe `forest_script.py` avec les mêmes patchs pour s'assurer qu'il s'exécute rapidement."""
    fake_mlflow = _make_fake_mlflow()
    _register_fake_mlflow_submodules(fake_mlflow)
    _register_fake_xgboost_and_imblearn()

    df = pd.DataFrame({
        "Age": [30, 40, 25, 60, 50, 35, 45, 29, 55, 38],
        "RowNumber": list(range(1, 11)),
        "CustomerId": list(range(10, 20)),
        "Surname": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        "Exited": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "Feature1": [0.1 * i for i in range(1, 11)]
    })
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: df)

    def fake_joblib_load(path):
        # forest_script may not call joblib.load at import; keep compatibility
        return None

    monkeypatch.setattr("joblib.load", fake_joblib_load)

    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)

    path = "src/models/forest_script.py"
    _import_script(path)


def test_train_model_callable_monkeypatched(tmp_path, monkeypatch):
    """Importe la fonction train_model depuis `forest_script` et l'exécute avec des données factices."""
    fake_mlflow = _make_fake_mlflow()
    _register_fake_mlflow_submodules(fake_mlflow)
    _register_fake_xgboost_and_imblearn()

    # Make sure any top-level reads are patched
    df = pd.DataFrame({
        "Age": [30, 40, 25, 60, 50, 35, 45, 29, 55, 38],
        "RowNumber": list(range(1, 11)),
        "CustomerId": list(range(10, 20)),
        "Surname": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        "Exited": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "Feature1": [0.1 * i for i in range(1, 11)]
    })
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: df)

    # Import the module (should not execute training at import time if guarded)
    mod = _import_script("src/models/forest_script.py")

    # Prepare small numeric arrays
    X_train = np.array([[0.1, 0.2], [0.2, 0.3]])
    y_train = np.array([0, 1])
    x_test_final = np.array([[0.15, 0.25]])
    y_test = np.array([1])

    # Patch matplotlib save/log to avoid IO
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)

    # Call the train_model function
    assert hasattr(mod, "train_model")
    mod.train_model(X_train, y_train, x_test_final, y_test, plot_name="test_plot", n_estimators=10, max_depth=2)
