import os

# Définir la structure du projet
structure = {
    "churn_prediction_mlflow": {
        "data": ["dataset.csv", "processed_data.csv"],
        "notebooks": ["1_data_preparation.ipynb", "3_training_mlflow.ipynb",
                      "4_model_analysis.ipynb", "5_demo_pipeline.ipynb"],
        "src": {
            "_files": ["preprocessing_pipeline.py", "utils.py"],
            "models": ["logistic_script.py", "forest_script.py", "xgboost_script.py"]
        },
        "mlruns": [],
        "outputs": {
            "figures": [],
            "reports": ["model_comparison.md"]
        },
        "tests": ["test_preprocessing.py", "test_models.py"],
        "_files": ["requirements.txt", "README.md", ".gitignore", "config.yaml", "setup.sh"]
    }
}

def create_structure(base_path, struct):
    for name, content in struct.items():
        current_path = os.path.join(base_path, name)
        os.makedirs(current_path, exist_ok=True)
        print(f"✅ Dossier créé : {current_path}")

        if isinstance(content, list):
            for item in content:
                file_path = os.path.join(current_path, item)
                open(file_path, 'a').close()
        elif isinstance(content, dict):
            # Créer les fichiers _files si présents
            if "_files" in content:
                for file_name in content["_files"]:
                    file_path = os.path.join(current_path, file_name)
                    open(file_path, 'a').close()
            # Créer les sous-dossiers
            for key, val in content.items():
                if key != "_files":
                    if isinstance(val, list):
                        subdir_path = os.path.join(current_path, key)
                        os.makedirs(subdir_path, exist_ok=True)
                        print(f"✅ Sous-dossier créé : {subdir_path}")
                        for file_name in val:
                            file_path = os.path.join(subdir_path, file_name)
                            open(file_path, 'a').close()
                    elif isinstance(val, dict):
                        create_structure(current_path, {key: val})

if __name__ == "__main__":
    base_dir = os.getcwd()
    create_structure(base_dir, structure)
    print("\n🎯 Structure du projet 'churn_prediction_mlflow' créée avec succès !")
