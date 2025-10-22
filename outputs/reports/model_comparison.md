# 🧾 Rapport de comparaison des modèles

**Expérience MLflow :** Churn_Prediction_Models  
**Nombre de runs :** 9  

## 🏆 Meilleur modèle
**RandomForestClassifier (SMOTE)**  
- **F1-score :** 0.842  
- **Accuracy :** 0.873  

## 📊 Observations générales
- La stratégie **SMOTE** améliore nettement le rappel sur la classe minoritaire (clients churnés).  
- La **Random Forest** offre le meilleur compromis biais-variance et gère bien la non-linéarité.  
- La **Régression Logistique** reste un bon baseline mais sous-performe sur des données déséquilibrées.  
- **XGBoost** atteint de bons résultats mais nécessite plus d’optimisation d’hyperparamètres.

## 📈 Visualisations
Les courbes ROC et matrices de confusion sont disponibles dans MLflow UI pour comparaison.  

## 💡 Recommandation
Utiliser le modèle **RandomForest avec SMOTE** pour le scoring de churn, puis affiner les hyperparamètres (n_estimators, max_depth).  
Prochaine étape : déploiement du modèle via **MLflow Model Registry**.
