# ExamenBloc2/src/train.py
"""
Entraînement du modèle ML sur les features préparées.

Ce script :
1. Charge les features préparées par prepare_features.py
2. Sépare X et y (variable cible : order_is_returned)
3. Sépare train/test
4. Entraîne un modèle LogisticRegression
5. Évalue le modèle sur le test
6. Stocker le modèle entraîné dans le dossier artifacts
7. Stocke les métriques d'évaluation dans un fichier texte dans artifacts
"""

import pandas as pd
import os
import joblib
from src.config import ARTIFACTS_DIR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.config import FEATURES_CSV_PATH

# --- Étape 1 : Chargement des features ---
df = pd.read_csv(FEATURES_CSV_PATH)
print("Features chargées :", df.shape)

# --- Étape 2 : Séparation X / y ---
y = df['order_is_returned'].astype(int)  # Variable cible (0/1)
X = df.drop(columns=['order_is_returned'])

# --- Étape 3 : Séparation train/test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Étape 4 : Entraînement du modèle LogisticRegression ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Étape 5 : Prédiction et score ---
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
print(f"Score du modèle sur le jeu de test : {score:.4f}")

# --- Étape 6 : Sauvegarde du modèle entraîné ---
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
model_path = ARTIFACTS_DIR / "logistic_regression_model.joblib"
joblib.dump(model, model_path)
print(f"Modèle sauvegardé dans {model_path}")

# --- Étape 7 : Sauvegarde des métriques d'évaluation ---
metrics_path = ARTIFACTS_DIR / "model_metrics.txt"
with open(metrics_path, 'w') as f:
    f.write(f"Accuracy: {score:.4f}\n")
print(f"Métriques sauvegardées dans {metrics_path}")



