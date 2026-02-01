# ExamenBloc2/src/prepare_features.py
"""
Préparation des features pour le Machine Learning.

Ce script :
1. Charge les données nettoyées générées par extract_transform.py
2. Supprime les colonnes non pertinentes pour le ML
3. Impute les valeurs manquantes
4. Traite les outliers (IQR)
5. Encode les variables catégorielles (OneHotEncoding)
6. Normalise les variables numériques (RobustScaler)
7. Sauvegarde les features finales dans FEATURES_CSV_PATH
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, RobustScaler # RobustScaler pour la normalisation robuste des données numériques et one-hot encoding des variables catégorielles
from sklearn.impute import SimpleImputer # pour l'imputation des valeurs manquantes
from sklearn.base import BaseEstimator, TransformerMixin # pour créer un transformer personnalisé (IQR)
from src.config import CLEAN_CSV_PATH, FEATURES_CSV_PATH  # chemins définis dans config.py

# --- Création du dossier features s'il n'existe pas ---
os.makedirs(os.path.dirname(FEATURES_CSV_PATH), exist_ok=True)

# --- Étape 1 : Chargement des données nettoyées ---
df = pd.read_csv(CLEAN_CSV_PATH)
print("Données chargées :", df.shape)

# --- Étape 2 : Suppression des colonnes non pertinentes ---
# Les identifiants et timestamps ne servent pas pour le ML
df = df.drop(columns=['event_id', 'event_time', 'order_id', 'customer_customer_id'])

# --- Étape 3 : Gestion des outliers avec IQR ---
class IQR(BaseEstimator, TransformerMixin):
    """Transformer basé sur l'écart interquartile pour limiter les outliers"""
    def fit(self, X, y=None):
        self.q1 = X.quantile(0.25)
        self.q3 = X.quantile(0.75)
        self.iqr = self.q3 - self.q1
        return self

    def transform(self, X, y=None):
        def clip_outliers(col):
            lower = self.q1[col.name] - 1.5 * self.iqr[col.name]
            upper = self.q3[col.name] + 1.5 * self.iqr[col.name]
            return col.clip(lower, upper)
        return X.apply(clip_outliers)

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
iqr_transformer = IQR()
df[numeric_cols] = iqr_transformer.fit_transform(df[numeric_cols])

# --- Étape 4 : Imputation des valeurs manquantes ---
# Numériques : mediane, Catégorielles : mode
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

categorical_cols = df.select_dtypes(include=['object', 'category']).columns
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# --- Étape 5 : Encodage des variables catégorielles (OneHot) ---
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
encoded_cats = encoder.fit_transform(df[categorical_cols])
encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))
df = df.drop(columns=categorical_cols).reset_index(drop=True)
df = pd.concat([df, encoded_cat_df], axis=1)

# --- Étape 6 : Normalisation des variables numériques (RobustScaler) ---
scaler = RobustScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# --- Étape 7 : Sauvegarde des features finales ---
# Ici on sauvegarde le dataset complet (train+test) pour le ML
df.to_csv(FEATURES_CSV_PATH, index=False)
print(f"Features prêtes pour le ML sauvegardées dans {FEATURES_CSV_PATH}")
