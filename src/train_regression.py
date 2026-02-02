# PreparationExamenBloc2/src/train_regression.py
"""
Etape 1 : Préparation des features pour le Machine Learning (Régression)

Ce script :
1. Charge les données nettoyées générées par extract_transform.py
2. Supprime les colonnes non pertinentes pour le ML
3. Split train/test AVANT preprocessing (anti data leakage)
4. Impute les valeurs manquantes
5. Traite les outliers (IQR)
6. Encode les variables catégorielles (OneHotEncoding)
7. Standardise les variables numériques (StandardScaler)
8. Sauvegarde les features train/test dans :
   - REGRESSION_FEATURES_TRAIN_CSV_PATH
   - REGRESSION_FEATURES_TEST_CSV_PATH

Etape 2 : Entraînement du modèle de régression

IMPORTANT (Option 2 / joblib) :
- La classe IQR doit être "importable" au moment du joblib.load().
- Donc IQR DOIT rester au niveau module (top-level) : src.train_regression.IQR
- Tout le reste doit être dans main() pour éviter qu'un simple import relance l'entraînement.
"""

import os
import json
import sys
import platform
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
import sklearn
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

from src.config import (
    CLEAN_CSV_PATH,
    REGRESSION_FEATURES_TRAIN_CSV_PATH,
    REGRESSION_FEATURES_TEST_CSV_PATH,
    ARTIFACTS_DIR,
    REGRESSION_MODEL_PATH,
    REGRESSION_MODEL_META_PATH,
)

from src.eco_impact import track_phase # decorator pour la mesure de carbone

# -------------------------------------------------------------------
# Transformer IQR (TOP-LEVEL)
# -------------------------------------------------------------------
# IMPORTANT : doit être au niveau module pour que joblib puisse retrouver
# src.train_regression.IQR au moment du load().
class IQR(BaseEstimator, TransformerMixin):
    """Transformer basé sur l'écart interquartile pour limiter les outliers (clipping)."""

    def fit(self, X, y=None):
        # X attendu: DataFrame pandas
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


def main():
    # ===============================================================
    
    with track_phase("ml_train_regression"):
    
        # ---------------------------------------------------------------
        # Création des dossiers de sortie (features + artifacts)
        # ---------------------------------------------------------------
        os.makedirs(os.path.dirname(REGRESSION_FEATURES_TRAIN_CSV_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(REGRESSION_FEATURES_TEST_CSV_PATH), exist_ok=True)
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(REGRESSION_MODEL_META_PATH), exist_ok=True)

        # ---------------------------------------------------------------
        # Étape 1 : Chargement des données nettoyées
        # ---------------------------------------------------------------
        df = pd.read_csv(CLEAN_CSV_PATH)
        print("Données chargées :", df.shape)

        # ---------------------------------------------------------------
        # Étape 2 : Suppression des colonnes non pertinentes
        # ---------------------------------------------------------------
        df = df.drop(columns=["event_id", "event_time", "order_id", "customer_customer_id"])

        # ---------------------------------------------------------------
        # Split AVANT preprocessing (anti data leakage)
        # ---------------------------------------------------------------
        target_col = "order_basket_value"
        y = df[target_col].astype(float)
        X = df.drop(columns=[target_col])

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # DF séparés pour fit sur train et transform sur test
        df_train = X_train_raw.copy()
        df_test = X_test_raw.copy()

        # ---------------------------------------------------------------
        # Colonnes numériques / catégorielles / bool
        # ---------------------------------------------------------------
        # Variables numériques continues (on exclut explicitement les bool)
        numeric_cols = df_train.select_dtypes(include=["int64", "float64"]).columns
        bool_cols_tmp = df_train.select_dtypes(include=["bool"]).columns
        numeric_cols = numeric_cols.drop(bool_cols_tmp, errors="ignore")

        categorical_cols = df_train.select_dtypes(include=["object", "category"]).columns

        # ---------------------------------------------------------------
        # Étape 4 : Imputation des valeurs manquantes
        # ---------------------------------------------------------------
        num_imputer = SimpleImputer(strategy="median")
        cat_imputer = SimpleImputer(strategy="most_frequent")

        # Numériques : imputation médiane
        if len(numeric_cols) > 0:
            df_train[numeric_cols] = num_imputer.fit_transform(df_train[numeric_cols])
            df_test[numeric_cols] = num_imputer.transform(df_test[numeric_cols])

        # Catégorielles : imputation mode
        if len(categorical_cols) > 0:
            df_train[categorical_cols] = cat_imputer.fit_transform(df_train[categorical_cols])
            df_test[categorical_cols] = cat_imputer.transform(df_test[categorical_cols])

        # ---------------------------------------------------------------
        # Étape 3 : IQR APRÈS imputation (clipping outliers)
        # ---------------------------------------------------------------
        iqr_transformer = IQR()
        if len(numeric_cols) > 0:
            df_train[numeric_cols] = iqr_transformer.fit_transform(df_train[numeric_cols])
            df_test[numeric_cols] = iqr_transformer.transform(df_test[numeric_cols])

        # ---------------------------------------------------------------
        # bool -> int (0/1)
        # ---------------------------------------------------------------
        bool_cols = df_train.select_dtypes(include=["bool"]).columns
        if len(bool_cols) > 0:
            df_train[bool_cols] = df_train[bool_cols].astype(int)
            df_test[bool_cols] = df_test[bool_cols].astype(int)

        # ---------------------------------------------------------------
        # Étape 5 : Encodage OneHot des colonnes catégorielles
        # ---------------------------------------------------------------
        encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")

        if len(categorical_cols) > 0:
            encoded_train = encoder.fit_transform(df_train[categorical_cols])
            encoded_test = encoder.transform(df_test[categorical_cols])

            feature_names = encoder.get_feature_names_out(categorical_cols)

            encoded_train_df = pd.DataFrame(encoded_train, index=df_train.index, columns=feature_names)
            encoded_test_df = pd.DataFrame(encoded_test, index=df_test.index, columns=feature_names)

            # On retire les colonnes catégorielles initiales puis on concatène les colonnes encodées
            df_train = df_train.drop(columns=categorical_cols)
            df_test = df_test.drop(columns=categorical_cols)

            df_train = pd.concat([df_train, encoded_train_df], axis=1)
            df_test = pd.concat([df_test, encoded_test_df], axis=1)
        else:
            # Pas de colonnes catégorielles -> pas d'encodage
            encoder = None

        # ---------------------------------------------------------------
        # Étape 6 : Standardisation des variables numériques (StandardScaler)
        # ---------------------------------------------------------------
        # Important : on standardise uniquement les variables numériques continues.
        scaler = StandardScaler()
        if len(numeric_cols) > 0:
            df_train[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
            df_test[numeric_cols] = scaler.transform(df_test[numeric_cols])

        # Vérification finale "sanitaire"
        print("Train shape:", df_train.shape, "Test shape:", df_test.shape)
        print("NaN train:", int(df_train.isna().sum().sum()), "NaN test:", int(df_test.isna().sum().sum()))
        assert list(df_train.columns) == list(df_test.columns), "Colonnes train/test différentes après preprocessing"

        # ---------------------------------------------------------------
        # Étape 7 : Sauvegarde des features finales (train/test)
        # ---------------------------------------------------------------
        df_train_full = pd.concat([df_train, y_train.rename(target_col)], axis=1)
        df_test_full = pd.concat([df_test, y_test.rename(target_col)], axis=1)

        df_train_full.to_csv(REGRESSION_FEATURES_TRAIN_CSV_PATH, index=False)
        df_test_full.to_csv(REGRESSION_FEATURES_TEST_CSV_PATH, index=False)

        print(
            f"Features prêtes pour le ML sauvegardées dans "
            f"{REGRESSION_FEATURES_TRAIN_CSV_PATH} et {REGRESSION_FEATURES_TEST_CSV_PATH}"
        )

        # IMPORTANT : on réutilise les bons X_train/X_test pour l'entraînement (déjà preprocessés)
        X_train = df_train
        X_test = df_test

        # ===============================================================
        # Etape 2 - Entraînement du modèle ML de régression
        # ===============================================================
        print("Features chargées :", df_train_full.shape, df_test_full.shape)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score = model.score(X_test, y_test)
        print(f"Score du modèle sur le jeu de test : {score:.4f}")

        # ---------------------------------------------------------------
        # Sauvegarde du modèle + preprocess (bundle)
        # ---------------------------------------------------------------
        bundle = {
            "model": model,
            "preprocess": {
                "iqr_transformer": iqr_transformer,
                "num_imputer": num_imputer,
                "cat_imputer": cat_imputer,
                "encoder": encoder,
                "scaler": scaler,
                "numeric_cols": list(numeric_cols),
                "categorical_cols": list(categorical_cols),
                "bool_cols": list(bool_cols),
                "feature_names_after_preprocess": list(X_train.columns),
            },
        }

        joblib.dump(bundle, REGRESSION_MODEL_PATH)
        print(f"Modèle sauvegardé dans {REGRESSION_MODEL_PATH}")

        # ---------------------------------------------------------------
        # Sauvegarde des métriques & métadonnées (JSON)
        # ---------------------------------------------------------------
        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(mean_absolute_percentage_error(y_test, y_pred))

        residuals = y_test - y_pred
        residuals_summary = {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
        }

        interpretability = None
        if hasattr(model, "coef_"):
            coef_series = pd.Series(model.coef_, index=X_train.columns).sort_values(ascending=False)
            interpretability = {
                "top_positive_coefs": [{"feature": k, "coef": float(v)} for k, v in coef_series.head(10).items()],
                "top_negative_coefs": [{"feature": k, "coef": float(v)} for k, v in coef_series.tail(10).items()],
            }

        meta = {
            "run": {
                "run_id": str(uuid4()),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "sklearn_version": sklearn.__version__,
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__,
            },
            "data": {
                "features_train_csv_path": str(REGRESSION_FEATURES_TRAIN_CSV_PATH),
                "features_test_csv_path": str(REGRESSION_FEATURES_TEST_CSV_PATH),
                "n_rows_total_train": int(df_train_full.shape[0]),
                "n_rows_total_test": int(df_test_full.shape[0]),
                "n_rows_total": int(df_train_full.shape[0] + df_test_full.shape[0]),
                "n_features": int(X_train.shape[1]),
                "feature_names": list(X_train.columns),
                "target_name": target_col,
                "split": {
                    "test_size": 0.2,
                    "random_state": 42,
                    "train_rows": int(X_train.shape[0]),
                    "test_rows": int(X_test.shape[0]),
                },
                "target_summary_total": {
                    "mean": float(y.mean()),
                    "std": float(y.std()),
                    "min": float(y.min()),
                    "max": float(y.max()),
                },
                "target_summary_test": {
                    "mean": float(y_test.mean()),
                    "std": float(y_test.std()),
                    "min": float(y_test.min()),
                    "max": float(y_test.max()),
                },
            },
            "model": {
                "artifact_model_path": str(REGRESSION_MODEL_PATH),
                "type": model.__class__.__module__ + "." + model.__class__.__name__,
                "params": model.get_params() if hasattr(model, "get_params") else None,
            },
            "metrics": {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "mape": mape,
            },
            "residuals": residuals_summary,
            "interpretability": interpretability,
        }

        with open(REGRESSION_MODEL_META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"Métriques + métadonnées sauvegardées en JSON dans {REGRESSION_MODEL_META_PATH}")


if __name__ == "__main__":
    main()
