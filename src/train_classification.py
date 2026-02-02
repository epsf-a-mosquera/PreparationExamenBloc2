# ExamenBloc2/src/train_classification.py
"""
Etape 1 : Préparation des features pour le Machine Learning.

Ce script :
1. Charge les données nettoyées générées par extract_transform.py
2. Supprime les colonnes non pertinentes pour le ML
3. Impute les valeurs manquantes
4. Traite les outliers (IQR)
5. Encode les variables catégorielles (OneHotEncoding)
6. Normalise les variables numériques (RobustScaler)
7. Sauvegarde les features finales dans CLASSIFICATION_FEATURES_TRAIN_CSV_PATH et CLASSIFICATION_FEATURES_TEST_CSV_PATH

Etape 2 : Entraînement du modèle ML de classification sur les features préparées.

IMPORTANT (Option 2 / joblib) :
- La classe IQR doit rester au niveau module (top-level) pour être importable par joblib :
  joblib aura besoin de retrouver src.train_classification.IQR au chargement.
- Tout le reste est placé dans main() pour éviter que l'entraînement se relance lors d'un import.
"""

import os
import json
import sys
import platform
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
import numpy as np

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin  # pour créer un transformer personnalisé (IQR)
from sklearn.model_selection import train_test_split      # split avant preprocessing (anti leakage)
from sklearn.impute import SimpleImputer                  # imputation des valeurs manquantes
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    log_loss,
    balanced_accuracy_score,
    matthews_corrcoef,
    brier_score_loss,
)

import joblib

from src.config import (
    CLEAN_CSV_PATH,
    CLASSIFICATION_FEATURES_TRAIN_CSV_PATH,
    CLASSIFICATION_FEATURES_TEST_CSV_PATH,
    ARTIFACTS_DIR,
    CLASSIFICATION_MODEL_PATH,
    CLASSIFICATION_MODEL_META_PATH,
)

# -------------------------------------------------------------------
# Transformer IQR (TOP-LEVEL)
# -------------------------------------------------------------------
# IMPORTANT: doit être au niveau module pour que joblib puisse retrouver
# src.train_classification.IQR au moment du load().
class IQR(BaseEstimator, TransformerMixin):
    """Transformer basé sur l'écart interquartile pour limiter les outliers (clipping)."""

    def fit(self, X, y=None):
        # X attendu: DataFrame pandas
        self.q1 = X.quantile(0.25)
        self.q3 = X.quantile(0.75)
        self.iqr = self.q3 - self.q1
        return self

    def transform(self, X, y=None):
        # On applique un clipping colonne par colonne selon [Q1-1.5*IQR, Q3+1.5*IQR]
        def clip_outliers(col):
            lower = self.q1[col.name] - 1.5 * self.iqr[col.name]
            upper = self.q3[col.name] + 1.5 * self.iqr[col.name]
            return col.clip(lower, upper)

        return X.apply(clip_outliers)


def main():
    # ---------------------------------------------------------------
    # Dossiers de sortie (features / artifacts)
    # ---------------------------------------------------------------
    os.makedirs(os.path.dirname(CLASSIFICATION_FEATURES_TRAIN_CSV_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CLASSIFICATION_FEATURES_TEST_CSV_PATH), exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CLASSIFICATION_MODEL_META_PATH), exist_ok=True)

    # ---------------------------------------------------------------
    # Étape 1 : Chargement des données nettoyées
    # ---------------------------------------------------------------
    df = pd.read_csv(CLEAN_CSV_PATH)
    print("Données chargées :", df.shape)

    # ---------------------------------------------------------------
    # Étape 2 : Suppression des colonnes non pertinentes
    # ---------------------------------------------------------------
    # Les identifiants et timestamps ne servent pas pour le ML
    df = df.drop(columns=["event_id", "event_time", "order_id", "customer_customer_id"])

    # ---------------------------------------------------------------
    # Split AVANT preprocessing (anti data leakage)
    # ---------------------------------------------------------------
    target_col = "order_is_returned"
    y = df[target_col].astype(int)       # bool -> int
    X = df.drop(columns=[target_col])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    df_train = X_train_raw.copy()
    df_test = X_test_raw.copy()

    # ---------------------------------------------------------------
    # Colonnes numériques/catégorielles/bool
    # ---------------------------------------------------------------
    # (On garde ta logique actuelle pour minimiser les changements)
    numeric_cols = df_train.select_dtypes(include=["int64", "float64"]).columns
    bool_cols_tmp = df_train.select_dtypes(include=["bool"]).columns
    numeric_cols = numeric_cols.drop(bool_cols_tmp, errors="ignore")

    categorical_cols = df_train.select_dtypes(include=["object", "category"]).columns

    # ---------------------------------------------------------------
    # Étape 4 : Imputation des valeurs manquantes
    # ---------------------------------------------------------------
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    if len(numeric_cols) > 0:
        df_train[numeric_cols] = num_imputer.fit_transform(df_train[numeric_cols])
        df_test[numeric_cols] = num_imputer.transform(df_test[numeric_cols])

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
    # bool -> int (0/1) APRÈS imputation
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

        df_train = df_train.drop(columns=categorical_cols)
        df_test = df_test.drop(columns=categorical_cols)

        df_train = pd.concat([df_train, encoded_train_df], axis=1)
        df_test = pd.concat([df_test, encoded_test_df], axis=1)
    else:
        feature_names = []
        encoder = None

    # ---------------------------------------------------------------
    # Étape 6 : Normalisation RobustScaler des variables numériques
    # ---------------------------------------------------------------
    scaler = RobustScaler()
    if len(numeric_cols) > 0:
        df_train[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
        df_test[numeric_cols] = scaler.transform(df_test[numeric_cols])

    print("Train shape:", df_train.shape, "Test shape:", df_test.shape)
    print("NaN train:", int(df_train.isna().sum().sum()), "NaN test:", int(df_test.isna().sum().sum()))
    assert list(df_train.columns) == list(df_test.columns), "Colonnes train/test différentes après preprocessing"

    # ---------------------------------------------------------------
    # Étape 7 : Sauvegarde des features finales
    # ---------------------------------------------------------------
    df_train_full = pd.concat([df_train, y_train.rename(target_col)], axis=1)
    df_test_full = pd.concat([df_test, y_test.rename(target_col)], axis=1)

    df_train_full.to_csv(CLASSIFICATION_FEATURES_TRAIN_CSV_PATH, index=False)
    df_test_full.to_csv(CLASSIFICATION_FEATURES_TEST_CSV_PATH, index=False)
    print(
        f"Features prêtes pour le ML sauvegardées dans "
        f"{CLASSIFICATION_FEATURES_TRAIN_CSV_PATH} et {CLASSIFICATION_FEATURES_TEST_CSV_PATH}"
    )

    # ===============================================================
    # Etape 2 - Entraînement du modèle ML de classification
    # ===============================================================

    print("Features chargées :", df_train_full.shape, df_test_full.shape)

    X_train = df_train
    X_test = df_test

    model = LogisticRegression(max_iter=2000, solver="liblinear")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    print(f"Score du modèle sur le jeu de test : {score:.4f}")

    # ---------------------------------------------------------------
    # Sauvegarde du modèle + preprocess (bundle)
    # ---------------------------------------------------------------
    # NOTE: Ici on dump un dict (bundle), pas un model brut.
    # Dans kafka_consumer, il faudra charger puis faire bundle["model"].
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

    joblib.dump(bundle, CLASSIFICATION_MODEL_PATH)
    print(f"Modèle sauvegardé dans {CLASSIFICATION_MODEL_PATH}")

    # ---------------------------------------------------------------
    # Sauvegarde métriques & métadonnées
    # ---------------------------------------------------------------
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    mcc = float(matthews_corrcoef(y_test, y_pred))

    try:
        roc_auc = float(roc_auc_score(y_test, y_proba))
    except ValueError:
        roc_auc = None

    try:
        pr_auc = float(average_precision_score(y_test, y_proba))
    except ValueError:
        pr_auc = None

    try:
        ll = float(log_loss(y_test, y_proba, labels=[0, 1]))
    except ValueError:
        ll = None

    try:
        brier = float(brier_score_loss(y_test, y_proba))
    except ValueError:
        brier = None

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    coef_series = pd.Series(model.coef_[0], index=X_train.columns).sort_values(ascending=False)
    top_positive = [{"feature": k, "coef": float(v)} for k, v in coef_series.head(10).items()]
    top_negative = [{"feature": k, "coef": float(v)} for k, v in coef_series.tail(10).items()]

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
            "features_train_csv_path": str(CLASSIFICATION_FEATURES_TRAIN_CSV_PATH),
            "features_test_csv_path": str(CLASSIFICATION_FEATURES_TEST_CSV_PATH),
            "n_rows_total_train": int(df_train_full.shape[0]),
            "n_rows_total_test": int(df_test_full.shape[0]),
            "n_rows_total": int(df_train_full.shape[0] + df_test_full.shape[0]),
            "n_features": int(X_train.shape[1]),
            "feature_names": list(X_train.columns),
            "target_name": "order_is_returned",
            "target_positive_label": 1,
            "class_balance_total": {
                "positive_rate": float(y.mean()),
                "n_positive": int((y == 1).sum()),
                "n_negative": int((y == 0).sum()),
            },
            "split": {
                "test_size": 0.2,
                "random_state": 42,
                "train_rows": int(X_train.shape[0]),
                "test_rows": int(X_test.shape[0]),
                "class_balance_test": {
                    "positive_rate": float(y_test.mean()),
                    "n_positive": int((y_test == 1).sum()),
                    "n_negative": int((y_test == 0).sum()),
                },
            },
        },
        "model": {
            "artifact_model_path": str(CLASSIFICATION_MODEL_PATH),
            "type": "sklearn.linear_model.LogisticRegression",
            "params": model.get_params(),
        },
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "balanced_accuracy": bal_acc,
            "matthews_corrcoef": mcc,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "log_loss": ll,
            "brier_score": brier,
            "threshold": 0.5,
        },
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": [[tn, fp], [fn, tp]],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "classification_report": report,
        "interpretability": {
            "top_positive_coefs": top_positive,
            "top_negative_coefs": top_negative,
        },
    }

    with open(CLASSIFICATION_MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Métriques + métadonnées sauvegardées en JSON dans {CLASSIFICATION_MODEL_META_PATH}")


if __name__ == "__main__":
    main()
