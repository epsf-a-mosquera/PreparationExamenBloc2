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

Caractéristiques données netoyées :
Types de données finaux :
event_id                       string[python]
event_time                datetime64[ns, UTC]
order_id                       string[python]
customer_customer_id           string[python]
customer_country                     category
order_device                         category
order_channel                        category
order_main_category                  category
order_n_items                           int64
order_basket_value                    float64
order_shipping_fee                    float64
order_discount                        float64
order_order_total                     float64
order_is_returned                        bool
event_year                              int32
event_month                             int32
event_day                               int32
event_hour                              int32
price_average_per_item                float64

                                    event_id                           event_time   order_id customer_customer_id customer_country order_device order_channel order_main_category  order_n_items  order_basket_value  order_shipping_fee  order_discount  order_order_total order_is_returned  event_year  event_month    event_day   event_hour  price_average_per_item
count                                   4000                                 4000       4000                 4000             4000         4000          4000                4000    4000.000000         4000.000000         4000.000000     4000.000000        4000.000000              4000      4000.0  4000.000000  4000.000000  4000.000000             4000.000000
unique                                  4000                                  NaN       2714                  599                5            3             5                   5            NaN                 NaN                 NaN             NaN                NaN                 2         NaN          NaN          NaN          NaN                     NaN
top     83442dbd-8504-49f4-af09-1986a21e3e8f                                  NaN  ORD-02461            CUST-0471               BE       mobile           ads              sports            NaN                 NaN                 NaN             NaN                NaN             False         NaN          NaN          NaN          NaN                     NaN
freq                                       1                                  NaN          6                   16              812         1364           935                 869            NaN                 NaN                 NaN             NaN                NaN              3127         NaN          NaN          NaN          NaN                     NaN
mean                                     NaN  2026-01-16 14:09:58.734317568+00:00        NaN                  NaN              NaN          NaN           NaN                 NaN       4.467500          253.481282            0.803818        4.732208         249.480648               NaN      2026.0     1.001000    16.063750    11.396500               86.382996
min                                      NaN     2026-01-01 00:57:38.968354+00:00        NaN                  NaN              NaN          NaN           NaN                 NaN       1.000000           10.060000            0.000000        0.000000           1.000000               NaN      2026.0     1.000000     1.000000     0.000000                1.306250
25%                                      NaN  2026-01-08 16:31:38.993940480+00:00        NaN                  NaN              NaN          NaN           NaN                 NaN       2.000000          129.060000            0.000000        0.000000         125.497500               NaN      2026.0     1.000000     8.000000     5.000000               29.255750
50%                                      NaN  2026-01-16 15:46:39.016147968+00:00        NaN                  NaN              NaN          NaN           NaN                 NaN       4.000000          252.105000            0.000000        0.000000         247.210000               NaN      2026.0     1.000000    16.000000    11.000000               56.407917
75%                                      NaN  2026-01-24 07:34:54.019168256+00:00        NaN                  NaN              NaN          NaN           NaN                 NaN       6.000000          377.362500            0.000000        6.362500         371.500000               NaN      2026.0     1.000000    24.000000    17.000000              103.049167
max                                      NaN     2026-02-01 00:44:39.009341+00:00        NaN                  NaN              NaN          NaN           NaN                 NaN       8.000000          499.960000            8.980000       30.000000         499.900000               NaN      2026.0     2.000000    31.000000    23.000000              498.930000
std                                      NaN                                  NaN        NaN                  NaN              NaN          NaN           NaN                 NaN       2.299297          142.047007            2.071895        8.487850         141.048239               NaN         0.0     0.031611     8.951967     6.920496               92.923256

"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, RobustScaler # RobustScaler pour la normalisation robuste des données numériques et one-hot encoding des variables catégorielles
from sklearn.impute import SimpleImputer # pour l'imputation des valeurs manquantes
from sklearn.base import BaseEstimator, TransformerMixin # pour créer un transformer personnalisé (IQR)
from sklearn.model_selection import train_test_split  # <-- ajouté (split avant preprocessing)
from src.config import CLEAN_CSV_PATH, CLASSIFICATION_FEATURES_TRAIN_CSV_PATH, CLASSIFICATION_FEATURES_TEST_CSV_PATH  # chemins définis dans config.py

# --- Création du dossier features s'il n'existe pas ---
os.makedirs(os.path.dirname(CLASSIFICATION_FEATURES_TRAIN_CSV_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CLASSIFICATION_FEATURES_TEST_CSV_PATH), exist_ok=True)

# --- Étape 1 : Chargement des données nettoyées ---
df = pd.read_csv(CLEAN_CSV_PATH)
print("Données chargées :", df.shape)

# --- Étape 2 : Suppression des colonnes non pertinentes ---
# Les identifiants et timestamps ne servent pas pour le ML
df = df.drop(columns=['event_id', 'event_time', 'order_id', 'customer_customer_id'])

# --- Split AVANT preprocessing (anti data leakage) ---
target_col = "order_is_returned"
y = df[target_col].astype(int)
X = df.drop(columns=[target_col])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

df_train = X_train.copy()
df_test  = X_test.copy()

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

numeric_cols = df_train.select_dtypes(include=['int64', 'float64']).columns
bool_cols_tmp = df_train.select_dtypes(include=['bool']).columns
numeric_cols = numeric_cols.drop(bool_cols_tmp, errors="ignore")


# --- Étape 4 : Imputation des valeurs manquantes ---
# Numériques : mediane, Catégorielles : mode
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

categorical_cols = df_train.select_dtypes(include=['object', 'category']).columns

if len(numeric_cols) > 0:
    df_train[numeric_cols] = num_imputer.fit_transform(df_train[numeric_cols])
    df_test[numeric_cols]  = num_imputer.transform(df_test[numeric_cols])

if len(categorical_cols) > 0:
    df_train[categorical_cols] = cat_imputer.fit_transform(df_train[categorical_cols])
    df_test[categorical_cols]  = cat_imputer.transform(df_test[categorical_cols])

#--- IQR APRÈS imputation ---
iqr_transformer = IQR()
if len(numeric_cols) > 0:
    df_train[numeric_cols] = iqr_transformer.fit_transform(df_train[numeric_cols])
    df_test[numeric_cols]  = iqr_transformer.transform(df_test[numeric_cols])

# --- bool -> int (0/1) APRÈS imputation ---
bool_cols = df_train.select_dtypes(include=['bool']).columns
if len(bool_cols) > 0:
    df_train[bool_cols] = df_train[bool_cols].astype(int)
    df_test[bool_cols]  = df_test[bool_cols].astype(int)

# --- Étape 5 : Encodage des variables catégorielles (OneHot) ---
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

if len(categorical_cols) > 0:
    encoded_train = encoder.fit_transform(df_train[categorical_cols])
    encoded_test  = encoder.transform(df_test[categorical_cols])

    feature_names = encoder.get_feature_names_out(categorical_cols)

    encoded_train_df = pd.DataFrame(encoded_train, index=df_train.index, columns=feature_names)
    encoded_test_df  = pd.DataFrame(encoded_test,  index=df_test.index,  columns=feature_names)

    df_train = df_train.drop(columns=categorical_cols)
    df_test  = df_test.drop(columns=categorical_cols)

    df_train = pd.concat([df_train, encoded_train_df], axis=1)
    df_test  = pd.concat([df_test,  encoded_test_df],  axis=1)
else:
    feature_names = []
    encoder = None

# --- Étape 6 : Normalisation des variables numériques (RobustScaler) ---
scaler = RobustScaler()
if len(numeric_cols) > 0:
    df_train[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
    df_test[numeric_cols]  = scaler.transform(df_test[numeric_cols])

print("Train shape:", df_train.shape, "Test shape:", df_test.shape)
print("NaN train:", int(df_train.isna().sum().sum()), "NaN test:", int(df_test.isna().sum().sum()))
assert list(df_train.columns) == list(df_test.columns), "Colonnes train/test différentes après preprocessing"


# --- Étape 7 : Sauvegarde des features finales ---
df_train_full = pd.concat([df_train, y_train.rename(target_col)], axis=1)
df_test_full  = pd.concat([df_test,  y_test.rename(target_col)], axis=1)

df_train_full.to_csv(CLASSIFICATION_FEATURES_TRAIN_CSV_PATH, index=False)
df_test_full.to_csv(CLASSIFICATION_FEATURES_TEST_CSV_PATH, index=False)
print(f"Features prêtes pour le ML sauvegardées dans {CLASSIFICATION_FEATURES_TRAIN_CSV_PATH} et {CLASSIFICATION_FEATURES_TEST_CSV_PATH}")



"""
Etape 2 - Entraînement du modèle ML de classification sur les features préparées.

Ce script :
1. Charge les features préparées par prepare_features.py (à partir du même df)
2. Sépare X et y (variable cible : order_is_returned)
3. Sépare train/test
4. Entraîne un modèle LogisticRegression
5. Évalue le modèle sur le test
6. Stocker le modèle entraîné dans le dossier artifacts
7. Stocke les métriques d'évaluation dans un fichier texte dans artifacts
"""

import joblib
from src.config import ARTIFACTS_DIR
from sklearn.linear_model import LogisticRegression
from src.config import CLASSIFICATION_MODEL_PATH, CLASSIFICATION_MODEL_META_PATH

import json
import sys
import platform
from datetime import datetime, timezone
from uuid import uuid4

import sklearn
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
    brier_score_loss
)


# --- Étape 1 : Chargement des features ---
# df = pd.read_csv(CLASSIFICATION_FEATURES_TRAIN_CSV_PATH) --> pas necessaire de reimporter pandas car df est déjà défini plus haut
print("Features chargées :", df_train_full.shape, df_test_full.shape)

# --- Étape 2 : Séparation X / y ---
# X_train, X_test, y_train, y_test sont déjà définis plus haut (split AVANT preprocessing)
X_train = df_train
X_test  = df_test


# --- Étape 4 : Entraînement du modèle LogisticRegression ---
model = LogisticRegression(max_iter=2000, solver="liblinear")
model.fit(X_train, y_train)

# --- Étape 5 : Prédiction et score ---
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
print(f"Score du modèle sur le jeu de test : {score:.4f}")

# --- Étape 6 : Sauvegarde du modèle entraîné ---
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
model_path = CLASSIFICATION_MODEL_PATH
joblib.dump(
    {
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
    },
    model_path
)

print(f"Modèle sauvegardé dans {model_path}")


# --- Étape 7 : Sauvegarde des métriques & métadonnées au format JSON ---
metrics_path = CLASSIFICATION_MODEL_META_PATH
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

# Probabilités (utile pour AUC, log_loss, calibration)
y_proba = model.predict_proba(X_test)[:, 1] # cette ligne extrait les probabilités de la classe positive (1)

# Métriques principales (attention aux types numpy -> float/int python)
acc = float(accuracy_score(y_test, y_pred))
prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

# Métriques utiles si classes déséquilibrées
bal_acc = float(balanced_accuracy_score(y_test, y_pred))
mcc = float(matthews_corrcoef(y_test, y_pred))

# AUC ROC / PR AUC : peuvent échouer si y_test n’a qu’une seule classe
try:
    roc_auc = float(roc_auc_score(y_test, y_proba))
except ValueError:
    roc_auc = None

try:
    pr_auc = float(average_precision_score(y_test, y_proba))
except ValueError:
    pr_auc = None

# Log loss / Brier score : mesures de qualité des probabilités
try:
    ll = float(log_loss(y_test, y_proba, labels=[0, 1]))
except ValueError:
    ll = None

try:
    brier = float(brier_score_loss(y_test, y_proba))
except ValueError:
    brier = None

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = [int(x) for x in cm.ravel()]

# Report détaillé par classe (dict JSON-friendly)
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

# Interprétation rapide : top features (coefficients) si LogisticRegression
coef_series = pd.Series(model.coef_[0], index=X_train.columns).sort_values(ascending=False)
top_positive = [{"feature": k, "coef": float(v)} for k, v in coef_series.head(10).items()]
top_negative = [{"feature": k, "coef": float(v)} for k, v in coef_series.tail(10).items()]

# Métadonnées run + data + modèle
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
        "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    },
    "classification_report": report,
    "interpretability": {
        "top_positive_coefs": top_positive,
        "top_negative_coefs": top_negative,
    },
}

# Écriture JSON
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"Métriques + métadonnées sauvegardées en JSON dans {metrics_path}")
