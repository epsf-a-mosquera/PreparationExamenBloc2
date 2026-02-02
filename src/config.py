#ExamenBloc2/src/config.py
"""
config.py
---------
Ce fichier centralise toute la configuration du projet.

Pourquoi c'est important :
- On utilise des variables d'environnement via un fichier .env.
- On évite de "hardcoder" les mots de passe dans le code.
- Tout le projet lit la config au même endroit => maintenance facile.
"""

# On importe "os" pour lire les variables d'environnement.
import os
# On importe "Path" pour manipuler les chemins de fichiers proprement (cross-platform).
from pathlib import Path
# On importe load_dotenv pour charger le fichier .env automatiquement.
from dotenv import load_dotenv
# Charge les variables d'environnement depuis ".env" si présent.
# Cela permet d'utiliser os.getenv(...) ensuite.
load_dotenv()


# ---------------------------
# Chemins du projet
# ---------------------------
# BASE_DIR = dossier racine du projet (ExamenBloc2/).
PROJECT_DIR = Path(__file__).resolve().parents[1]
# Dossier data/
DATA_DIR = PROJECT_DIR / "data"
# Dossier où on met le JSON brut
RAW_DIR = DATA_DIR / "raw"
# Dossier où on met les données nettoyées
PROCESSED_DIR = DATA_DIR / "processed"
# Dossier où on sauvegarde les features prêtes pour le ML
FEATURES_DIR = DATA_DIR / "features"
# Dossier où on sauvegarde les élemtents générés par le model ml exemple : modèles, métriques, etc.
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"


# ---------------------------
# Fichiers de données
# ---------------------------
# Dataset brut (JSON Lines : 1 JSON par ligne)
RAW_JSONL_PATH = RAW_DIR / "orders_events.jsonl"
# Dataset nettoyé (CSV)
CLEAN_CSV_PATH = PROCESSED_DIR / "orders_events_cleaned.csv"
# Dataset nettoyé (parquet)
CLEAN_PARQUET_PATH = PROCESSED_DIR / "orders_events_cleaned.parquet"
# Dataset nettoyé (json)
CLEAN_JSONL_PATH = PROCESSED_DIR / "orders_events_cleaned.json"
# Dataset des features prêtes pour le ML de classification (CSV) avec normalisation robuste des variables numériques
CLASSIFICATION_FEATURES_TRAIN_CSV_PATH = FEATURES_DIR / "classification_orders_events_features_train.csv"
CLASSIFICATION_FEATURES_TEST_CSV_PATH = FEATURES_DIR / "classification_orders_events_features_test.csv"
# Dataset des features prêtes pour le ML de regression (CSV) avec standardisation robuste des variables numériques
REGRESSION_FEATURES_TRAIN_CSV_PATH = FEATURES_DIR / "regression_orders_events_features_train.csv"
REGRESSION_FEATURES_TEST_CSV_PATH = FEATURES_DIR / "regression_orders_events_features_test.csv"


#---------------------------
# Model ML et métriques
#---------------------------
# Dossier des artefacts (modèles, métriques, etc.)
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
# Chemin du modèle sauvegardé avec joblib
CLASSIFICATION_MODEL_PATH = ARTIFACTS_DIR / "classification_model.joblib"
REGRESSION_MODEL_PATH = ARTIFACTS_DIR / "regression_model.joblib"
# Chemin pour sauvegarder des infos (métriques et métadonnées) au format JSON
CLASSIFICATION_MODEL_META_PATH = ARTIFACTS_DIR / "classification_model_meta.json"
REGRESSION_MODEL_META_PATH = ARTIFACTS_DIR / "regression_model_meta.json"

# ---------------------------
# Configuration DB MySQL
# ---------------------------
# On lit les variables depuis l'environnement.
MYSQL_USER = os.getenv("MYSQL_USER", "exam")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "exampwd")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "examdb")

# NB : dans docker-compose, le service MySQL s'appelle "mysql".
# Depuis la machine host, on passe par localhost + port exposé.
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))

# ---------------------------
# URL SQLAlchemy pour MySQL.
# ---------------------------
# On utilise "pymysql" comme driver Python.
SQLALCHEMY_DATABASE_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
)

# ---------------------------
# Kafka
# ---------------------------
# Dans docker-compose : kafka est accessible depuis les containers sous "kafka:9092".
# Depuis la machine host, on peut accéder via "localhost:9092" (port exposé).
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:29092")
# Topic Kafka pour les événements
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "events")


