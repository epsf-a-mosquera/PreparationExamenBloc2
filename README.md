# Projet Bloc 2 — ETL & ML + SQL + Kafka + Eco-impact (CodeCarbon)

## 1) Objectif du projet

Ce projet met en œuvre un pipeline **Data Engineer + ML** complet, de bout en bout :

1. **Génération** de données réalistes (JSON Lines) simulant des événements de commandes e-commerce.  
2. **Exploration / Transformation / Nettoyage** (ETL) via `pandas` (types, doublons, valeurs manquantes, features).  
3. **Chargement** (ingestion) dans une base **MySQL** via **SQLAlchemy ORM**.  
4. **Entraînement** de modèles **Machine Learning** avec `scikit-learn` :
   - Classification : prédire la probabilité de retour (`return_proba`)
   - Régression : exemple de modèle (prédiction de valeur panier, ou autre cible)  
5. **Traitement streaming** via **Kafka** :
   - Producer : envoie des événements “clean”
   - Consumer : ingère en base + fait l’inférence ML + stocke les prédictions  
6. **Mesure de l’impact écologique** du pipeline (énergie / CO2e) via **CodeCarbon** et génération d’un rapport consolidé.

---

## 2) Architecture (vue globale)

### Flux Batch (ETL + DB + ML)
- `generate_data.py` → génère `data/raw/orders_events.jsonl`
- `extract_transform.py` → produit :
  - `data/processed/orders_events_cleaned.csv`
  - `data/processed/orders_events_cleaned.parquet`
  - `data/processed/orders_events_cleaned.json`
- `ingest.py` → charge le CSV clean dans MySQL (tables ORM)
- `train_classification.py` → entraîne + sauvegarde modèle + artefacts
- `train_regression.py` → entraîne + sauvegarde modèle + artefacts (exemple)

### Flux Streaming (Kafka)
- `kafka_producer.py` → lit `orders_events_cleaned.json` et envoie sur un topic Kafka
- `kafka_consumer.py` → consomme, transforme en ligne “clean”, ingère en DB, calcule `return_proba` via modèle joblib, stocke la prédiction
- `kafka_pipeline.py` → lance consumer + producer automatiquement (1 seule commande)

### Eco-impact (CodeCarbon)
- `eco_impact.py` → fournit `track_phase(...)` pour instrumenter les scripts
- CodeCarbon écrit les mesures dans `reports/emissions/`
- `eco_report.py` → fusionne tous les fichiers `emissions.csv*` puis génère :
  - `reports/eco_report.md`
  - `reports/eco_report.csv`

---

## 3) Contenu du repository

```
PreparationExamenBloc2/
├─ src/
│  ├─ config.py
│  ├─ generate_data.py
│  ├─ extract_transform.py
│  ├─ db_models.py
│  ├─ ingest.py
│  ├─ ingest_predictions.py
│  ├─ train_classification.py
│  ├─ train_regression.py
│  ├─ kafka_producer.py
│  ├─ kafka_consumer.py
│  ├─ kafka_pipeline.py
│  ├─ eco_impact.py
│  └─ eco_report.py
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ features/
├─ artifacts/
│  ├─ classification_model.joblib
│  ├─ classification_model_meta.json
│  ├─ regression_model.joblib
│  └─ regression_model_meta.json
└─ reports/
   ├─ emissions/
   ├─ eco_report.md
   └─ eco_report.csv
```

---

## 4) Prérequis

- Python 3.10+
- Docker + Docker Compose
- Un environnement virtuel Python (`venv`)
- Librairies principales :
  - `pandas`, `numpy`
  - `sqlalchemy`, `pymysql`
  - `scikit-learn`, `joblib`
  - `kafka-python`
  - `codecarbon`
  - `python-dotenv`
  - (optionnel) `tabulate` pour des tableaux markdown plus propres

---

## 5) Configuration (.env)

Créer un fichier `.env` à la racine du projet.

Exemple :

```bash
# --- MySQL ---
MYSQL_USER=exam
MYSQL_PASSWORD=exampwd
MYSQL_DATABASE=examdb
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306

# --- Kafka ---
KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:29092
KAFKA_TOPIC=events

# --- CodeCarbon ---
CC_OUTPUT_DIR=reports/emissions
CC_COUNTRY_ISO_CODE=FRA
```

> Remarque : `CC_OUTPUT_DIR` indique où CodeCarbon écrit `emissions.csv` et ses backups.

---

## 6) Installation (Python)

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install pandas numpy sqlalchemy pymysql scikit-learn joblib kafka-python python-dotenv codecarbon tabulate
```

---

## 7) Lancer les services (MySQL + Kafka)

Le projet s’appuie sur Docker (via un `docker-compose.yml` dans le dossier projet si présent).

Exemple d’usage :

```bash
docker compose up -d
docker ps
```

> Vérifier que MySQL et Kafka sont bien démarrés avant de lancer les scripts.

---

## 8) Exécution du pipeline (ordre recommandé)

### Étape A — Générer des données (JSONL brut)
```bash
python -m src.generate_data
```

Sortie :
- `data/raw/orders_events.jsonl`

### Étape B — ETL : extraction + transformation + nettoyage
```bash
python -m src.extract_transform
```

Sorties :
- `data/processed/orders_events_cleaned.csv`
- `data/processed/orders_events_cleaned.parquet`
- `data/processed/orders_events_cleaned.json`

### Étape C — Ingestion SQL (batch)
```bash
python -m src.ingest
```

Crée (si besoin) et remplit les tables :
- `table_customers`
- `table_orders`
- `table_events`

### Étape D — Entraînement ML

**Classification**
```bash
python -m src.train_classification
```

Produit :
- `artifacts/classification_model.joblib` (bundle modèle + preprocessing)
- `artifacts/classification_model_meta.json` (métriques + métadonnées)

**Régression**
```bash
python -m src.train_regression
```

Produit :
- `artifacts/regression_model.joblib`
- `artifacts/regression_model_meta.json`

### Étape E — Pipeline Kafka (streaming)
```bash
python -m src.kafka_pipeline
```

- Le producer envoie les événements clean au topic
- Le consumer :
  - ingère Customer/Order/Event
  - calcule `return_proba`
  - insère dans `table_predictions`

---

## 9) Mesure d’impact écologique (CodeCarbon)

Les scripts sont instrumentés via `track_phase(...)` (défini dans `eco_impact.py`), ce qui permet de mesurer par phase :

- énergie consommée (kWh estimé)
- émissions CO2e (kg)

### Générer le rapport final
```bash
python -m src.eco_report
```

Sorties :
- `reports/eco_report.md` (rapport lisible)
- `reports/eco_report.csv` (table agrégée)

> Le script `eco_report.py` fusionne aussi les backups `emissions.csv*` générés par CodeCarbon.

---

## 10) Modèle ML et inférence dans Kafka (explication)

Le fichier joblib sauvegarde un **bundle** :

```python
{
  "model": <sklearn_model>,
  "preprocess": {...}
}
```

Dans `kafka_consumer.py` :
1. on reconstruit les features brutes à partir du message Kafka  
2. on applique **exactement le même preprocessing** (imputation, IQR, OneHot, scaler, alignement colonnes)  
3. on calcule `predict_proba` et on stocke `return_proba`

Cela garantit que l’inférence reçoit les mêmes colonnes (mêmes noms / même ordre) que pendant l’entraînement.

---

## 11) Qualité & robustesse (tests / sécurité)

Le projet intègre plusieurs bonnes pratiques :

- dédoublonnage par `event_id`
- gestion des valeurs manquantes (catégorielles et numériques)
- normalisation des champs texte (strip, lower/upper)
- UPSERT “fichier” (CSV/Parquet) puis UPSERT SQL via `session.merge`
- validations minimales dans Kafka (JSON valide + clés obligatoires)
- détection de doublons côté consumer (skip si déjà en base)

---

## 12) Limites et pistes d’amélioration

- **Performance ingestion SQL** : améliorer en batch (bulk insert/update) au lieu de `merge` ligne à ligne.
- **Streaming** : faire des micro-batches côté consumer pour réduire l’overhead.
- **Eco-impact** : faire un run “propre” (un seul tracker à la fois) pour éviter tout risque de double comptage.
- **Observabilité** : ajouter logs structurés (JSON) et métriques.

---

## 13) Commandes rapides (récap)

```bash
python -m src.generate_data
python -m src.extract_transform
python -m src.ingest
python -m src.train_classification
python -m src.train_regression
python -m src.kafka_pipeline
python -m src.eco_report
```
