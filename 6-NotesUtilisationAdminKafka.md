# Guide d’utilisation complet — Kafka Producer / Consumer / Admin + Ingestion MySQL + Prédictions

Ce guide explique **pas à pas** comment utiliser :
- `src/admin_kafka.py` : **administration des topics Kafka** (create/delete/list)
- `src/kafka_producer.py` : **envoi** des événements clean (JSONL) dans Kafka
- `src/kafka_consumer.py` : **consommation**, ingestion en base via `ingest.py` + prédictions via `ingest_predictions.py`
- MySQL via SQLAlchemy ORM : tables `table_customers`, `table_orders`, `table_events`, `table_predictions`

---

## 1) Pré-requis

### 1.1 Outils nécessaires
- Python (idéalement 3.10+)
- Kafka (via Docker Compose dans ton projet)
- MySQL (ou MariaDB) accessible via `SQLALCHEMY_DATABASE_URL`
- Les dépendances Python installées (kafka-python, sqlalchemy, pandas, joblib, etc.)

### 1.2 Fichiers attendus dans le projet
- `src/config.py` doit contenir au minimum :
  - `KAFKA_BOOTSTRAP_SERVERS`
  - `KAFKA_TOPIC`
  - `CLEAN_JSONL_PATH`
  - `SQLALCHEMY_DATABASE_URL`
  - `CLASSIFICATION_MODEL_PATH`
- Le fichier `CLEAN_JSONL_PATH` doit exister : **1 JSON par ligne**
- Le modèle ML (pipeline) doit exister : `CLASSIFICATION_MODEL_PATH`

### 1.3 Rappel sur les scripts d’ingestion
- `src/ingest.py` : ingère (UPSERT) `Customer`, `Order`, `Event`
- `src/ingest_predictions.py` : ingère (UPSERT) `Prediction`

✅ Recommandation : `ingest.py` et `ingest_predictions.py` doivent exposer des fonctions réutilisables :
- `ingest_clean_dataframe(df, engine=...)`
- `ingest_predictions_dataframe(df, engine=..., validate_fk=True)`

---

## 2) Comprendre le flux de bout en bout

### 2.1 Le Producer
`kafka_producer.py` :
- lit un fichier **JSON Lines**
- envoie chaque ligne JSON dans Kafka (topic `KAFKA_TOPIC`)
- optionnel : met `event_id` comme **key Kafka** (meilleure cohérence par partition)

### 2.2 Le Consumer
`kafka_consumer.py` :
- lit chaque message du topic Kafka
- parse JSON + vérifie clés obligatoires
- déduplique (par `event_id`)
- construit une ligne “clean”
- appelle :
  - `ingest_clean_dataframe()` → Customer/Order/Event
  - modèle ML → `return_proba`
  - `ingest_predictions_dataframe()` → Prediction

### 2.3 Admin
`admin.py` :
- crée, supprime ou liste les topics Kafka
- utile pour “reset” le flux (repartir de zéro)

---

## 3) Préparer l’environnement

### 3.1 Lancer Kafka (Docker)
Depuis la racine du projet (selon ton docker-compose) :

```bash
docker compose up -d
```

Puis vérifier que Kafka tourne :

```bash
docker ps
```

> Tu dois voir un container Kafka (et souvent Zookeeper / Kraft selon la stack).

### 3.2 Vérifier la base MySQL
Ton `SQLALCHEMY_DATABASE_URL` doit être correct, ex :
- `mysql+pymysql://user:password@localhost:3306/dbname`
- ou via docker network : `mysql+pymysql://user:password@mysql:3306/dbname`

---

## 4) Administrer le topic Kafka (admin_kafka.py)

### 4.1 Lister les topics
```bash
python -m src.admin_kafka list
```

### 4.2 Créer le topic
```bash
python -m src.admin_kafka create --name "<ton_topic>" --partitions 1 --replication 1
```

💡 Si tu veux utiliser directement la valeur `KAFKA_TOPIC` du `config.py`, tu peux omettre `--name` si `admin_kafka.py` est codé avec `default=KAFKA_TOPIC`.

### 4.3 Supprimer le topic (reset)
```bash
python -m src.admin_kafka delete --name "<ton_topic>"
```

Ensuite tu peux recréer :
```bash
python -m src.admin_kafka create --name "<ton_topic>" --partitions 1 --replication 1
```

---

## 5) Lancer le Consumer

Le consumer doit être lancé **avant** le producer pour voir le flux en direct :

```bash
python -m src.kafka_consumer
```

Tu dois voir un log du type :
- `[OK] Consumer démarré. En attente de messages...`

---

## 6) Lancer le Producer

Dans un second terminal :

```bash
python -m src.kafka_producer
```

Résultat attendu :
- logs indiquant le nombre de messages envoyés
- côté consumer : logs `[OK] event_id=... ingéré | proba_return=...`

---

## 7) Format attendu des événements (JSONL)

Chaque ligne de `CLEAN_JSONL_PATH` doit être un JSON **avec au minimum** :

```json
{
  "event_id": "uuid",
  "order_id": "uuid",
  "customer_customer_id": "uuid"
}
```

Recommandé (car utile DB + modèle ML) :

```json
{
  "event_id": "uuid",
  "event_time": "2026-02-02T10:00:00Z",
  "order_id": "uuid",
  "customer_customer_id": "uuid",
  "customer_country": "FR",
  "order_device": "mobile",
  "order_channel": "web",
  "order_main_category": "fashion",
  "order_n_items": 3,
  "order_basket_value": 120.0,
  "order_shipping_fee": 5.0,
  "order_discount": 10.0,
  "order_order_total": 115.0,
  "order_is_returned": false
}
```

💡 Si certains champs sont absents, le consumer peut appliquer des **valeurs par défaut** (ex: `"unknown"`, `0.0`, `False`) pour respecter `nullable=False`.

---

## 8) Vérifier les données en base MySQL

### 8.1 Tables attendues
- `table_customers`
- `table_orders`
- `table_events`
- `table_predictions`

### 8.2 Vérifications SQL rapides
Exemples :

```sql
SELECT COUNT(*) FROM table_customers;
SELECT COUNT(*) FROM table_orders;
SELECT COUNT(*) FROM table_events;
SELECT COUNT(*) FROM table_predictions;
```

Pour contrôler les dernières prédictions :

```sql
SELECT event_id, order_id, customer_customer_id, return_proba
FROM table_predictions
ORDER BY event_id DESC
LIMIT 10;
```

---

## 9) Dépannage (problèmes fréquents)

### 9.1 Le consumer ignore tous les messages (doublons)
- Si tu relances le producer avec le même JSONL, `event_id` étant PK dans `table_events`, le consumer peut détecter que ça existe déjà et “skip”.
✅ Solution : reset topic + vider la base ou utiliser de nouveaux event_id.
commande avec admin_kafka.py pour reset le topic.
```bash
python -m src.admin_kafka delete --name "$KAFKA_TOPIC"
python -m src.admin_kafka create --name "$KAFKA_TOPIC" --partitions 1 --replication 1
```

### 9.2 Erreur FK lors de l’ingestion Prediction
Cause : `Prediction.order_id` et `Prediction.customer_customer_id` référencent des tables.
Si l’order/customer/event n’a pas été ingéré avant la prédiction → insertion échoue.

✅ Solution :
- garder l’ordre : ingest (Customer/Order/Event) **avant** Prediction (c’est ce que fait le consumer)
- activer `validate_fk=True` dans `ingest_predictions_dataframe`

### 9.3 Le producer tourne mais rien n’arrive côté consumer
- mauvais `KAFKA_BOOTSTRAP_SERVERS`
- mauvais topic
- Kafka non accessible depuis ton environnement (docker network)

✅ Vérifie :
- `python -m src.admin_kafka list`
- logs docker (`docker logs <container_kafka>`)

### 9.4 Le modèle ML plante (features manquantes)
Ton pipeline ML peut attendre des colonnes spécifiques.
✅ Solution :
- aligner `build_model_features()` du consumer avec les features utilisées à l’entraînement

---

## 10) Commandes “recette” (copier-coller)

### 10.1 Reset complet du topic
```bash
python -m src.admin_kafka delete --name "$KAFKA_TOPIC"
python -m src.admin_kafka create --name "$KAFKA_TOPIC" --partitions 1 --replication 1
```

### 10.2 Lancer consumer
```bash
python -m src.kafka_consumer
```

### 10.3 Lancer producer (autre terminal)
```bash
python -m src.kafka_producer
```

---

## 11) Bonnes pratiques (examen)
- Lancer le consumer **avant** le producer
- Conserver les logs (preuve de fonctionnement)
- Démontrer :
  - ingestion tables
  - déduplication event_id
  - prédictions insérées en DB

---

## 12) Résumé rapide
- `admin.py` : créer/supprimer/lister les topics Kafka
- `kafka_producer.py` : envoie les events clean JSONL dans Kafka
- `kafka_consumer.py` : consomme, ingère en base via `ingest.py`, calcule proba via modèle, ingère via `ingest_predictions.py`
- Les tables MySQL se remplissent automatiquement via SQLAlchemy ORM
