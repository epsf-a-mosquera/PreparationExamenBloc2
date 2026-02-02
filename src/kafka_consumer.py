# ExamenBloc2/src/kafka_consumer.py
"""
kafka_consumer.py
-----------------
Consumer Kafka pour l'examen Bloc 2.

Rôle :
- Consommer des événements depuis un topic Kafka (KAFKA_TOPIC)
- Transformer/normaliser l'événement JSON en une ligne "clean" compatible avec ingest.py
- Ingestion en base MySQL via SQLAlchemy en réutilisant :
    - src.ingest.ingest_clean_dataframe()   -> Customer / Order / Event
    - src.ingest_predictions.ingest_predictions_dataframe() -> Prediction
- Faire l'inférence ML (classification) pour calculer return_proba

Pourquoi ce design :
- On centralise toute la logique d'UPSERT dans ingest.py / ingest_predictions.py
  => la consommation Kafka reste simple et robuste.
- On évite les doublons via event_id (PK de Event) en vérifiant l'existence avant de ré-ingérer.
- Logs explicites pour debug.

Usage :
    python -m src.kafka_consumer
"""

import json
from datetime import datetime, timezone

import pandas as pd
from joblib import load
from kafka import KafkaConsumer
from sqlalchemy import create_engine, select

from src.config import (
    CLASSIFICATION_MODEL_PATH,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    SQLALCHEMY_DATABASE_URL,
)
from src.db_models import Base, Event
from src.ingest import ingest_clean_dataframe
from src.ingest_predictions import ingest_predictions_dataframe


# ============================================================
# 0) Initialisation DB
# ============================================================

def ensure_tables(engine) -> None:
    """
    Crée les tables si elles n'existent pas déjà.
    """
    Base.metadata.create_all(engine)


def event_already_ingested(engine, event_id: str) -> bool:
    """
    Vérifie si un event_id existe déjà dans la table Event.
    Comme event_id est PK, si l'event est présent, on considère qu'il a déjà été ingéré.
    """
    # On ouvre une connexion courte (transaction implicite read-only)
    with engine.connect() as conn:
        res = conn.execute(select(Event.event_id).where(Event.event_id == event_id)).first()
        return res is not None


# ============================================================
# 1) Normalisation message Kafka -> "row clean"
# ============================================================

def to_utc_datetime(value, fallback_ms: int | None = None) -> datetime:
    """
    Convertit une valeur en datetime UTC.
    - Si value est déjà ISO string -> parse
    - Si value est None/invalide -> fallback sur timestamp Kafka (ms) si fourni
    - Sinon -> maintenant (UTC)
    """
    if value is not None:
        try:
            # pandas gère bien beaucoup de formats
            dt = pd.to_datetime(value, utc=True, errors="coerce")
            if not pd.isna(dt):
                return dt.to_pydatetime()
        except Exception:
            pass

    if fallback_ms is not None:
        return datetime.fromtimestamp(fallback_ms / 1000.0, tz=timezone.utc)

    return datetime.now(tz=timezone.utc)


def build_clean_row(event_json: dict, kafka_timestamp_ms: int | None) -> dict:
    """
    Construit une ligne "clean" (dict) conforme aux colonnes attendues par ingest.py.

    On met des valeurs par défaut quand certaines clés sont absentes, car en DB
    plusieurs colonnes sont nullable=False.
    """
    # Identifiants indispensables
    event_id = str(event_json["event_id"])
    order_id = str(event_json["order_id"])
    customer_id = str(event_json["customer_customer_id"])

    # event_time : on essaye event_json["event_time"], sinon timestamp kafka
    event_time = to_utc_datetime(event_json.get("event_time"), fallback_ms=kafka_timestamp_ms)

    # Valeurs business (avec defaults raisonnables)
    customer_country = event_json.get("customer_country", "unknown")

    order_device = event_json.get("order_device", "unknown")
    order_channel = event_json.get("order_channel", "unknown")
    order_main_category = event_json.get("order_main_category", "unknown")

    order_n_items = int(event_json.get("order_n_items", 0))
    order_basket_value = float(event_json.get("order_basket_value", 0.0))
    order_shipping_fee = float(event_json.get("order_shipping_fee", 0.0))
    order_discount = float(event_json.get("order_discount", 0.0))
    order_order_total = float(event_json.get("order_order_total", 0.0))

    # Label : si absent dans le flux, on met False par défaut
    # (sinon l'INSERT échouera car nullable=False dans ta table Order)
    order_is_returned = bool(event_json.get("order_is_returned", False))

    # Features dérivées
    price_average_per_item = order_order_total / max(order_n_items, 1)

    # Découpage date/heure
    event_year = int(event_time.year)
    event_month = int(event_time.month)
    event_day = int(event_time.day)
    event_hour = int(event_time.hour)

    # Ligne "clean" conforme à ingest.py
    return {
        "event_id": event_id,
        "event_time": event_time,
        "order_id": order_id,
        "customer_customer_id": customer_id,
        "customer_country": customer_country,
        "order_device": order_device,
        "order_channel": order_channel,
        "order_main_category": order_main_category,
        "order_n_items": order_n_items,
        "order_basket_value": order_basket_value,
        "order_shipping_fee": order_shipping_fee,
        "order_discount": order_discount,
        "order_order_total": order_order_total,
        "order_is_returned": order_is_returned,
        "event_year": event_year,
        "event_month": event_month,
        "event_day": event_day,
        "event_hour": event_hour,
        "price_average_per_item": price_average_per_item,
    }


def build_model_features(clean_row: dict) -> pd.DataFrame:
    """
    Construit le DataFrame features pour le modèle ML (1 ligne).
    On réutilise les colonnes typiques de ton dataset clean.
    """
    return pd.DataFrame([{
        "customer_country": clean_row["customer_country"],
        "order_device": clean_row["order_device"],
        "order_channel": clean_row["order_channel"],
        "order_main_category": clean_row["order_main_category"],
        "order_n_items": clean_row["order_n_items"],
        "order_basket_value": clean_row["order_basket_value"],
        "order_shipping_fee": clean_row["order_shipping_fee"],
        "order_discount": clean_row["order_discount"],
        "order_order_total": clean_row["order_order_total"],
        "price_average_per_item": clean_row["price_average_per_item"],
        "event_year": clean_row["event_year"],
        "event_month": clean_row["event_month"],
        "event_day": clean_row["event_day"],
        "event_hour": clean_row["event_hour"],
    }])


# ============================================================
# 2) MAIN - Boucle Kafka
# ============================================================

def main() -> None:
    # Engine DB créé une fois, puis réutilisé
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

    # Crée les tables au démarrage
    ensure_tables(engine)

    # Charger le modèle ML (pipeline complet)
    model = load(CLASSIFICATION_MODEL_PATH)

    # Kafka consumer
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda b: b.decode("utf-8"),
        group_id="examenbloc2-consumer",
    )

    print("[OK] Consumer démarré. En attente de messages...")

    try:
        for msg in consumer:
            raw = msg.value

            # 1) Parse JSON
            try:
                event_json = json.loads(raw)
            except json.JSONDecodeError:
                print("[WARN] Message non-JSON -> ignoré")
                continue

            # 2) Validation minimale des clés obligatoires
            required = {"event_id", "order_id", "customer_customer_id"}
            missing = required - set(event_json.keys())
            if missing:
                print(f"[WARN] Champs manquants {sorted(list(missing))} -> ignoré")
                continue

            event_id = str(event_json["event_id"])

            # 3) Déduplication : si déjà en base, on skip
            if event_already_ingested(engine, event_id):
                print(f"[INFO] Doublon event_id={event_id} -> skip")
                continue

            # 4) Normaliser vers une ligne clean
            clean_row = build_clean_row(event_json, kafka_timestamp_ms=msg.timestamp)
            df_clean = pd.DataFrame([clean_row])

            # 5) Ingestion Customer / Order / Event via ingest.py
            try:
                ingest_clean_dataframe(df_clean, engine=engine)
            except Exception as e:
                print(f"[ERROR] Ingestion (Customer/Order/Event) KO event_id={event_id} -> {e}")
                continue

            # 6) Inférence ML
            try:
                X = build_model_features(clean_row)
                proba_return = float(model.predict_proba(X)[0][1])
            except Exception as e:
                print(f"[ERROR] Inference ML KO event_id={event_id} -> {e}")
                continue

            # 7) Ingestion Prediction via ingest_predictions.py
            df_pred = pd.DataFrame([{
                "event_id": clean_row["event_id"],
                "order_id": clean_row["order_id"],
                "customer_customer_id": clean_row["customer_customer_id"],
                "return_proba": proba_return,
            }])

            try:
                ingest_predictions_dataframe(df_pred, engine=engine, validate_fk=True)
            except Exception as e:
                print(f"[ERROR] Ingestion Prediction KO event_id={event_id} -> {e}")
                continue

            print(f"[OK] event_id={event_id} ingéré | proba_return={proba_return:.3f}")

    finally:
        consumer.close()
        print("[OK] Consumer arrêté proprement.")


if __name__ == "__main__":
    main()
