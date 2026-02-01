#ExamenBloc2/src/kafka_producer.py
"""
Cette scriopt met en place un pipeline Kafka simple. Le but : montrer un flux continu.
Producer : lit le dataset propre en csv et envoie à un topic Kafka, ce documents sont produits par un script extract_transform.py dans le dossier PROCESSED_DIR et nommé orders_events_cleaned.csv (CLEAN_CSV_PATH)
Consumer : lit messages et insère en DB en utilisant le script d'insertion DB fourni (ingest_data.py).

Tips :
Chaque ordre a un event_id unique (UUID) dans le dataset.
En consumer, ignore si déjà inséré.
"""
################
# kafka_producer
################
import json
import time

from kafka import KafkaProducer

from src.config import RAW_JSONL_PATH, KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC


def main(delay_s: float = 0.01) -> None:
    """Lit le JSONL brut et l'envoie sur Kafka."""
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        # value_serializer transforme dict->bytes (ici on envoie déjà string JSON)
        value_serializer=lambda v: v.encode("utf-8"),
    )

    sent = 0

    with open(RAW_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # On envoie la ligne JSON telle quelle (string)
            producer.send(KAFKA_TOPIC, value=line)
            sent += 1

            # flush périodique pour s'assurer que ça part
            if sent % 200 == 0:
                producer.flush()
                print(f"[INFO] envoyés: {sent}")

            # petite pause pour simuler un flux temps réel
            time.sleep(delay_s)

    producer.flush()
    producer.close()

    print(f"[OK] Producer terminé. Total envoyés : {sent}")

