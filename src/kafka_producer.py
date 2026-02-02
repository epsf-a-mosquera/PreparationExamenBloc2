#PreparationExamenBloc2/src/kafka_producer.py
"""
kafka_producer.py
-----------------
Producer Kafka pour l'examen Bloc 2.

Rôle :
- Lire un fichier JSON Lines (1 JSON par ligne) contenant des événements cleans généré par extract_transform.py
- Envoyer chaque événement dans un topic Kafka (KAFKA_TOPIC)

Pourquoi ce design :
- Découplage : le producer se contente d'émettre des événements, le consumer les traite.
- Robustesse : on ignore les lignes vides / JSON invalides au lieu de planter.
- Traçabilité : logs + compteur envoyés.
- Optionnel : key Kafka (event_id) pour aider le partitionnement et la cohérence.

Usage :
- Lancer Kafka via docker-compose
- Lancer le consumer dans un terminal
- Lancer ce producer dans un autre terminal

Exemples :
    python -m src.kafka_producer
    python -m src.kafka_producer   # selon ton setup
"""

################
# kafka_producer
################

import json
import time
from typing import Optional

from kafka import KafkaProducer

from src.config import CLEAN_JSONL_PATH, KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC

from src.eco_impact import track_phase  # NEW : mesure producer Kafka



def build_producer() -> KafkaProducer:
    """
    Construit et retourne un KafkaProducer.

    Paramètres importants :
    - bootstrap_servers : adresse du broker Kafka
    - value_serializer  : convertit la valeur envoyée en bytes (obligatoire pour Kafka)
    - key_serializer    : convertit la clé en bytes (utile pour partitionner par event_id)
    - acks='all'        : (optionnel) garantit que Kafka confirme l'écriture (plus sûr)
    """
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        # On envoie des strings JSON => on encode en UTF-8
        value_serializer=lambda v: v.encode("utf-8"),
        # La key (si fournie) est aussi une string => encode UTF-8
        key_serializer=lambda v: v.encode("utf-8") if v is not None else None,
        # acks='all' = le broker confirme que le message est bien répliqué (plus fiable)
        # Inconvénient : un peu plus lent (acceptable pour l'examen)
        acks="all",
        retries=3,  # en cas d'erreur transitoire, Kafka-python réessaie
        linger_ms=5,  # micro-batching léger (perf) sans compliquer
    )
    return producer


def extract_key_from_json_line(json_line: str) -> Optional[str]:
    """
    Extrait une clé Kafka depuis la ligne JSON.

    Ici on prend event_id (id unique) si présent.
    Avantage :
    - même event_id -> même partition (souvent), cohérence côté consumer
    """
    try:
        obj = json.loads(json_line)
        event_id = obj.get("event_id")
        if isinstance(event_id, str) and event_id.strip():
            return event_id
        return None
    except json.JSONDecodeError:
        # Si JSON invalide, pas de key
        return None


def main(
    delay_s: float = 0.01,
    max_messages: Optional[int] = None,
    flush_every: int = 200,
    wait_ack: bool = False,
) -> None:
    # NEW : mesure le temps d'émission des messages (CPU + I/O)
    with track_phase("kafka_producer_send"):
        """
        Lit le JSONL clean et l'envoie sur Kafka.

        Args:
            delay_s: délai entre 2 messages (simule un flux temps réel)
            max_messages: si défini, limite le nombre total de messages envoyés (pratique en exam)
            flush_every: flush périodique pour s'assurer que les messages partent
            wait_ack: si True, on attend l'accusé de réception Kafka pour chaque message (plus sûr, plus lent)
        """
        # 1) Vérification fichier d'entrée
        if not CLEAN_JSONL_PATH.exists():
            raise FileNotFoundError(
                f"Fichier introuvable: {CLEAN_JSONL_PATH}. "
                f"As-tu généré les données (extract_transform.py) ?"
            )

        # 2) Création du producer
        producer = build_producer()

        sent = 0
        skipped_empty = 0
        skipped_invalid_json = 0

        print(f"[START] Producer Kafka -> topic='{KAFKA_TOPIC}' broker='{KAFKA_BOOTSTRAP_SERVERS}'")
        print(f"[INFO] Source JSONL: {CLEAN_JSONL_PATH}")

        # 3) Lecture du fichier ligne par ligne
        with open(CLEAN_JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                # a) Nettoyage de la ligne
                line = line.strip()

                # b) Ignorer lignes vides
                if not line:
                    skipped_empty += 1
                    continue

                # c) Vérifier JSON valide (robustesse)
                #    (Sans ça, le consumer pourrait planter si tu n'as pas géré ce cas)
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    skipped_invalid_json += 1
                    continue

                # d) Extraire une key Kafka (event_id)
                key = extract_key_from_json_line(line)

                # e) Envoyer message
                future = producer.send(KAFKA_TOPIC, key=key, value=line)

                # f) Option : attendre l'ack broker (garantie que c'est écrit)
                if wait_ack:
                    # future.get() bloque jusqu'à confirmation ou erreur
                    future.get(timeout=10)

                sent += 1

                # g) Flush périodique
                if sent % flush_every == 0:
                    producer.flush()
                    print(f"[INFO] envoyés: {sent} (empty_skipped={skipped_empty}, invalid_json_skipped={skipped_invalid_json})")

                # h) Stop si on a atteint max_messages
                if max_messages is not None and sent >= max_messages:
                    break

                # i) Délai pour simuler un flux
                time.sleep(delay_s)

        # 4) Flush final + fermeture
        producer.flush()
        producer.close()

        print("[DONE] Producer terminé.")
        print(f"       Total envoyés            : {sent}")
        print(f"       Lignes vides ignorées    : {skipped_empty}")
        print(f"       JSON invalides ignorés   : {skipped_invalid_json}")


if __name__ == "__main__":
    # Valeurs par défaut adaptées à l'examen
    main(delay_s=0.01, max_messages=None, flush_every=200, wait_ack=False)
