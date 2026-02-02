# ExamenBloc2/src/admin_kafka.py
"""
admin.py
--------
Petit outil CLI pour administrer les topics Kafka (create / delete / list).

Pourquoi c'est utile ?
- Créer le topic avant de lancer producer/consumer
- Supprimer le topic pour repartir de zéro ("reset" du flux)

Exemples :
    python -m src.admin create --name orders_events --partitions 1 --replication 1
    python -m src.admin delete --name orders_events
    python -m src.admin list
"""

import argparse

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError, UnknownTopicOrPartitionError

from src.config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC


def build_admin_client(bootstrap_servers: str) -> KafkaAdminClient:
    """
    Construit le client d'administration Kafka.
    - bootstrap_servers : ex "localhost:9092" ou "kafka:9092" (docker)
    """
    return KafkaAdminClient(bootstrap_servers=bootstrap_servers, client_id="examenbloc2-admin")


def list_topics(client: KafkaAdminClient) -> None:
    """Liste les topics disponibles."""
    topics = client.list_topics()
    print("Topics Kafka :")
    for t in sorted(topics):
        print(f" - {t}")


def create_topic(client: KafkaAdminClient, topic: str, num_partitions: int, replication_factor: int) -> None:
    """
    Crée un topic Kafka.
    - num_partitions : parallélisme (consommation en groupe)
    - replication_factor : tolérance aux pannes (souvent 1 en environnement examen / 1 broker)
    """
    topic_list = [NewTopic(name=topic, num_partitions=num_partitions, replication_factor=replication_factor)]

    try:
        client.create_topics(new_topics=topic_list, validate_only=False)
        print(f"✅ Topic '{topic}' créé (partitions={num_partitions}, replication={replication_factor}).")
    except TopicAlreadyExistsError:
        print(f"ℹ️ Topic '{topic}' existe déjà -> aucune action.")
    except Exception as e:
        print(f"❌ Erreur création topic '{topic}': {e}")


def delete_topic(client: KafkaAdminClient, topic: str) -> None:
    """
    Supprime un topic Kafka.
    ⚠️ Selon la config Kafka, la suppression peut être async et prendre quelques secondes.
    """
    try:
        client.delete_topics([topic])
        print(f"✅ Topic '{topic}' supprimé.")
    except UnknownTopicOrPartitionError:
        print(f"ℹ️ Topic '{topic}' n'existe pas -> aucune action.")
    except Exception as e:
        print(f"❌ Erreur suppression topic '{topic}': {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Admin Kafka Topics (create/delete/list).")

    # On propose 3 actions : create, delete, list
    parser.add_argument("action", choices=["create", "delete", "list"], help="Action à exécuter")

    # Paramètres optionnels
    parser.add_argument(
        "--bootstrap",
        type=str,
        default=str(KAFKA_BOOTSTRAP_SERVERS),
        help="Bootstrap servers Kafka (par défaut depuis src.config)",
    )

    # Nom du topic : par défaut on prend KAFKA_TOPIC (depuis config)
    parser.add_argument(
        "--name",
        type=str,
        default=str(KAFKA_TOPIC),
        help="Nom du topic (par défaut KAFKA_TOPIC depuis src.config)",
    )

    # Uniquement pour create
    parser.add_argument("--partitions", type=int, default=1, help="Nombre de partitions (create)")
    parser.add_argument("--replication", type=int, default=1, help="Replication factor (create)")

    args = parser.parse_args()

    client = build_admin_client(args.bootstrap)

    try:
        if args.action == "list":
            list_topics(client)

        elif args.action == "create":
            create_topic(
                client=client,
                topic=args.name,
                num_partitions=args.partitions,
                replication_factor=args.replication,
            )

        elif args.action == "delete":
            delete_topic(client=client, topic=args.name)

    finally:
        # Toujours fermer proprement la connexion admin
        client.close()


if __name__ == "__main__":
    main()
