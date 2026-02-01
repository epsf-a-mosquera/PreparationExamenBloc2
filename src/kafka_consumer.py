################
# kafka_consumer
################



import json

import pandas as pd
from joblib import load
from kafka import KafkaConsumer
from sqlalchemy.exc import IntegrityError

from src.config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, MODEL_PATH
from src.db import get_engine, get_session_factory
from src.db_models import Base, Customer, Order, IngestedEvent, Prediction
from src.validation import validate_event_schema, flatten_event


def ensure_tables() -> None:
    """Crée les tables si besoin."""
    engine = get_engine(echo=False)
    Base.metadata.create_all(engine)


def main() -> None:
    """Boucle de consommation Kafka."""
    ensure_tables()

    # Charger modèle ML (pipeline complet)
    model = load(MODEL_PATH)

    # Kafka consumer
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda b: b.decode("utf-8"),
    )

    engine = get_engine(echo=False)
    SessionLocal = get_session_factory(engine)
    session = SessionLocal()

    print("[OK] Consumer démarré. En attente de messages...")

    try:
        for msg in consumer:
            raw = msg.value

            # 1) Parse JSON
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                print("[WARN] Message non-JSON -> ignoré")
                continue

            # 2) Validation schéma
            ok, err = validate_event_schema(event)
            if not ok:
                print(f"[WARN] Message invalide ({err}) -> ignoré")
                continue

            # 3) Dédoublonnage par event_id via table ingested_events
            event_id = event["event_id"]

            # Si event_id déjà vu => skip
            already = session.query(IngestedEvent).filter_by(event_id=event_id).one_or_none()
            if already is not None:
                print(f"[INFO] Doublon event_id={event_id} -> skip")
                continue

            # On enregistre l'event_id comme "traité"
            session.add(IngestedEvent(event_id=event_id))
            session.commit()

            # 4) Aplatir l'événement
            flat = flatten_event(event)

            # 5) Upsert customer
            customer = session.query(Customer).filter_by(customer_id=flat["customer_id"]).one_or_none()
            if customer is None:
                customer = Customer(customer_id=flat["customer_id"], country=flat["country"])
                session.add(customer)
                session.flush()

            # 6) Insert order (unique order_id)
            order = Order(
                order_id=flat["order_id"],
                customer_db_id=customer.id,
                device=flat["device"],
                channel=flat["channel"] if flat["channel"] is not None else "unknown",
                main_category=flat["main_category"],
                n_items=int(flat["n_items"]),
                basket_value=float(flat["basket_value"]),
                shipping_fee=float(flat["shipping_fee"]),
                discount=float(flat["discount"]) if flat["discount"] is not None else 0.0,
                order_total=float(flat["order_total"]),
                avg_item_price=float(flat["order_total"]) / max(int(flat["n_items"]), 1),
                is_returned=int(flat["is_returned"]),
                event_time=flat["event_time"],
            )

            session.add(order)

            try:
                session.commit()
            except IntegrityError:
                # Doublon order_id => on ignore
                session.rollback()
                print(f"[INFO] Doublon order_id={flat['order_id']} -> skip order insert")

            # 7) Inference ML (proba retour)
            # On construit un DataFrame 1 ligne avec les features attendues
            X = pd.DataFrame([{
                "country": flat["country"],
                "device": flat["device"],
                "channel": flat["channel"] if flat["channel"] is not None else "unknown",
                "main_category": flat["main_category"],
                "n_items": flat["n_items"],
                "basket_value": flat["basket_value"],
                "shipping_fee": flat["shipping_fee"],
                "discount": flat["discount"] if flat["discount"] is not None else 0.0,
                "order_total": flat["order_total"],
                "avg_item_price": float(flat["order_total"]) / max(int(flat["n_items"]), 1),
            }])

            # predict_proba retourne [proba_classe0, proba_classe1]
            proba_return = float(model.predict_proba(X)[0][1])

            # 8) Sauvegarder la prédiction
            pred = Prediction(order_id=flat["order_id"], event_id=event_id, return_proba=proba_return)
            session.add(pred)

            try:
                session.commit()
            except IntegrityError:
                # Si déjà une prédiction pour cet event_id, on ignore
                session.rollback()

            print(f"[OK] event_id={event_id} ingéré | proba_return={proba_return:.3f}")

    finally:
        session.close()
        consumer.close()



if __name__ == "__main__":
    main()