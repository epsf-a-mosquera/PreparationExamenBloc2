#PreparationExamenBloc2/src/ingest_predictions.py

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.config import SQLALCHEMY_DATABASE_URL
from src.db_models import Base, Prediction, Order, Customer, Event
from src.eco_impact import track_phase  # NEW : mesure ingestion SQL (mode CLI)


def ingest_predictions_dataframe(df: pd.DataFrame, engine=None, validate_fk: bool = True) -> dict:
    # NEW : mesure l'ingestion batch CSV -> MySQL
    with track_phase("sql_ingestion_predictions_batch"):
        """
        Ingestion UPSERT (merge) des prédictions depuis un DataFrame.

        Colonnes attendues :
        - event_id
        - order_id
        - customer_customer_id
        - return_proba
        """
        if engine is None:
            engine = create_engine(SQLALCHEMY_DATABASE_URL)

        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

        required = {"event_id", "order_id", "customer_customer_id", "return_proba"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"Colonnes manquantes pour ingest_predictions_dataframe: {sorted(list(missing))}")

        df = df.copy()
        df["return_proba"] = pd.to_numeric(df["return_proba"], errors="coerce")

        # Validation FK (optionnel)
        if validate_fk:
            with SessionLocal() as session:
                existing_orders = set(session.execute(select(Order.order_id)).scalars().all())
                existing_customers = set(session.execute(select(Customer.customer_customer_id)).scalars().all())
                existing_events = set(session.execute(select(Event.event_id)).scalars().all())

            df = df[
                df["order_id"].isin(existing_orders)
                & df["customer_customer_id"].isin(existing_customers)
                & df["event_id"].isin(existing_events)
            ].copy()

        predictions_done = set()
        predictions_count = 0

        with SessionLocal() as session:
            with session.begin():
                for row in df.itertuples(index=False):
                    event_id = getattr(row, "event_id")
                    if event_id in predictions_done:
                        continue

                    predictions_done.add(event_id)
                    predictions_count += 1

                    pred = Prediction(
                        event_id=event_id,
                        order_id=getattr(row, "order_id"),
                        customer_customer_id=getattr(row, "customer_customer_id"),
                        return_proba=float(getattr(row, "return_proba")),
                    )
                    session.merge(pred)

        return {"predictions_upserted": predictions_count}
