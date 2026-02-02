# ExamenBloc2/src/ingest.py
"""
Script d'ingestion (load) des données nettoyées dans une base SQL via SQLAlchemy ORM.

✅ IMPORTANT (pour Kafka) :
- Ce fichier expose une fonction réutilisable : ingest_clean_dataframe(df, engine=...)
- L'ingestion à partir d'un CSV est dans main() (donc pas exécutée à l'import)
"""

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from src.config import SQLALCHEMY_DATABASE_URL, CLEAN_CSV_PATH
from src.db_models import Base, Event, Order, Customer


def none_if_nan(value):
    """Convertit NaN/NaT en None (SQL)."""
    if pd.isna(value):
        return None
    return value


def ingest_clean_dataframe(df: pd.DataFrame, engine=None) -> dict:
    """
    Ingestion UPSERT (merge) depuis un DataFrame déjà "clean".

    Args:
        df: DataFrame avec les colonnes clean (voir projet)
        engine: engine SQLAlchemy optionnel (réutilisable côté Kafka)

    Returns:
        dict avec compteurs utiles (info/debug)
    """
    if engine is None:
        engine = create_engine(SQLALCHEMY_DATABASE_URL)

    # Tables
    Base.metadata.create_all(engine)

    # Convert event_time si nécessaire
    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)

    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    customers_done = set()
    orders_done = set()
    events_done = set()

    customers_count = 0
    orders_count = 0
    events_count = 0

    try:
        with SessionLocal() as session:
            with session.begin():
                for row in df.itertuples(index=False):

                    cust_id = getattr(row, "customer_customer_id")
                    order_id = getattr(row, "order_id")
                    event_id = getattr(row, "event_id")

                    # A) Customer
                    if cust_id not in customers_done:
                        customers_done.add(cust_id)
                        customers_count += 1
                        customer = Customer(
                            customer_customer_id=none_if_nan(cust_id),
                            customer_country=none_if_nan(getattr(row, "customer_country")),
                        )
                        session.merge(customer)

                    # B) Order
                    if order_id not in orders_done:
                        orders_done.add(order_id)
                        orders_count += 1
                        order = Order(
                            order_id=none_if_nan(order_id),
                            customer_customer_id=none_if_nan(cust_id),
                            order_device=none_if_nan(getattr(row, "order_device")),
                            order_channel=none_if_nan(getattr(row, "order_channel")),
                            order_main_category=none_if_nan(getattr(row, "order_main_category")),
                            order_n_items=none_if_nan(getattr(row, "order_n_items")),
                            order_basket_value=none_if_nan(getattr(row, "order_basket_value")),
                            order_shipping_fee=none_if_nan(getattr(row, "order_shipping_fee")),
                            order_discount=none_if_nan(getattr(row, "order_discount")),
                            order_order_total=none_if_nan(getattr(row, "order_order_total")),
                            order_is_returned=none_if_nan(getattr(row, "order_is_returned")),
                            price_average_per_item=none_if_nan(getattr(row, "price_average_per_item")),
                        )
                        session.merge(order)

                    # C) Event
                    if event_id not in events_done:
                        events_done.add(event_id)
                        events_count += 1
                        event = Event(
                            event_id=none_if_nan(event_id),
                            event_time=none_if_nan(getattr(row, "event_time")),
                            event_year=none_if_nan(getattr(row, "event_year")),
                            event_month=none_if_nan(getattr(row, "event_month")),
                            event_day=none_if_nan(getattr(row, "event_day")),
                            event_hour=none_if_nan(getattr(row, "event_hour")),
                            order_id=none_if_nan(order_id),
                            customer_customer_id=none_if_nan(cust_id),
                        )
                        session.merge(event)

        return {
            "customers_upserted": customers_count,
            "orders_upserted": orders_count,
            "events_upserted": events_count,
        }

    except SQLAlchemyError as e:
        raise RuntimeError(f"SQLAlchemyError ingestion clean dataframe: {e}") from e


def main():
    """Mode CLI : ingestion depuis le CSV clean."""
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    Base.metadata.create_all(engine)

    df = pd.read_csv(CLEAN_CSV_PATH)
    print(f"✅ CSV clean chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes.")

    stats = ingest_clean_dataframe(df, engine=engine)
    print("✅ Ingestion terminée :", stats)


if __name__ == "__main__":
    main()
