# ExamenBloc2/src/ingest.py
"""
Script pour ingérer les données nettoyées dans une base de données SQL.

Fonctionnalités :
1. Se connecte à la base de données via SQLAlchemy.
2. Crée les tables définies dans db_models.py si elles n'existent pas.
3. Lit les données nettoyées depuis le CSV.
4. Insère les données dans les tables Customer, Order et Event.
   - Utilise merge pour éviter les doublons sur les clés primaires.
Remarques :
- S'assurer que la base de données est accessible avant d'exécuter le script.
- Si des données existent déjà dans les tables, elles seront mises à jour selon la clé primaire.
"""

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
from src.config import SQLALCHEMY_DATABASE_URL, CLEAN_CSV_PATH
from src.db_models import Base, Event, Order, Customer

# --- Étape 1 : Créer une connexion à la base de données ---
engine = create_engine(SQLALCHEMY_DATABASE_URL)
print("Connexion à la base de données établie.")

# --- Étape 2 : Créer les tables si elles n'existent pas ---
Base.metadata.create_all(engine)
print("Tables créées (si elles n'existaient pas).")

# --- Étape 3 : Charger les données nettoyées depuis CSV ---
df = pd.read_csv(CLEAN_CSV_PATH)
print("Données nettoyées chargées :", df.shape)

# --- Étape 4 : Insertion des données dans les tables ---
try:
    with engine.begin() as connection:  # transaction automatique
        for _, row in df.iterrows():
            # --- Table Customer ---
            customer = Customer(
                customer_customer_id=row['customer_customer_id'],
                customer_country=row['customer_country']
            )
            connection.merge(customer)  # merge : insert ou update si existe déjà

            # --- Table Order ---
            order = Order(
                order_id=row['order_id'],
                customer_customer_id=row['customer_customer_id'],
                order_device=row['order_device'],
                order_channel=row['order_channel'],
                order_main_category=row['order_main_category'],
                order_n_items=row['order_n_items'],
                order_basket_value=row['order_basket_value'],
                order_shipping_fee=row['order_shipping_fee'],
                order_discount=row['order_discount'],
                order_order_total=row['order_order_total'],
                order_is_returned=row['order_is_returned'],
                price_average_per_item=row['price_average_per_item']
            )
            connection.merge(order)  # merge : insert ou update

            # --- Table Event ---
            event = Event(
                event_id=row['event_id'],
                event_time=row['event_time'],
                event_year=row['event_year'],
                event_month=row['event_month'],
                event_day=row['event_day'],
                event_hour=row['event_hour'],
                order_id=row['order_id'],
                customer_customer_id=row['customer_customer_id']
            )
            connection.merge(event)  # merge : insert ou update

    print("Ingestion terminée avec succès.")
except SQLAlchemyError as e:
    print("Erreur lors de l'ingestion :", e)
