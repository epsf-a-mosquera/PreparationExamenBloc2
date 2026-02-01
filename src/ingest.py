# ExamenBloc2/src/ingest.py
"""
Script d'ingestion (load) des données nettoyées (CSV) dans une base SQL via SQLAlchemy ORM.

Objectif :
- Lire un CSV "clean"
- Insérer / mettre à jour (upsert) les enregistrements dans 3 tables :
  - Customer
  - Order
  - Event

⚠️ Point important :
- La méthode `merge()` est une méthode ORM => elle fonctionne uniquement avec une `Session`.
- Si on utilise `engine.begin()` on obtient une `Connection` (SQLAlchemy Core) => pas de `merge()`.

Choix technique :
- On utilise `SessionLocal()` + `session.begin()` pour avoir :
  - Transaction automatique (commit si OK, rollback si erreur)
  - `session.merge()` pour éviter les doublons sur les clés primaires (PK)
  
Struture du CSV clean :
event_id                       string[python]
event_time                datetime64[ns, UTC]
order_id                       string[python]
customer_customer_id           string[python]
customer_country                     category
order_device                         category
order_channel                        category
order_main_category                  category
order_n_items                           int64
order_basket_value                    float64
order_shipping_fee                    float64
order_discount                        float64
order_order_total                     float64
order_is_returned                        bool
event_year                              int32
event_month                             int32
event_day                               int32
event_hour                              int32
price_average_per_item                float64

"""

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from src.config import SQLALCHEMY_DATABASE_URL, CLEAN_CSV_PATH
from src.db_models import Base, Event, Order, Customer


# ============================================================
# 0) Fonctions utilitaires
# ============================================================

def none_if_nan(value):
    """
    Convertit NaN/NaT (Pandas) en None (SQL).
    Utile car MySQL/SQLAlchemy peuvent mal gérer certains NaN.
    """
    if pd.isna(value):
        return None
    return value

# ============================================================
# 1) CONNEXION À LA BASE
# ============================================================

# create_engine() crée "le moteur" SQLAlchemy.
# - Il ne se connecte pas forcément immédiatement, mais prépare la connexion.
engine = create_engine(SQLALCHEMY_DATABASE_URL)
print("✅ Connexion à la base de données établie (engine créé).")

# ============================================================
# 2) CRÉATION DES TABLES (SI BESOIN)
# ============================================================

# Base.metadata.create_all(engine) va :
# - regarder les modèles ORM déclarés dans db_models.py
# - créer les tables correspondantes si elles n'existent pas déjà
Base.metadata.create_all(engine)
print("✅ Tables créées (si elles n'existaient pas).")

# ============================================================
# 3) CHARGEMENT DU CSV NETTOYÉ
# ============================================================

# Lecture du CSV clean.
# df contiendra toutes les colonnes dont tu as besoin pour remplir tes tables.
df = pd.read_csv(CLEAN_CSV_PATH)
print(f"✅ Données nettoyées chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes.")

# Conversion type (recommandée) :
# - si event_time est une string, on la convertit en datetime
# - errors="coerce" => si une valeur est invalide => NaT (équivalent NaN pour dates)
if "event_time" in df.columns:
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")

# (Optionnel) Diagnostic rapide : combien de doublons par identifiant ?
if "customer_customer_id" in df.columns:
    dup_customers = int(df["customer_customer_id"].duplicated().sum())
    print(f"ℹ️ Doublons customer_customer_id dans le CSV : {dup_customers}")

if "order_id" in df.columns:
    dup_orders = int(df["order_id"].duplicated().sum())
    print(f"ℹ️ Doublons order_id dans le CSV : {dup_orders}")

if "event_id" in df.columns:
    dup_events = int(df["event_id"].duplicated().sum())
    print(f"ℹ️ Doublons event_id dans le CSV : {dup_events}")

# ============================================================
# 4) CRÉATION DE LA SESSION ORM
# ============================================================

# sessionmaker fabrique une "factory" de Session.
# - autocommit=False => tu contrôles les commits (via session.begin() ici)
# - autoflush=False => évite de pousser en DB automatiquement à chaque opération
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# ============================================================
# 5) INGESTION / UPSERT (MERGE) DANS LES TABLES
# ============================================================

try:
    # `with SessionLocal() as session:` assure :
    # - fermeture propre de la session à la fin du bloc
    with SessionLocal() as session:

        # `with session.begin():` démarre une transaction.
        # - Si tout se passe bien => COMMIT automatique
        # - Si une erreur survient => ROLLBACK automatique
        with session.begin():
            # sets pour ne faire qu'un seul merge par PK (optimisation)
            customers_done = set()
            orders_done = set()
            events_done = set()
            # compteurs (pur info)
            customers_count = 0
            orders_count = 0
            events_count = 0
            
            # Parcours ligne par ligne du DataFrame
            # itertuples() est plus rapide que iterrows()
            # et renvoie des objets avec attributs accessibles.
            for row in df.itertuples(index=False):

                # ----------------------------
                # A) UPSERT Customer
                # ----------------------------
                # On construit un objet ORM Customer à partir de la ligne
                # un seul fois par customer_customer_id
                if getattr(row, "customer_customer_id") in customers_done:
                    pass  # déjà fait
                else:
                    customers_done.add(getattr(row, "customer_customer_id"))
                    customers_count += 1
                    customer = Customer(
                        customer_customer_id=getattr(row, "customer_customer_id"),
                        customer_country=getattr(row, "customer_country"),
                    )
                    # session.merge(obj) :
                    # - si la PK n'existe pas => INSERT
                    # - si la PK existe => UPDATE
                    # ⚠️ merge peut faire un SELECT préalable => OK pour 4000 lignes
                    session.merge(customer)

                # ----------------------------
                # B) UPSERT Order
                # ----------------------------
                # une seule fois par order_id
                if getattr(row, "order_id") in orders_done:
                    pass  # déjà fait
                else:
                    orders_done.add(getattr(row, "order_id"))
                    orders_count += 1
                    order = Order(
                        order_id=getattr(row, "order_id"),
                        customer_customer_id=getattr(row, "customer_customer_id"),
                        order_device=getattr(row, "order_device"),
                        order_channel=getattr(row, "order_channel"),
                        order_main_category=getattr(row, "order_main_category"),
                        order_n_items=getattr(row, "order_n_items"),
                        order_basket_value=getattr(row, "order_basket_value"),
                        order_shipping_fee=getattr(row, "order_shipping_fee"),
                        order_discount=getattr(row, "order_discount"),
                        order_order_total=getattr(row, "order_order_total"),
                        order_is_returned=getattr(row, "order_is_returned"),
                        price_average_per_item=getattr(row, "price_average_per_item"),
                    )   
                    session.merge(order)

                # ----------------------------
                # C) UPSERT Event
                # ----------------------------
                # une seule fois par event_id
                if getattr(row, "event_id") in events_done:
                    pass  # déjà fait
                else:
                    events_done.add(getattr(row, "event_id"))
                    events_count += 1
                    event = Event(
                        event_id=getattr(row, "event_id"),
                        event_time=getattr(row, "event_time"),
                        event_year=getattr(row, "event_year"),
                        event_month=getattr(row, "event_month"),
                        event_day=getattr(row, "event_day"),
                        event_hour=getattr(row, "event_hour"),
                        order_id=getattr(row, "order_id"),
                        customer_customer_id=getattr(row, "customer_customer_id"),
                    )
                    session.merge(event)

        # Ici, si on arrive sans exception :
        # - la transaction a été commit automatiquement par session.begin()
        # - la session sera fermée automatiquement en sortant du bloc
    print("✅ Ingestion terminée avec succès.")

except SQLAlchemyError as e:
    # Toute erreur SQLAlchemy arrive ici :
    # - la transaction est rollback automatiquement si elle était ouverte
    print("❌ Erreur lors de l'ingestion SQLAlchemy :", e)

except Exception as e:
    # capturer tout autre type d'erreur (par ex. KeyError, TypeError, etc.)
    print("❌ Erreur inattendue lors de l'ingestion :", e)
