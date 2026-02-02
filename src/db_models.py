#ExamenBloc2/src/db_models.py
'''
celui-ci est le script qui va définir les modèles de données pour interagir avec la base de données MySQL en utilisant SQLAlchemy ORM.
Il inclut la définition des tables, des colonnes, des types de données, et des relations entre les tables si nécessaire.
'''
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey # Importer les types de colonnes nécessaires
from sqlalchemy.orm import declarative_base # Importer la base déclarative pour les modèles
from sqlalchemy.orm import relationship # Importer la fonction pour définir les relations entre les tables
from sqlalchemy import create_engine # Importer la fonction pour créer une connexion à la base de données
from src.config import  SQLALCHEMY_DATABASE_URL # Importer l'URL de connexion MySQL depuis le fichier de configuration avec SQLALCHEMY_DATABASE_URL
# from src.config import MYSQL_TABLE_NAME # Importer le nom de la table depuis le fichier de configuration
Base = declarative_base() # Créer la base déclarative pour les modèles
"""
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

class Event(Base):
    """
    Modèle SQLAlchemy pour représenter un événement de commande dans la base de données MySQL.
    """
    __tablename__ = "table_events"  # Nom de la table dans la base de données

    event_id = Column(String(36), primary_key=True)  # ID unique de l'événement (UUID)
    event_time = Column(DateTime(timezone=True), nullable=False)  # Horodatage de l'événement
    event_year = Column(Integer, nullable=False)  # Année de l'événement
    event_month = Column(Integer, nullable=False)  # Mois de l'événement
    event_day = Column(Integer, nullable=False)  # Jour de l'événement
    event_hour = Column(Integer, nullable=False)  # Heure de l'événement
    order_id = Column(String(36), ForeignKey("table_orders.order_id"), nullable=False)  # ID de la commande associée à l'événement
    customer_customer_id = Column(String(36), ForeignKey("table_customers.customer_customer_id"), nullable=False) # ID du client associé à l'événement
    # Relations
    # "back_populates" permet de synchroniser les deux côtés de la relation
    customer = relationship("Customer", back_populates="events")  # Relation avec le client de l'événement
    order = relationship("Order", back_populates="events")  # Relation avec la commande de l'événement
    prediction = relationship("Prediction", back_populates="event", uselist=False)  # Relation avec la prédiction associée à l'événement (1-1)
    
    
class Order(Base):
    """
    Modèle SQLAlchemy pour représenter une commande dans la base de données MySQL.
    """
    __tablename__ = "table_orders"  # Nom de la table dans la base de données

    order_id = Column(String(36), primary_key=True)  # ID unique de la commande (UUID)
    customer_customer_id = Column(String(36), ForeignKey("table_customers.customer_customer_id"), nullable=False)  # ID du client associé à la commande
    order_device = Column(String(20), nullable=False)  # Type d'appareil utilisé pour passer la commande
    order_channel = Column(String(20), nullable=False)  # Canal par lequel la commande a été passée
    order_main_category = Column(String(50), nullable=False)  # Catégorie principale des produits commandés
    order_n_items = Column(Integer, nullable=False)  # Nombre d'articles dans la commande
    order_basket_value = Column(Float, nullable=False)  # Valeur totale du panier avant frais et remises
    order_shipping_fee = Column(Float, nullable=False)  # Frais de livraison pour la commande
    order_discount = Column(Float, nullable=False)  # Remise appliquée à la commande
    order_order_total = Column(Float, nullable=False)  # Valeur totale finale de la commande
    order_is_returned = Column(Boolean, nullable=False)  # Indicateur si la commande a été retournée
    price_average_per_item = Column(Float, nullable=False)  # Prix moyen par article dans la commande
    # Relations
    # "back_populates" permet de synchroniser les deux côtés de la relation
    customer = relationship("Customer", back_populates="orders")  # Relation avec le client de la commande
    events = relationship("Event", back_populates="order")  # Relation avec les événements de la commande
    predictions = relationship("Prediction", back_populates="order")  # Relation avec les prédictions de la commande


class Customer(Base):
    """
    Modèle SQLAlchemy pour représenter un client dans la base de données MySQL.
    """
    __tablename__ = "table_customers"  # Nom de la table dans la base de données

    customer_customer_id = Column(String(36), primary_key=True)  # ID unique du client (UUID)
    customer_country = Column(String(50), nullable=False)  # Pays du client
    
    # Relations
    # "back_populates" permet de synchroniser les deux côtés de la relation
    orders = relationship("Order", back_populates="customer")  # Relation avec les commandes du client
    events = relationship("Event", back_populates="customer")  # Relation avec les événements du client
    predictions = relationship("Prediction", back_populates="customer")  # Relation avec les prédictions du client
 
class Prediction(Base):
    """
    Modèle SQLAlchemy pour représenter une prédiction dans la base de données MySQL.
    """
    __tablename__ = "table_predictions"  # Nom de la table dans la base de données

    event_id = Column(String(36), ForeignKey("table_events.event_id"), primary_key=True)  # ID unique de l'événement (UUID)
    order_id = Column(String(36), ForeignKey("table_orders.order_id"), nullable=False)  # ID de la commande associée à la prédiction
    customer_customer_id = Column(String(36), ForeignKey("table_customers.customer_customer_id"), nullable=False)  # ID du client associé à la commande
    return_proba = Column(Float, nullable=False)  # Probabilité que la commande soit retournée

    # Relations
    # "back_populates" permet de synchroniser les deux côtés de la relation
    customer = relationship("Customer", back_populates="predictions")  # Relation avec le client de la commande
    order = relationship("Order", back_populates="predictions")  # Relation avec la commande de la prédiction
    event = relationship("Event", back_populates="prediction", uselist=False)  # Relation avec l'événement de la prédiction
    

if __name__ == "__main__":
    engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True)
    Base.metadata.create_all(engine)
    print("Tables ensured (created if missing).") 
