#PreparationExamenBloc2/src/generate_data.py
"""
generate_data.py
----------------
Ce script génère un dataset réaliste au format JSON Lines (.jsonl).

Pourquoi (examen) :
- Tu peux t'entraîner sans dataset externe.
- Tu peux simuler valeurs manquantes, doublons, variations de schéma, etc.

Format JSONL :
- 1 ligne = 1 événement JSON
- Avantage : lecture stream (Kafka) facile
"""

import json
import random
import uuid
from datetime import datetime, timedelta

from src.config import RAW_JSONL_PATH, RAW_DIR
from src.eco_impact import track_phase  # NEW : mesure carbone de la génération


def random_date(start_days_ago: int = 30) -> str:
    """Retourne une date ISO (string) dans les X derniers jours."""
    # On choisit un offset aléatoire en jours
    offset_days = random.randint(0, start_days_ago)
    # On ajoute un offset aléatoire en minutes pour varier encore
    offset_minutes = random.randint(0, 24 * 60)
    # On calcule la date
    dt = datetime.utcnow() - timedelta(days=offset_days, minutes=offset_minutes)
    # On renvoie au format ISO
    return dt.isoformat() + "Z"


def generate_one_event() -> dict:
    """Génère un événement de commande (dict)."""
    # event_id unique : sert à dédoublonner
    event_id = str(uuid.uuid4())

    # customer_id : on simule des répétitions (clients récurrents)
    customer_id = f"CUST-{random.randint(1, 600):04d}"

    # order_id unique (commande)
    order_id = f"ORD-{random.randint(1, 5000):05d}"

    # Pays (catégoriel)
    country = random.choice(["FR", "ES", "DE", "IT", "BE"])

    # Device (catégoriel)
    device = random.choice(["mobile", "desktop", "tablet"])

    # Canal marketing (catégoriel)
    channel = random.choice(["seo", "ads", "email", "direct", "affiliate"])

    # Nombre d'articles (numérique)
    n_items = random.randint(1, 8)

    # Catégorie principale (catégoriel)
    main_category = random.choice(["electronics", "fashion", "home", "sports", "beauty"])

    # Prix panier (numérique) : on simule une distribution
    basket_value = round(random.uniform(10, 500), 2)

    # Frais de livraison (numérique) : parfois gratuit
    shipping_fee = 0.0 if basket_value > 80 else round(random.uniform(2, 9), 2)

    # Discount : parfois 0, parfois petit
    discount = 0.0 if random.random() < 0.7 else round(random.uniform(2, 30), 2)

    # Total
    order_total = round(max(basket_value + shipping_fee - discount, 1.0), 2)

    # Label ML (is_returned) : on crée une logique plausible
    # Exemple : fashion retourne plus, petits paniers aussi
    base_return_prob = 0.18
    if main_category == "fashion":
        base_return_prob += 0.15
    if device == "mobile":
        base_return_prob += 0.03
    if order_total < 30:
        base_return_prob += 0.08
    if channel == "email":
        base_return_prob -= 0.02

    # On tire la variable binaire
    is_returned = 1 if random.random() < base_return_prob else 0

    # Valeurs manquantes simulées :
    # - 3% channel manquant
    if random.random() < 0.03:
        channel = None

    # - 2% discount manquant
    if random.random() < 0.02:
        discount = None

    # On construit l'événement final
    event = {
        "event_id": event_id,
        "event_time": random_date(30),
        "order_id": order_id,
        "customer": {
            "customer_id": customer_id,
            "country": country,
        },
        "order": {
            "device": device,
            "channel": channel,
            "main_category": main_category,
            "n_items": n_items,
            "basket_value": basket_value,
            "shipping_fee": shipping_fee,
            "discount": discount,
            "order_total": order_total,
            "is_returned": is_returned,
        },
    }

    return event


def main(n_events: int = 4000, with_duplicates: int = 50) -> None:
    """
    Génère n_events événements dans un fichier JSONL.
    Ajoute volontairement quelques doublons d'event_id.
    """
    # NEW : On mesure TOUTE la phase de génération de données
    # => impact CPU + écritures disque associées
    with track_phase("generate_data"):
        # On s'assure que le dossier existe
        RAW_DIR.mkdir(parents=True, exist_ok=True)

        # On garde une liste d'événements pour pouvoir dupliquer certains event_id
        events = []

        for _ in range(n_events):
            events.append(generate_one_event())

        # Ajout de doublons : on réécrit certains events (même event_id)
        for _ in range(with_duplicates):
            events.append(random.choice(events))

        # On mélange l'ordre pour simuler du "flux"
        random.shuffle(events)

        # On écrit en JSON Lines
        with open(RAW_JSONL_PATH, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")

        print(f"[OK] Dataset généré : {RAW_JSONL_PATH} ({len(events)} lignes)")


if __name__ == "__main__":
    # Lancement standard
    main()