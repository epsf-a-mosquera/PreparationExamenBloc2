#PreparationExamenBloc2/src/eco_impact.py
"""
Objectif :
- Fournir un outil simple (track_phase) pour mesurer l'impact carbone (CO2e)
  de différentes phases du projet : ETL, ingestion SQL, training ML, Kafka, etc.
- Écrire les mesures dans un fichier CSV unique : reports/emissions/emissions.csv

Pourquoi cette approche (examen) ?
- Modifications MINIMALES dans tes scripts : tu ajoutes juste "with track_phase(...):"
- Chaque phase produit 1 entrée mesurée (durée, énergie, émissions)
- Tu peux ensuite agréger toutes les phases avec eco_report.py

CodeCarbon écrit par défaut un fichier emissions.csv dans output_dir. (doc)
"""

from __future__ import annotations  # Permet d'utiliser certains types "modernes" selon la version Python

import os  # Accès aux variables d'environnement (CC_OFFLINE, CC_OUTPUT_DIR, ...)
from pathlib import Path  # Gestion robuste des chemins (Linux/Windows)
from contextlib import contextmanager  # Pour créer un context manager "with ..."
from typing import Iterator  # Pour typer le yield du context manager

from dotenv import load_dotenv  # Pour charger les variables depuis .env


def _project_root() -> Path:
    """
    Retourne le dossier racine du projet.

    Ici : src/eco_impact.py -> parents[1] = racine du projet.
    Exemple :
      /.../PreparationExamenBloc2/src/eco_impact.py
      parents[1] => /.../PreparationExamenBloc2/
    """
    return Path(__file__).resolve().parents[1]


def _to_bool(value: str) -> bool:
    """
    Convertit une string d'env en bool.
    Ex: "true", "1", "yes" => True.
    """
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _resolve_output_dir() -> Path:
    """
    Résout le dossier de sortie (output_dir) pour emissions.csv.

    - Si CC_OUTPUT_DIR n'est pas défini -> défaut "reports/emissions"
    - Si c'est un chemin relatif -> on le rattache à la racine du projet
    - On crée le dossier s'il n'existe pas (important pour ton exigence)
    """
    # Charge le .env (si présent) pour que os.getenv(...) récupère les valeurs
    load_dotenv()

    # Valeur par défaut si la variable n'existe pas
    raw = os.getenv("CC_OUTPUT_DIR", "reports/emissions")

    # Transforme en Path
    p = Path(raw)

    # Si le chemin est relatif, on le rend relatif au projet
    if not p.is_absolute():
        p = _project_root() / p

    # Crée le dossier (parents=True => crée aussi reports/ si besoin)
    p.mkdir(parents=True, exist_ok=True)

    return p


def _build_tracker(phase_name: str):
    """
    Construit et retourne un tracker CodeCarbon.

    - Offline recommandé en examen (pas besoin d'internet) :
        OfflineEmissionsTracker(country_iso_code="FRA", ...)
    - Online si CC_OFFLINE=false :
        EmissionsTracker(...)
    """
    load_dotenv()  # On recharge au cas où le .env a été modifié

    # Lire les paramètres
    offline = _to_bool(os.getenv("CC_OFFLINE", "true"))
    country_iso_code = os.getenv("CC_COUNTRY_ISO_CODE", "FRA")
    measure_power_secs = int(os.getenv("CC_MEASURE_POWER_SECS", "10"))

    # Dossier où sera écrit emissions.csv
    output_dir = _resolve_output_dir()

    # On construit un project_name clair => utile pour regrouper ensuite
    # Exemple : "bloc2::etl_extract_transform"
    project_name = f"bloc2::{phase_name}"

    # Import local (au moment de l'usage) pour éviter tout problème d'import au chargement
    if offline:
        # Mode offline : nécessite country_iso_code (ISO-3)
        # Doc : OfflineEmissionsTracker(country_iso_code=...) (CodeCarbon)
        from codecarbon import OfflineEmissionsTracker  # type: ignore

        tracker = OfflineEmissionsTracker(
            project_name=project_name,
            country_iso_code=country_iso_code,
            output_dir=str(output_dir),
            output_file="emissions.csv",
            measure_power_secs=measure_power_secs,
            save_to_file=True,     # écrit le CSV
            save_to_api=False,     # évite dépendance réseau
        )
        return tracker

    # Mode online : utilise intensité carbone via sources réseau (si disponible)
    from codecarbon import EmissionsTracker  # type: ignore

    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=str(output_dir),
        output_file="emissions.csv",
        measure_power_secs=measure_power_secs,
        save_to_file=True,
        save_to_api=False,
    )
    return tracker


@contextmanager
def track_phase(phase_name: str) -> Iterator[None]:
    """
    Context manager pour mesurer une phase de ton pipeline.

    Usage minimal dans un script :
        from src.eco_impact import track_phase

        def main():
            with track_phase("etl"):
                ... ton code ...

    Fonctionnement :
    - start() démarre la mesure (puissance/énergie)
    - stop() termine et écrit une ligne dans emissions.csv
    """
    tracker = _build_tracker(phase_name)  # Construit le tracker configuré
    tracker.start()  # Démarre la mesure

    try:
        yield  # Le code "mesuré" s'exécute ici
    finally:
        # stop() finalise et écrit dans emissions.csv
        tracker.stop()

