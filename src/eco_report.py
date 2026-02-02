# PreparationExamenBloc2/src/eco_report.py
"""
eco_report.py
-------------
Objectif :
- Lire les fichiers CodeCarbon dans reports/emissions/
  (emissions.csv + backups emissions.csv_*.bak, emissions.csv.bak, etc.)
- Fusionner toutes les mesures (car CodeCarbon peut "backuper" le CSV)
- Agréger les résultats par phase (project_name)
- Générer :
    - un rapport lisible : reports/eco_report.md
    - un tableau agrégé : reports/eco_report.csv

Pourquoi plusieurs fichiers ?
- CodeCarbon peut créer des backups quand le format change
  (warning "CSV format has changed, backing up old emission file")
- Ou lorsqu'il y a des écritures concurrentes (ex: kafka_pipeline lance producer + consumer)

Résultat :
- Tu obtiens une vue complète de toutes les phases trackées, même si elles
  sont réparties dans plusieurs emissions.csv*.bak
"""

from __future__ import annotations

import os  # Lire CC_OUTPUT_DIR
from glob import glob  # Trouver emissions.csv*
from pathlib import Path  # Gérer chemins de manière portable
from typing import List

import pandas as pd  # Lecture et agrégation CSV
from dotenv import load_dotenv  # Charger .env


# ============================================================
# 1) Helpers chemins
# ============================================================

def _project_root() -> Path:
    """
    Retourne le dossier racine du projet.

    Ici : src/eco_report.py -> parents[1] = PreparationExamenBloc2/
    (même logique que config.py et eco_impact.py)
    """
    return Path(__file__).resolve().parents[1]


def _resolve_emissions_dir() -> Path:
    """
    Détermine le dossier où CodeCarbon écrit ses fichiers d'émissions.

    Règles :
    - Si CC_OUTPUT_DIR est défini dans .env ou l'environnement, on l'utilise
    - Sinon, par défaut : "reports/emissions"
    - Si c'est un chemin relatif, on le rend relatif à la racine du projet
    """
    load_dotenv()

    output_dir_raw = os.getenv("CC_OUTPUT_DIR", "reports/emissions")
    output_dir = Path(output_dir_raw)

    # Si le chemin est relatif, on le rattache à la racine du projet
    if not output_dir.is_absolute():
        output_dir = _project_root() / output_dir

    return output_dir


def _list_emissions_files(emissions_dir: Path) -> List[Path]:
    """
    Liste tous les fichiers CodeCarbon potentiels :
      - emissions.csv
      - emissions.csv.bak
      - emissions.csv_4.bak, emissions.csv_5.bak, etc.

    On utilise un glob, car sur ton environnement tu as beaucoup de backups.
    """
    pattern = str(emissions_dir / "emissions.csv*")
    files = [Path(p) for p in glob(pattern)]

    # On garde uniquement les fichiers existants et réguliers
    files = [f for f in files if f.exists() and f.is_file()]

    # Tri pour un ordre stable
    files.sort()

    return files


# ============================================================
# 2) Normalisation colonnes (compat versions CodeCarbon)
# ============================================================

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonise les noms de colonnes, car CodeCarbon peut changer selon versions.

    Exemples possibles :
    - run-id -> run_id
    - duration(s) -> duration
    - emissions(kg) -> emissions
    - energy_consumed(kWh) -> energy_consumed

    On applique un rename uniquement si la colonne source existe.
    """
    rename_map = {
        "run-id": "run_id",
        "duration(s)": "duration",
        "emissions(kg)": "emissions",
        "energy_consumed(kWh)": "energy_consumed",
    }

    cols_present = {c: rename_map[c] for c in rename_map if c in df.columns}
    if cols_present:
        df = df.rename(columns=cols_present)

    return df


def _read_one_emissions_file(path: Path) -> pd.DataFrame:
    """
    Lit un fichier emissions.csv (ou backup) en DataFrame.

    - Ajoute une colonne '__source_file' pour faciliter le debug.
    - Normalise les noms de colonnes.
    """
    tmp = pd.read_csv(path)

    # Ajout de la source : pratique pour comprendre d'où vient une ligne
    tmp["__source_file"] = str(path)

    # Normalisation colonnes
    tmp = _normalize_columns(tmp)

    return tmp


def _load_all_emissions(files: List[Path]) -> pd.DataFrame:
    """
    Charge tous les fichiers émissions trouvés et les concatène.

    - Ignore les fichiers illisibles (sans faire échouer le rapport)
    - Dédoublonne ensuite (run_id si présent)
    """
    dfs: List[pd.DataFrame] = []

    for fp in files:
        try:
            dfs.append(_read_one_emissions_file(fp))
        except Exception as e:
            # On log, mais on n'arrête pas le report
            print(f"[WARN] Impossible de lire {fp} -> ignoré ({e})")

    if not dfs:
        raise ValueError(
            "Aucun fichier d'émissions lisible. "
            "Vérifie le contenu de reports/emissions/."
        )

    df = pd.concat(dfs, ignore_index=True)

    # Dédoublonnage :
    # - Si run_id existe : c'est la clé la plus fiable (chaque tracking a un run_id unique)
    if "run_id" in df.columns:
        df = df.drop_duplicates(subset=["run_id"], keep="last")
    else:
        # Sinon, fallback : on dédoublonne sur un subset raisonnable
        subset = [c for c in ["project_name", "timestamp", "duration", "emissions"] if c in df.columns]
        if subset:
            df = df.drop_duplicates(subset=subset, keep="last")

    return df


# ============================================================
# 3) Report (agrégation + export)
# ============================================================

def main() -> None:
    """
    Point d'entrée :
    - Trouve les fichiers CodeCarbon
    - Fusionne les mesures
    - Agrège par project_name (phase)
    - Écrit reports/eco_report.csv + reports/eco_report.md
    """
    # 1) Localiser le dossier d'émissions
    emissions_dir = _resolve_emissions_dir()

    # On ne crée pas emissions_dir ici : normalement eco_impact.py/CodeCarbon le crée.
    # Mais si tu veux éviter un crash "dossier absent", tu peux le créer :
    emissions_dir.mkdir(parents=True, exist_ok=True)

    # 2) Lister tous les fichiers d'émissions
    files = _list_emissions_files(emissions_dir)

    if not files:
        raise FileNotFoundError(
            f"Aucun fichier 'emissions.csv*' trouvé dans {emissions_dir}. "
            "Lance d'abord au moins un script instrumenté avec track_phase(...)."
        )

    # 3) Charger et fusionner toutes les mesures
    df = _load_all_emissions(files)

    if df.empty:
        raise ValueError(
            f"Les fichiers emissions.csv* trouvés dans {emissions_dir} sont vides. "
            "Lance un script instrumenté puis relance eco_report.py."
        )

    # 4) Vérifier colonnes minimales
    # Pour faire un rapport, on a au moins besoin de :
    # - project_name : le nom de la phase (ex: bloc2::etl, bloc2::ml_train_regression, etc.)
    # - emissions : valeur CO2e en kg
    required_cols = {"project_name", "emissions"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(
            f"Colonnes manquantes dans les fichiers émissions: {sorted(list(missing))}. "
            f"Colonnes disponibles: {list(df.columns)}"
        )

    # 5) Préparer la map d'agrégation :
    # emissions = somme (kgCO2e)
    # energy_consumed = somme (kWh) si présent
    # duration = somme (s) si présent
    agg_map = {"emissions": "sum"}

    if "energy_consumed" in df.columns:
        agg_map["energy_consumed"] = "sum"
    if "duration" in df.columns:
        agg_map["duration"] = "sum"

    # 6) Agrégation par phase (project_name)
    summary = (
        df.groupby("project_name", as_index=False)
          .agg(agg_map)
          .sort_values("emissions", ascending=False)
    )

    # 7) Colonnes dérivées pour lecture humaine
    summary["emissions_g"] = summary["emissions"] * 1000.0  # kg -> g

    if "duration" in summary.columns:
        summary["duration_min"] = summary["duration"] / 60.0  # s -> min

    # 8) Totaux
    total_emissions_kg = float(summary["emissions"].sum())
    total_emissions_g = float(summary["emissions_g"].sum())

    total_energy_kwh = float(summary["energy_consumed"].sum()) if "energy_consumed" in summary.columns else None
    total_duration_s = float(summary["duration"].sum()) if "duration" in summary.columns else None

    # 9) Dossier de sortie reports/
    reports_dir = _project_root() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 10) Export CSV agrégé
    out_csv = reports_dir / "eco_report.csv"
    summary.to_csv(out_csv, index=False)

    # 11) Export Markdown lisible
    out_md = reports_dir / "eco_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Rapport d'impact écologique (CodeCarbon)\n\n")

        # On indique clairement qu'on a fusionné plusieurs fichiers
        f.write("Source : fichiers fusionnés depuis `reports/emissions/emissions.csv*`\n\n")
        f.write("Fichiers pris en compte :\n")
        for fp in files:
            f.write(f"- `{fp}`\n")
        f.write("\n")

        f.write("## Résumé par phase\n\n")

        # IMPORTANT : summary.to_markdown nécessite le package optionnel 'tabulate'
        # Pour éviter un crash en exam, on prévoit un fallback.
        try:
            f.write(summary.to_markdown(index=False))
        except ImportError:
            f.write("⚠️  'tabulate' non installé : affichage en tableau texte (fallback)\n\n")
            f.write(summary.to_string(index=False))

        f.write("\n\n## Totaux\n\n")
        f.write(f"- Emissions totales : {total_emissions_kg:.6f} kgCO2e ({total_emissions_g:.2f} gCO2e)\n")

        if total_energy_kwh is not None:
            f.write(f"- Énergie totale : {total_energy_kwh:.6f} kWh\n")

        if total_duration_s is not None:
            f.write(f"- Durée totale : {total_duration_s:.1f} s ({total_duration_s/60.0:.1f} min)\n")

        f.write("\n## Interprétation (guide rapide)\n\n")
        f.write("- `project_name` : nom de la phase trackée (une étape du pipeline).\n")
        f.write("- `emissions` : CO2e estimé (kg) pour la phase.\n")
        f.write("- `emissions_g` : même info en grammes (plus lisible).\n")
        f.write("- `energy_consumed` : énergie estimée (kWh) si disponible.\n")
        f.write("- `duration` : temps total de calcul (secondes) si disponible.\n")
        f.write("- Compare les phases : la plus forte émission = phase la plus coûteuse.\n")
        f.write("- Note : sur certaines VM, CodeCarbon estime l'énergie (mode cpu_load/TDP), c'est une estimation.\n")

    print(f"[OK] Rapport écrit : {out_md}")
    print(f"[OK] Tableau agrégé : {out_csv}")


if __name__ == "__main__":
    main()
