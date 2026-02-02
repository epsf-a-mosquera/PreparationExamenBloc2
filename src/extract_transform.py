#ExamenBloc2/src/extract_transform.py
'''
Celui-ci est le script d'extraction et de transformation des données JSON.
Il lit les fichiers JSON, extrait les informations pertinentes, les transforme en un format structuré (DataFrame pandas), après nous rélisons des opérations de nettoyage et de transformation des données (les operation du notebook d'exploration).
Il sauvegarde ensuite les données transformées dans un fichier CSV et un fichier parquet pour une utilisation ultérieure.
à la fin, il génère des features pour le modèle ML.
'''
import os # Pour les opérations liées au système de fichiers
import pathlib # Pour la manipulation des chemins de fichiers 
import json # Pour lire les fichiers JSON
import math # Pour les opérations mathématiques si nécessaire
import pandas as pd # Pour la manipulation des données
import numpy as np # Pour les opérations numériques
from src.config import RAW_JSONL_PATH, CLEAN_CSV_PATH, CLEAN_PARQUET_PATH, CLEAN_JSONL_PATH # Importer les configurations depuis le fichier config.py

# Fonction generique pour aplatir les colonnes imbriquées dans un DataFrame pandas
def flatten_nested_columns(df):
    """
    Aplati automatiquement les colonnes contenant des dictionnaires ou des listes
    dans un DataFrame pandas.
    """
    # Identifier les colonnes contenant des données imbriquées (dict ou list)
    nested_cols = [
        col for col in df.columns
        if df[col]
        .dropna()  # ignorer les valeurs NaN
        .apply(lambda x: isinstance(x, (dict, list)))  # vérifier le type
        .any()  # au moins une valeur imbriquée
    ]
    # Parcourir chaque colonne imbriquée
    for col in nested_cols:
        flattened = pd.json_normalize(df[col])          # Transformer le contenu JSON en colonnes plates
        flattened.columns = [f"{col}_{subcol}" for subcol in flattened.columns]     # Renommer les colonnes générées avec le préfixe de la colonne d'origine
        df = df.drop(columns=[col])                      # Supprimer la colonne imbriquée d'origine
        df = pd.concat([df, flattened], axis=1)     # Concaténer les nouvelles colonnes aplaties au DataFrame principal

    # Retourner le DataFrame final aplati
    return df

def upsert_clean_dataset(
    df_new: pd.DataFrame,
    csv_path: str,
    parquet_path: str,
    jsonl_path: str,
    unique_id_col: str = "event_id",
    updated_at_col: str = "event_time",
) -> pd.DataFrame:
    """
    Réalise un UPSERT "fichier" (CSV/Parquet) :
      - ajoute les nouvelles lignes si l'identifiant unique n'existe pas
      - met à jour les lignes existantes si l'identifiant existe déjà
        (en gardant la version la plus récente selon updated_at_col)

    Pourquoi on réécrit le fichier ?
      - Parce qu'un CSV/Parquet ne permet pas de "modifier en place" une ligne existante.
      - Donc on lit l'existant, on fusionne, on dédoublonne, puis on réécrit.
    """

    # ----------------------------
    # 1) Charger l'existant si présent
    # ----------------------------
    # On privilégie le Parquet si disponible (souvent mieux typé que le CSV).
    df_existing = None

    if os.path.exists(parquet_path):
        # Lire le dataset existant en Parquet
        df_existing = pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        # Sinon, fallback sur le CSV
        df_existing = pd.read_csv(csv_path)

    # Si aucun fichier clean n'existe, l'UPSERT revient simplement à "écrire df_new"
    if df_existing is None:
        df_merged = df_new.copy()
    else:
        # ----------------------------
        # 2) Aligner les colonnes (robuste aux ajouts de colonnes dans le temps)
        # ----------------------------
        # On crée l'union des colonnes entre df_existing et df_new
        all_cols = sorted(set(df_existing.columns).union(set(df_new.columns)))

        # On réindexe les deux DataFrames sur les mêmes colonnes
        # (les colonnes absentes seront remplies par NaN)
        df_existing = df_existing.reindex(columns=all_cols)
        df_new = df_new.reindex(columns=all_cols)

        # ----------------------------
        # 3) Fusionner : existant + nouveau
        # ----------------------------
        df_merged = pd.concat([df_existing, df_new], ignore_index=True)

    # ----------------------------
    # 4) Sécurités : clés / dates
    # ----------------------------
    # Si la colonne identifiant n'existe pas, on ne peut pas upsert => erreur explicite
    if unique_id_col not in df_merged.columns:
        raise KeyError(
            f"Colonne identifiant unique '{unique_id_col}' introuvable. "
            f"Colonnes disponibles: {list(df_merged.columns)}"
        )

    # Si la colonne de "date de mise à jour" existe, on s'en sert pour garder le plus récent
    # (sinon, on garde arbitrairement la dernière occurrence)
    if updated_at_col in df_merged.columns:
        # Convertir en datetime si ce n'est pas déjà le cas (robuste aux lectures CSV)
        df_merged[updated_at_col] = pd.to_datetime(
            df_merged[updated_at_col],
            errors="coerce",
            utc=True
        )

        # Trier du plus récent au plus ancien
        df_merged = df_merged.sort_values(by=updated_at_col, ascending=False)

        # Dédoublonner sur l'identifiant unique :
        # keep="first" => garde la ligne la plus récente (car tri descendant)
        df_merged = df_merged.drop_duplicates(subset=[unique_id_col], keep="first")
    else:
        # Pas de colonne temps => on garde la dernière occurrence rencontrée
        df_merged = df_merged.drop_duplicates(subset=[unique_id_col], keep="last")

    # Optionnel mais propre : remettre un index propre
    df_merged = df_merged.reset_index(drop=True)

    # ----------------------------
    # 5) Écrire le dataset final
    # ----------------------------
    # CSV : facile à relire, mais types parfois moins fiables
    df_merged.to_csv(csv_path, index=False)

    # Parquet : mieux pour les types/perf
    df_merged.to_parquet(parquet_path, index=False)
    
    # json : mieux pour les types/perf
    df_merged.to_json(jsonl_path, orient="records", lines=True) # lines=True json lines format

    return df_merged


# Fonction principale d'extraction et de transformation
def extract_transform():
    # Sécurité : vérifier que le fichier source existe
    if not os.path.exists(RAW_JSONL_PATH):
        raise FileNotFoundError(f"Le fichier source {RAW_JSONL_PATH} n'existe pas.")
    
    # Étape d'extraction et de transformation
    # convertir le fichier JSON RAW_JSONL_PATH dans un DataFrame pandas
    df=pd.read_json(RAW_JSONL_PATH, lines=True)
    # aplatissement des colonnes imbriquées (si nécessaire)
    df=flatten_nested_columns(df)
    # Nettoyage léger des colonnes teste (strip des espaces, lowercase, uppercase, etc.)
    # On définit les colonnes texte qu'on veut normaliser
    # Objectif : éviter des différences artificielles ("Mobile" vs "mobile" vs " mobile ")
    text_cols = [
        "event_id",
        "order_id",
        "customer_customer_id",
        "customer_country",
        "order_device",
        "order_channel",
        "order_main_category"
    ]
    # On boucle sur chaque colonne
    for col in text_cols:
        if col in df.columns:                       # on s'assure que la colonne est bien présente
            df[col] = df[col].astype("string")      # "string" pandas gère mieux les NA que object classique
            df[col] = df[col].str.strip()           # strip() supprime les espaces au début et à la fin
    
    # On met en minuscules pour homogénéiser certaines catégories
    # (pas event_id / order_id / customer_id, car ce sont des identifiants)
    for col in text_cols:
        if col not in ["event_id", "order_id", "customer_customer_id", "customer_country"]:
            df[col] = df[col].str.lower()       # lower() met le texte en minuscules
    
    # Pour le pays, on préfère des codes ISO en MAJ (DE, ES, IT)
    if "customer_country" in df.columns:
        df["customer_country"] = df["customer_country"].str.upper()  # upper() met en majuscules
        
    # Conversion des types de données du DataFrame
    # ======================================================
    # Conversion des colonnes d'identifiants
    # → utilisation du type 'string' pandas (plus robuste que 'object')
    # ======================================================
    df["event_id"] = df["event_id"].astype("string")
    df["order_id"] = df["order_id"].astype("string")
    df["customer_customer_id"] = df["customer_customer_id"].astype("string")

    # ======================================================
    # Conversion de la colonne temporelle
    # → transformation de la date ISO 8601 en datetime avec timezone UTC
    # pd.to_datetime transforme une colonne texte en type datetime utilisable (tri, extraction d'année, etc.)
    # errors="coerce" : si un format n'est pas convertible, pandas met NaT (équivalent datetime de NaN)
    # utc=True : force la timezone UTC (cohérent avec le suffixe "Z" dans tes dates)
    # ======================================================
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)

    # ======================================================
    # Conversion des colonnes catégorielles
    # → type 'category' pour réduire la mémoire et faciliter les analyses
    # ======================================================
    categorical_cols = [
        "customer_country",
        "order_device",
        "order_channel",
        "order_main_category"
    ]

    df[categorical_cols] = df[categorical_cols].astype("category")

    # ======================================================
    # Conversion des colonnes numériques entières
    # → nombre d’articles dans la commande
    # ======================================================
    df["order_n_items"] = df["order_n_items"].astype("int64")

    # ======================================================
    # Conversion des colonnes numériques décimales
    # → montants financiers (float)
    # ======================================================
    numeric_cols = [
        "order_basket_value",
        "order_shipping_fee",
        "order_discount",
        "order_order_total"
    ]

    df[numeric_cols] = df[numeric_cols].astype("float64")

    # ======================================================
    # Conversion de l’indicateur de retour
    # → transformation de 0 / 1 en booléen (False / True)
    # ======================================================
    df["order_is_returned"] = df["order_is_returned"].astype("bool")
    
    # Gestion des doublons (duplicated + drop_duplicates)
    # drop_duplicates() supprime les doublons
    # keep="first" garde la première occurrence et supprime les suivantes
    df = df.drop_duplicates(keep="first")
    
    # Doublons métier sur event_id (garder le plus récent)
    df = df.sort_values(by="event_time", ascending=False)
    df = df.drop_duplicates(subset=["event_id"], keep="first")
    
    # Gestion des valeurs manquantes (NaN)
    
    # Fill NA catégoriels (mode ou valeur “unknown”)
    # Pour customer_country, si manquant => "UN" (Unknown)
    # Creer la nouvelle categorie 
    if "customer_country" in df.columns:
        df["customer_country"] = (
            df["customer_country"]
            .cat.add_categories(["UN"])
            .fillna("UN")
        )

    # Pour device/channel/category, si manquant => la mode (valeur la plus fréquente)
    for col in ["order_device", "order_channel", "order_main_category"]:
        if col in df.columns:
            mode_value = df[col].mode()[0]  # mode() renvoie une Series, on prend la première valeur
            df[col] = df[col].fillna(mode_value)  # fillna remplace NA par la mode

    
    # Fill NA numériques (stratégie simple et robuste)
    # (1) shipping_fee et discount :
    # Si absent, il est souvent raisonnable de considérer 0 (pas de frais / pas de remise)
    for col in ["order_shipping_fee", "order_discount"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)  # remplace NA par 0.0 (float)

    # (2) order_n_items :
    # On remplace par la médiane (plus robuste que la moyenne si outliers)
    if "order_n_items" in df.columns:
        median_items = df["order_n_items"].median()     # calcule la médiane
        df["order_n_items"] = df["order_n_items"].fillna(median_items)  # remplace NA par médiane
        df["order_n_items"] = df["order_n_items"].round().astype(int)   # arrondi puis cast en int

    # (3) order_basket_value :
    # Si absent, on remplace par la médiane (stratégie baseline)
    if "order_basket_value" in df.columns:
        median_basket = df["order_basket_value"].median()                  # calcule médiane
        df["order_basket_value"] = df["order_basket_value"].fillna(median_basket)  # remplace NA
        
    # Extraire des features temporelles
    # dt.year / dt.month / dt.day / dt.hour :
    # fonctionne uniquement si event_time est bien en datetime
    df["event_year"] = df["event_time"].dt.year      # extrait l'année
    df["event_month"] = df["event_time"].dt.month    # extrait le mois
    df["event_day"] = df["event_time"].dt.day        # extrait le jour
    df["event_hour"] = df["event_time"].dt.hour      # extrait l'heure
    
    # On crée une nouvelle colonne prix moyen par article "price_average_per_item"
    # = order_basket_value / order_n_items
    # On utilise apply + lambda pour appliquer la fonction ligne par ligne
    df["price_average_per_item"] = df.apply(
        lambda row: row["order_basket_value"] / row["order_n_items"] if row["order_n_items"] > 0 else np.nan,
        axis=1  # axis=1 signifie qu'on applique la fonction sur les lignes
    )
    
    # ======================================================
    # UPSERT fichier (au lieu d'écraser)
    # - si event_id n'existe pas => ajout
    # - si event_id existe déjà   => mise à jour (plus récent event_time)
    # ======================================================
    df = upsert_clean_dataset(
        df_new=df,
        csv_path=CLEAN_CSV_PATH,
        parquet_path=CLEAN_PARQUET_PATH,
        jsonl_path=CLEAN_JSONL_PATH,
        unique_id_col="event_id",     # <-- identifiant unique métier
        updated_at_col="event_time",  # <-- critère pour choisir la version la plus récente
    )

    print(f"DataFrame clean UPSERT sauvegardé dans {CLEAN_CSV_PATH} et {CLEAN_PARQUET_PATH} et {CLEAN_JSONL_PATH}")

    
    # Génération de statistiques descriptives pour vérification
    print("Aperçu du DataFrame nettoyé :")
    print(df.info())
    print("Types de données finaux :")
    print(df.dtypes)
    print(f"aperçu des données :")
    print(df[categorical_cols].head(10))
    with open("describe.txt", "w", encoding="utf-8") as f:
        f.write(df.describe(include="all").to_string())


if __name__ == "__main__":
    extract_transform()