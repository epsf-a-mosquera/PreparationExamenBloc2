"""
Configuration des chemins pour le notebook Jupyter
"""
from pathlib import Path  # Pour manipuler les chemins de fichiers de façon indépendante du système d'exploitation
import json                # Pour lire/écrire des fichiers JSON
import math                # Fournit des fonctions mathématiques (ex: arrondis, contrôle de valeurs)
import pandas as pd        # Librairie pour l'analyse et la manipulation de données tabulaires
import numpy as np         # Gestion avancée des tableaux et des valeurs manquantes (np.nan)
import matplotlib.pyplot as plt  # Visualisation de données

# ------------------------------------------------------------------------
# Définition des chemins de projet à partir de src/config.py
# ------------------------------------------------------------------------

from src.config import PROJECT_DIR, DATA_DIR, RAW_DIR, PROCESSED_DIR, RAW_JSONL_PATH

# PROJECT_DIR : dossier racine du projet
# PROJECT_DIR = Path(__file__).resolve().parents[1] 
# Dans le notebook ce rasin est différent, il faut ajuster en conséquence si besoin
# PROJECT_DIR = Path.cwd().parents[1]

# DATA_DIR : dossier principal pour toutes les données
# DATA_DIR = PROJECT_DIR / "data"

# RAW_DIR : dossier pour les données brutes
# RAW_DIR = DATA_DIR / "raw"

# PROCESSED_DIR : dossier pour les données nettoyées
# PROCESSED_DIR = DATA_DIR / "processed"

# Dataset brut (JSON Lines : 1 JSON par ligne)
# RAW_JSONL_PATH = RAW_DIR / "orders_events.jsonl"

# ------------------------------------------------------------------------
# Vérification et création des dossiers si nécessaire
# ------------------------------------------------------------------------
for folder in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------
# Affichage des chemins pour vérification
# ------------------------------------------------------------------------
print("Chemins configurés :")
print(f"Data folder      : {DATA_DIR}")
print(f"Raw data folder  : {RAW_DIR}")
print(f"Raw data path    : {RAW_JSONL_PATH}")
print(f"Processed folder : {PROCESSED_DIR}")


with RAW_JSONL_PATH.open('r', encoding='utf-8') as f: # On ouvre le fichier brut
    lines = f.read().splitlines() # On lit toutes les lignes (une ligne = un JSON)
print (lines[0]) # Affiche la première ligne pour vérification
print (len(lines)) # Affiche le nombre total de lignes pour vérification

#Lecture des données JSON Lines dans une liste de dictionnaires
data = [json.loads(line) for line in lines]
print(f"Nombre total d'enregistrements chargés : {len(data)}")  # Affiche le nombre total d'enregistrements chargés

#Conversion de la liste de dictionnaires en DataFrame pandas pour une manipulation plus facile
df = pd.DataFrame(data)
print(f"Dimensions du DataFrame : {df.shape}")  # Affiche les dimensions du DataFrame (lignes, colonnes)
print("Aperçu des premières lignes du DataFrame :")
print(df.head())  # Affiche les premières lignes du DataFrame pour vérification

# Affichage des types de données pour chaque colonne pour chaque dimensionnement du DataFrame
print("Types de données des colonnes :")
print(df.dtypes)    # Affiche les types de données des colonnes du DataFrame pour vérification

def flatten_json_columns(df):
    """
    Aplatit automatiquement toutes les colonnes contenant des dictionnaires.
    Args:
        df (pd.DataFrame): DataFrame avec éventuellement des colonnes de type dict
    Returns:
        pd.DataFrame: DataFrame aplati
    """
    import pandas as pd 
    df_flat = df.copy()
    for col in df_flat.columns:
        # Vérifie si la colonne contient des dictionnaires (au moins une ligne)
        if df_flat[col].apply(lambda x: isinstance(x, dict)).any():
            # Aplatit la colonne
            normalized = pd.json_normalize(df_flat[col])
            # Renomme les colonnes avec un préfixe pour éviter les collisions
            normalized.columns = [f"{col}_{subcol}" for subcol in normalized.columns]
            # Supprime la colonne originale et joint les nouvelles colonnes aplaties
            df_flat = df_flat.drop(columns=[col]).join(normalized)
    return df_flat

# Applique l'aplatissement des colonnes JSON
df_flat = flatten_json_columns(df)
print(f"Dimensions du DataFrame aplati : {df_flat.shape}")  # Affiche les dimensions du DataFrame aplati
print("Aperçu des premières lignes du DataFrame aplati :")
print(df_flat.head())  # Affiche les premières lignes du DataFrame aplati pour vérification
print("Types de données des colonnes :")
print(df_flat.dtypes)    # Affiche les types de données des colonnes du DataFrame aplati pour vérification

'''
Dimensions du DataFrame aplati : (4050, 14)
Aperçu des premières lignes du DataFrame aplati :
                               event_id                   event_time  \
0  4421d292-0f86-4453-bab9-995256885cca  2026-01-08T17:59:18.973023Z   
1  798cbc4e-706e-460a-aa59-46e3549fc957  2026-01-21T08:14:19.037883Z   
2  6cf51636-aa0a-4ea6-be74-2bf2b170859b  2026-01-08T07:08:19.025822Z   
3  373b53a7-a484-459d-a157-5426cd99b979  2026-01-10T08:46:19.012009Z   
4  002c11e5-c0b6-4004-ad00-1b981537a126  2026-01-15T16:58:19.023840Z   

    order_id customer_customer_id customer_country order_device order_channel  \
0  ORD-00563            CUST-0500               DE       tablet         email   
1  ORD-03012            CUST-0458               ES       tablet         email   
2  ORD-04779            CUST-0239               DE       mobile         email   
3  ORD-01318            CUST-0180               IT       tablet           ads   
4  ORD-03652            CUST-0171               ES       mobile     affiliate   

  order_main_category  order_n_items  order_basket_value  order_shipping_fee  \
0         electronics              4              193.86                0.00   
1              beauty              5              240.11                0.00   
2         electronics              2               82.33                0.00   
3              beauty              4              348.46                0.00   
4             fashion              4               26.53                7.29   

   order_discount  order_order_total  order_is_returned  
0             0.0             193.86                  0  
1             0.0             240.11                  0  
2             0.0              82.33                  0  
3             0.0             348.46                  1  
4             0.0              33.82                  0  
Types de données des colonnes :
event_id                 object
event_time               object
order_id                 object
customer_customer_id     object
customer_country         object
order_device             object
order_channel            object
order_main_category      object
order_n_items             int64
order_basket_value      float64
order_shipping_fee      float64
order_discount          float64
order_order_total       float64
order_is_returned         int64
dtype: object
'''
# Conversion des types de données du DataFrame aplati
df_flat_converted = pd.DataFrame()  # Nouveau DataFrame pour les données converties
# ======================================================
# Conversion des colonnes d'identifiants
# → utilisation du type 'string' pandas (plus robuste que 'object')
# ======================================================
df_flat_converted["event_id"] = df_flat["event_id"].astype("string")
df_flat_converted["order_id"] = df_flat["order_id"].astype("string")
df_flat_converted["customer_customer_id"] = df_flat["customer_customer_id"].astype("string")

# ======================================================
# Conversion de la colonne temporelle
# → transformation de la date ISO 8601 en datetime avec timezone UTC
# ======================================================
df_flat_converted["event_time"] = pd.to_datetime(df_flat["event_time"], utc=True)

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

df_flat_converted[categorical_cols] = df_flat[categorical_cols].astype("category")

# ======================================================
# Conversion des colonnes numériques entières
# → nombre d’articles dans la commande
# ======================================================
df_flat_converted["order_n_items"] = df_flat["order_n_items"].astype("int64")

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

df_flat_converted[numeric_cols] = df_flat[numeric_cols].astype("float64")

# ======================================================
# Conversion de l’indicateur de retour
# → transformation de 0 / 1 en booléen (False / True)
# ======================================================
df_flat_converted["order_is_returned"] = df_flat["order_is_returned"].astype("bool")

# ======================================================
# Vérification finale des types de données
# ======================================================
df_flat_converted.dtypes

# Doublons exacts (toutes colonnes)
# duplicated() renvoie une Series booléenne :
# True si la ligne est un doublon exact d'une ligne précédente
# sum() compte le nombre de True => le nombre de doublons
nb_dup_exact = df_flat_converted.duplicated().sum()

# On affiche le nombre de doublons exacts
print("Doublons exacts (toutes colonnes) :", nb_dup_exact)

# drop_duplicates() supprime les doublons
# keep="first" garde la première occurrence et supprime les suivantes
df_flat_converted_unique = df_flat_converted.drop_duplicates(keep="first")

# On affiche la nouvelle taille
print("Dimensions après suppression doublons exacts :", df_flat_converted_unique.shape)

# duplicated(subset=["event_id"]) détecte les doublons uniquement sur event_id
# si event_id doit être unique, on veut n'en garder qu'un seul
nb_dup_event = df_flat_converted_unique.duplicated(subset=["event_id"]).sum()
print("Doublons sur event_id :", nb_dup_event)

# Pour garder l'événement le plus récent, on trie d'abord par event_time
# sort_values("event_time") met les plus anciennes en haut, les plus récentes en bas
df_flat_converted_unique = df_flat_converted_unique.sort_values("event_time")

# drop_duplicates(..., keep="last") garde la dernière occurrence (donc la plus récente après tri)
df_flat_converted_unique = df_flat_converted_unique.drop_duplicates(subset=["event_id"], keep="last")

# On affiche la taille finale après ce dédoublonnage
print("Dimensions après dédoublonnage event_id :", df_flat_converted_unique.shape)

##############################
# Contrôles de cohérence métier
##############################
# On crée un dictionnaire pour stocker des "compteurs" d'erreurs/valeurs suspectes
checks = {}

# (1) order_n_items négatif => suspect
checks["order_n_items_negative"] = (df_flat_converted_unique["order_n_items"] < 0).sum()

# (2) Montants négatifs => suspect (à adapter si tu gères des avoirs/retours en négatif)
money_cols = ["order_basket_value", "order_shipping_fee", "order_discount", "order_order_total"]
for col in money_cols:
    checks[f"{col}_negative"] = (df_flat_converted_unique[col] < 0).sum()

# (3) event_time manquant => difficilement exploitable
checks["event_time_missing"] = df_flat_converted_unique["event_time"].isna().sum()

# On convertit le dict en DataFrame pour l'afficher joliment
print (pd.DataFrame.from_dict(checks, orient="index", columns=["count"]))

# ======================================================

##########################
# # Gestion des valeurs manquantes (NaN)
##########################
# Détection des NA (isna + any + sum)
# isna() -> DataFrame booléen (True si NA)
# any(axis=0) -> pour chaque colonne : True si au moins un NA existe
cols_with_na = df_flat_converted_unique.isna().any(axis=0)

# On liste les colonnes concernées
print("Colonnes contenant au moins un NA :")
print(list(cols_with_na[cols_with_na].index))

# any(axis=1) -> pour chaque ligne : True si au moins un NA existe dans la ligne
# sum() compte combien de lignes contiennent au moins un NA
rows_with_any_na = df_flat_converted_unique.isna().any(axis=1).sum()
print("Nombre de lignes avec au moins un NA :", rows_with_any_na)

# Nombre de NA par colonne (tri décroissant)
print(df_flat_converted_unique.isna().sum().sort_values(ascending=False).head(20))

'''
Colonnes contenant au moins un NA :
['order_channel', 'order_discount']
Nombre de lignes avec au moins un NA : 193
order_channel           124
order_discount           71
customer_customer_id      0
event_time                0
event_id                  0
order_id                  0
order_device              0
customer_country          0
order_n_items             0
order_main_category       0
order_basket_value        0
order_shipping_fee        0
order_order_total         0
order_is_returned         0
dtype: int64
'''

# Fill NA catégoriels (mode ou valeur “unknown”)
df_flat_converted_unique_without_na = df_flat_converted_unique.copy()  # Copie pour remplir les NA
# Pour customer_country, si manquant => "UN" (Unknown)
# Creer la nouvelle categorie 
if "customer_country" in df_flat_converted_unique.columns:
    df_flat_converted_unique["customer_country"] = (
        df_flat_converted_unique["customer_country"]
        .cat.add_categories(["UN"])
        .fillna("UN")
    )

if "customer_country" in df_flat_converted_unique.columns:
    df_flat_converted_unique_without_na["customer_country"] = df_flat_converted_unique["customer_country"].fillna("UN")  # fillna remplace NA par "UN"

# Pour device/channel/category, si manquant => la mode (valeur la plus fréquente)
for col in ["order_device", "order_channel", "order_main_category"]:
    if col in df_flat_converted_unique.columns:
        mode_value = df_flat[col].mode()[0]  # mode() renvoie une Series, on prend la première valeur
        df_flat_converted_unique_without_na[col] = df_flat_converted_unique[col].fillna(mode_value)  # fillna remplace NA par la mode

# Fill NA numériques (stratégie simple et robuste)
# (1) shipping_fee et discount :
# Si absent, il est souvent raisonnable de considérer 0 (pas de frais / pas de remise)
for col in ["order_shipping_fee", "order_discount"]:
    if col in df_flat_converted_unique.columns:
        df_flat_converted_unique_without_na[col] = df_flat_converted_unique[col].fillna(0.0)  # remplace NA par 0.0 (float)

# (2) order_n_items :
# On remplace par la médiane (plus robuste que la moyenne si outliers)
if "order_n_items" in df_flat_converted_unique.columns:
    median_items = df_flat_converted_unique["order_n_items"].median()     # calcule la médiane
    df_flat_converted_unique_without_na["order_n_items"] = df_flat_converted_unique["order_n_items"].fillna(median_items)  # remplace NA par médiane
    df_flat_converted_unique_without_na["order_n_items"] = df_flat_converted_unique["order_n_items"].round().astype(int)   # arrondi puis cast en int

# (3) order_basket_value :
# Si absent, on remplace par la médiane car la moyenne peut être biaisée par des valeurs extrêmes
if "order_basket_value" in df_flat_converted_unique.columns:
    median_basket = df_flat_converted_unique["order_basket_value"].median()                  # calcule médiane
    df_flat_converted_unique_without_na["order_basket_value"] = df_flat_converted_unique["order_basket_value"].fillna(median_basket)  # remplace NA
    

################################
# Cohérence total = basket + shipping - discount
################################
# On calcule un total attendu à partir des composantes
# (basket_value + shipping_fee - discount)
df_flat_converted_unique_without_na["order_total_expected"] = (
    df_flat_converted_unique_without_na["order_basket_value"]          # valeur panier
    + df_flat_converted_unique_without_na["order_shipping_fee"]        # frais de livraison
    - df_flat_converted_unique_without_na["order_discount"]            # remise
)

# On calcule l'écart entre le total présent et le total attendu
df_flat_converted_unique_without_na["order_total_delta"] = (
    df_flat_converted_unique_without_na["order_order_total"] 
    - df_flat_converted_unique_without_na["order_total_expected"]
)
# On définit une tolérance d'arrondi (ex: 1 centime)
tolerance = 0.01

# On crée un flag : True si incohérence (écart > tolérance), False sinon
df_flat_converted_unique_without_na["order_total_inconsistent"] = df_flat_converted_unique_without_na["order_total_delta"].abs() > tolerance

# On compte combien de order_order_total sont manquants avant correction
missing_total_before = df_flat_converted_unique_without_na["order_order_total"].isna().sum()
print("order_order_total manquants (avant fill) :", missing_total_before)

# Si order_order_total est NA, on le remplace par le total attendu calculé
df_flat_converted_unique_without_na["order_order_total"] = df_flat_converted_unique_without_na["order_order_total"].fillna(df_flat_converted_unique_without_na["order_total_expected"])
# On affiche combien de lignes sont incohérentes
print("Nombre de lignes avec total incohérent :", df_flat_converted_unique_without_na["order_total_inconsistent"].sum())

# On montre quelques exemples incohérents pour investigation
print(df_flat_converted_unique_without_na[df_flat_converted_unique_without_na["order_total_inconsistent"]].head(20))

# ======================================================
# Apply / Lambda (exemples concrets)
# ======================================================
df_final =df_flat_converted_unique_without_na.copy()  # Copie pour transformations finales
# Extraire des features temporelles

# dt.year / dt.month / dt.day / dt.hour :
# fonctionne uniquement si event_time est bien en datetime
df_final["event_year"] = df_final["event_time"].dt.year      # extrait l'année
df_final["event_month"] = df_final["event_time"].dt.month    # extrait le mois
df_final["event_day"] = df_final["event_time"].dt.day        # extrait le jour
df_final["event_hour"] = df_final["event_time"].dt.hour      # extrait l'heure

# On affiche quelques lignes pour vérifier
print(df_final[["event_time", "event_year", "event_month", "event_day", "event_hour"]].head(10))
print(df_final.head(10))

########################################
# Apply + lambda sur lignes : prix moyen par article
#########################################
# On crée une nouvelle colonne "price_per_item"
# = order_basket_value / order_n_items
# On utilise apply + lambda pour appliquer la fonction ligne par ligne
df_final["price_per_item"] = df_final.apply(
    lambda row: row["order_basket_value"] / row["order_n_items"] if row["order_n_items"] > 0 else np.nan,
    axis=1  # axis=1 signifie qu'on applique la fonction sur les lignes
)
# On affiche quelques lignes pour vérifier
print(df_final[["order_basket_value", "order_n_items", "price_per_item"]].head(10))

#######################################
# Vérif NA finale + types + aperçu
#######################################
print("Vérification finale des NA :")
print(df_final.isna().sum())    # Affiche le nombre de NA par colonne
print("Types de données finaux :")
print(df_final.dtypes)          # Affiche les types de données finaux
print("Vérifie si event_id est unique (souvent attendu)")
print(df_final["event_id"].nunique() == df_final.shape[0])  # Vérifie si tous les event_id sont uniques
print("Aperçu final des données :")
print(df_final.head())          # Affiche les premières lignes du DataFrame final
print(f"Dimensions finales du DataFrame : {df_final.shape}")  # Affiche les dimensions finales du DataFrame

"""
Vérification finale des NA :
event_id                    0
order_id                    0
customer_customer_id        0
event_time                  0
customer_country            0
order_device                0
order_channel               0
order_main_category         0
order_n_items               0
order_basket_value          0
order_shipping_fee          0
order_discount              0
order_order_total           0
order_is_returned           0
order_total_expected        0
order_total_delta           0
order_total_inconsistent    0
event_year                  0
event_month                 0
event_day                   0
event_hour                  0
price_per_item              0
dtype: int64
Types de données finaux :
event_id                         string[python]
order_id                         string[python]
customer_customer_id             string[python]
event_time                  datetime64[ns, UTC]
customer_country                       category
order_device                           category
order_channel                          category
order_main_category                    category
order_n_items                             int64
order_basket_value                      float64
order_shipping_fee                      float64
order_discount                          float64
order_order_total                       float64
order_is_returned                          bool
order_total_expected                    float64
order_total_delta                       float64
order_total_inconsistent                   bool
event_year                                int32
event_month                               int32
event_day                                 int32
event_hour                                int32
price_per_item                          float64
dtype: object
Vérifie si event_id est unique (souvent attendu)
True
Aperçu final des données :
                                  event_id   order_id customer_customer_id  \
3246  924d78f0-8ead-4a1d-960a-444e1ce458fa  ORD-03149            CUST-0037   
1089  50bf179d-e4ed-4d20-b995-dfaabeb633fe  ORD-03305            CUST-0006   
3323  36df23d9-f1ec-433e-b539-24bce867f5a3  ORD-04082            CUST-0202   
3694  8c60222f-93a7-4cb8-b54f-cf073386ff1b  ORD-00440            CUST-0307   
3915  25eed853-9882-4652-9da8-fff6d25cc1fd  ORD-02192            CUST-0450   

                           event_time customer_country order_device  \
3246 2025-12-31 15:53:19.021865+00:00               IT       tablet   
1089 2025-12-31 15:57:19.033003+00:00               IT       mobile   
3323 2025-12-31 15:58:19.027022+00:00               DE       tablet   
3694 2025-12-31 16:06:18.971905+00:00               FR      desktop   
3915 2025-12-31 16:12:19.035292+00:00               DE       tablet   

     order_channel order_main_category  order_n_items  order_basket_value  \
3246     affiliate         electronics              8              498.90   
1089           seo             fashion              2              491.11   
3323           ads                home              8              287.46   
3694        direct             fashion              8               95.67   
3915     affiliate                home              5              364.16   

      ...  order_order_total  order_is_returned  order_total_expected  \
3246  ...             492.66              False                492.66   
1089  ...             491.11              False                491.11   
3323  ...             287.46              False                287.46   
3694  ...              95.67               True                 95.67   
3915  ...             364.16              False                364.16   

      order_total_delta  order_total_inconsistent  event_year  event_month  \
3246       5.684342e-14                     False        2025           12   
1089       0.000000e+00                     False        2025           12   
3323       0.000000e+00                     False        2025           12   
3694       0.000000e+00                     False        2025           12   
3915       0.000000e+00                     False        2025           12   

      event_day  event_hour  price_per_item  
3246         31          15        62.36250  
1089         31          15       245.55500  
3323         31          15        35.93250  
3694         31          16        11.95875  
3915         31          16        72.83200  

[5 rows x 22 columns]
Dimensions finales du DataFrame : (4000, 22)
"""

############################
# Export du DataFrame final nettoyé (CSV et Parquet) dans PROCESSED_DIR
#############################
# Chemin du fichier CSV
processed_csv_path = PROCESSED_DIR / "orders_events_cleaned.csv"
# Export en CSV (index=False pour ne pas inclure l'index pandas)
df_final.to_csv(processed_csv_path, index=False)
print(f"DataFrame final exporté en CSV vers : {processed_csv_path}")
# Chemin du fichier Parquet
processed_parquet_path = PROCESSED_DIR / "orders_events_cleaned.parquet"
# Export en Parquet (index=False pour ne pas inclure l'index pandas)
df_final.to_parquet(processed_parquet_path, index=False)
print(f"DataFrame final exporté en Parquet vers : {processed_parquet_path}")

###########################
# Imports Matplotlib + réglages d’affichage
###########################

# On importe matplotlib.pyplot : c'est le module principal pour tracer des graphiques
import matplotlib.pyplot as plt

# On importe matplotlib.dates pour mieux formater les dates sur l'axe X (utile en time series)
import matplotlib.dates as mdates

# On affiche les graphiques "dans" le notebook (si tu es sur Jupyter)
# %matplotlib inline

# On définit une taille par défaut des figures (évite d'avoir des plots trop petits)
# plt.rcParams["figure.figsize"] = (10, 5)

# On définit une résolution plus propre (utile si tu exportes des images)
# plt.rcParams["figure.dpi"] = 120

# On liste les colonnes numériques (int/float) automatiquement
num_cols = df_final.select_dtypes(include=["number"]).columns

# On affiche les stats descriptives des colonnes numériques
# describe() calcule count, mean, std, min, quartiles, max
print(df_final[num_cols].describe().T)

# On liste les colonnes catégorielles (type category)
cat_cols = df_final.select_dtypes(include=["category"]).columns

# On affiche les colonnes catégorielles détectées
print("Colonnes catégorielles :", list(cat_cols))

# On calcule le taux global de retour (bool -> mean : True=1, False=0)
# mean() sur bool donne directement une proportion
return_rate = df_final["order_is_returned"].mean()

# On affiche le taux global de retour en %
print(f"Taux global de retour : {return_rate*100:.2f}%")

