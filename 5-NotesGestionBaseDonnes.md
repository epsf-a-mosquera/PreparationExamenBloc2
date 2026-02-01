# checklist très complète (prête à copier/coller en cellules Jupyter) pour explorer et modifier une base MySQL via SQLAlchemy
## 0) Cellule “connexion” (Engine + test)
```python
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

# ⚠️ Adapte le driver selon ton install :
# - mysql+pymysql:// ...
# - mysql+mysqlconnector:// ...
DB_URL = "mysql+pymysql://USER:PASSWORD@HOST:3306/DBNAME?charset=utf8mb4"

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,   # évite les connexions mortes
    future=True,          # style SQLAlchemy 2.0
)

# Test rapide
with engine.connect() as conn:
    print(conn.execute(text("SELECT VERSION()")).scalar())
```

## 1) Explorer la structure avec SQLAlchemy Inspector (tables, colonnes, PK/FK, index…)

1.1 Cellule “inspecteur” + helpers

```python
from sqlalchemy import inspect

insp = inspect(engine)

def show_tables(schema=None):
    return insp.get_table_names(schema=schema)

def show_views(schema=None):
    return insp.get_view_names(schema=schema)

def show_columns(table, schema=None):
    return insp.get_columns(table, schema=schema)

def show_pk(table, schema=None):
    return insp.get_pk_constraint(table, schema=schema)

def show_fks(table, schema=None):
    return insp.get_foreign_keys(table, schema=schema)

def show_indexes(table, schema=None):
    return insp.get_indexes(table, schema=schema)

def show_unique_constraints(table, schema=None):
    return insp.get_unique_constraints(table, schema=schema)

def show_check_constraints(table, schema=None):
    return insp.get_check_constraints(table, schema=schema)

def show_table_comment(table, schema=None):
    return insp.get_table_comment(table, schema=schema)

def show_table_options(table, schema=None):
    return insp.get_table_options(table, schema=schema)
```

## 1.2 Lister schémas / tables / vues

```python
insp.get_schema_names()      # schémas (souvent ['information_schema', 'mysql', ..., 'DBNAME'])
show_tables()                # tables du schéma par défaut (DBNAME)
show_views()                 # vues
```

## 1.3 Détails d’une table
```python
table = show_tables()[0]     # exemple: première table

show_columns(table)
show_pk(table)
show_fks(table)
show_indexes(table)
show_unique_constraints(table)
show_check_constraints(table)
show_table_comment(table)
show_table_options(table)
```

## 2) Reflection (charger un schéma existant en objets SQLAlchemy)

Pratique pour manipuler sans avoir les classes ORM sous la main.

```python
from sqlalchemy import MetaData, Table

metadata = MetaData()
metadata.reflect(bind=engine)          # charge toutes les tables
list(metadata.tables.keys())           # noms complets: "table"

t = metadata.tables[table]             # objet Table
t.columns.keys()                       # colonnes
```

## 3) Lire les données (SELECT) depuis un notebook
### 3.1 Lecture rapide en “dict” (SQL brut)
```python
from sqlalchemy import text

with engine.connect() as conn:
    rows = conn.execute(text(f"SELECT * FROM `{table}` LIMIT 10")).mappings().all()
rows
```

### 3.2 Lecture en pandas (super pratique pour explorer)
```python
import pandas as pd

df = pd.read_sql(text(f"SELECT * FROM `{table}` LIMIT 50"), engine)
df
```

### 3.3 Compter, vérifier valeurs uniques, NULL, etc.
```python
with engine.connect() as conn:
    total = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`")).scalar()
total

col = show_columns(table)[0]["name"]
with engine.connect() as conn:
    stats = conn.execute(text(f"""
        SELECT
          SUM(CASE WHEN `{col}` IS NULL THEN 1 ELSE 0 END) AS nulls,
          COUNT(DISTINCT `{col}`) AS distinct_count
        FROM `{table}`
    """)).mappings().one()
stats
```

## 4) Insérer / mettre à jour / supprimer (DML)

⚠️ Avec MySQL, pense à commit (via Session ou conn.commit()).

### 4.1 INSERT (SQL brut)
```python
with engine.begin() as conn:  # begin() => commit auto si tout OK
    conn.execute(
        text(f"INSERT INTO `{table}` (col1, col2) VALUES (:c1, :c2)"),
        {"c1": "value1", "c2": 123},
    )
```

### 4.2 INSERT multiple
```python
rows_to_insert = [
    {"c1": "a", "c2": 1},
    {"c1": "b", "c2": 2},
]

with engine.begin() as conn:
    conn.execute(
        text(f"INSERT INTO `{table}` (col1, col2) VALUES (:c1, :c2)"),
        rows_to_insert,
    )
```

### 4.3 UPDATE
```python
with engine.begin() as conn:
    conn.execute(
        text(f"UPDATE `{table}` SET col2 = :v WHERE col1 = :k"),
        {"v": 999, "k": "value1"},
    )
```
### 4.4 DELETE
```python
with engine.begin() as conn:
    conn.execute(
        text(f"DELETE FROM `{table}` WHERE col1 = :k"),
        {"k": "value1"},
    )
```
## 5) Ajouter une colonne / index / contrainte (DDL)

⚠️ Le DDL est “destructeur” si tu te trompes. Le mieux en vrai projet : Alembic (migrations).
Mais dans un notebook, tu peux le faire comme ci-dessous.

### 5.1 Ajouter une colonne
```python
new_col_name = "new_col"
new_col_def  = "VARCHAR(255) NULL"   # exemple

with engine.begin() as conn:
    conn.execute(text(f"ALTER TABLE `{table}` ADD COLUMN `{new_col_name}` {new_col_def}"))
```

### 5.2 Ajouter une colonne avec valeur par défaut
```python
with engine.begin() as conn:
    conn.execute(text(f"ALTER TABLE `{table}` ADD COLUMN `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP`"))
```

### 5.3 Ajouter un index
```python
with engine.begin() as conn:
    conn.execute(text(f"CREATE INDEX idx_{table}_col1 ON `{table}` (col1)"))
```

### 5.4 Ajouter une contrainte UNIQUE
```python
with engine.begin() as conn:
    conn.execute(text(f"ALTER TABLE `{table}` ADD CONSTRAINT uq_{table}_col1 UNIQUE (col1)"))
```

### 5.5 Ajouter une clé étrangère (FK)
```python
# Exemple: table.colA -> other_table.id
with engine.begin() as conn:
    conn.execute(text(f"""
        ALTER TABLE `{table}`
        ADD CONSTRAINT fk_{table}_colA
        FOREIGN KEY (colA) REFERENCES other_table(id)
        ON DELETE RESTRICT ON UPDATE CASCADE
    """))
```
## 6) Modifier une colonne (type, NULL/NOT NULL, rename)
### 6.1 Modifier le type / nullabilité (MySQL: MODIFY)
```python
# Exemple: col2 devient BIGINT NOT NULL
with engine.begin() as conn:
    conn.execute(text(f"ALTER TABLE `{table}` MODIFY COLUMN col2 BIGINT NOT NULL"))
```
### 6.2 Renommer une colonne

MySQL 8 permet RENAME COLUMN :
```python
with engine.begin() as conn:
    conn.execute(text(f"ALTER TABLE `{table}` RENAME COLUMN old_name TO new_name"))
```

Si ta version ne supporte pas, il faut CHANGE COLUMN avec le type complet :
```python
with engine.begin() as conn:
    conn.execute(text(f"ALTER TABLE `{table}` CHANGE COLUMN old_name new_name VARCHAR(255) NULL"))
```
### 6.3 Supprimer une colonne
```python
with engine.begin() as conn:
    conn.execute(text(f"ALTER TABLE `{table}` DROP COLUMN `new_col`"))
```
## 7) Voir tout ce que MySQL sait (information_schema)

Quand tu veux être ultra exhaustif (tables, colonnes, triggers, procédures…), information_schema est la source.

### 7.1 Tables et lignes estimées
```python
db = "DBNAME"
q = """
SELECT TABLE_NAME, TABLE_TYPE, ENGINE, TABLE_ROWS
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = :db
ORDER BY TABLE_NAME
pd.read_sql(text(q), engine, params={"db": db})
```

### 7.2 Colonnes (type, default, null…)
```python
q = """
SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_DEFAULT, EXTRA
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = :db
ORDER BY TABLE_NAME, ORDINAL_POSITION
"""
pd.read_sql(text(q), engine, params={"db": db})
```
### 7.3 Contraintes (PK/UNIQUE/FK)
```python
q = """
SELECT tc.TABLE_NAME, tc.CONSTRAINT_NAME, tc.CONSTRAINT_TYPE
FROM information_schema.TABLE_CONSTRAINTS tc
WHERE tc.TABLE_SCHEMA = :db
ORDER BY tc.TABLE_NAME, tc.CONSTRAINT_TYPE, tc.CONSTRAINT_NAME
"""
pd.read_sql(text(q), engine, params={"db": db})
```
### 7.4 Détails des clés étrangères
```python
q = """
SELECT
  kcu.TABLE_NAME,
  kcu.CONSTRAINT_NAME,
  kcu.COLUMN_NAME,
  kcu.REFERENCED_TABLE_NAME,
  kcu.REFERENCED_COLUMN_NAME
FROM information_schema.KEY_COLUMN_USAGE kcu
WHERE kcu.TABLE_SCHEMA = :db AND kcu.REFERENCED_TABLE_NAME IS NOT NULL
ORDER BY kcu.TABLE_NAME, kcu.CONSTRAINT_NAME, kcu.ORDINAL_POSITION
"""
pd.read_sql(text(q), engine, params={"db": db})
```

### 7.5 Index
```python
q = """
SELECT TABLE_NAME, INDEX_NAME, NON_UNIQUE, SEQ_IN_INDEX, COLUMN_NAME, INDEX_TYPE
FROM information_schema.STATISTICS
WHERE TABLE_SCHEMA = :db
ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX
"""
pd.read_sql(text(q), engine, params={"db": db})
```
### 7.6 Vues
```python
q = """
SELECT TABLE_NAME
FROM information_schema.VIEWS
WHERE TABLE_SCHEMA = :db
ORDER BY TABLE_NAME
"""
pd.read_sql(text(q), engine, params={"db": db})
```
### 7.7 Triggers
```python
q = """
SELECT TRIGGER_NAME, EVENT_MANIPULATION, EVENT_OBJECT_TABLE, ACTION_TIMING
FROM information_schema.TRIGGERS
WHERE TRIGGER_SCHEMA = :db
ORDER BY TRIGGER_NAME
"""
pd.read_sql(text(q), engine, params={"db": db})
```
### 7.8 Procédures / fonctions
```python
q = """
SELECT ROUTINE_NAME, ROUTINE_TYPE
FROM information_schema.ROUTINES
WHERE ROUTINE_SCHEMA = :db
ORDER BY ROUTINE_TYPE, ROUTINE_NAME
"""
pd.read_sql(text(q), engine, params={"db": db})
```
## 8) Afficher le SQL de création (DDL “SHOW CREATE …”)

Très utile pour copier la structure exacte.
```python
with engine.connect() as conn:
    ddl = conn.execute(text(f"SHOW CREATE TABLE `{table}`")).mappings().one()
ddl
```

Pour une vue :
```python
view = show_views()[0]
with engine.connect() as conn:
    ddl = conn.execute(text(f"SHOW CREATE VIEW `{view}`")).mappings().one()
ddl
```

## 9) Transactions et “rollback” (pour tester sans casser)
```python
from sqlalchemy.orm import Session
from sqlalchemy import text

with Session(engine) as session:
    try:
        session.execute(text(f"UPDATE `{table}` SET col2=col2+1 WHERE 1=0"))  # exemple
        # session.commit()  # commente pour ne rien appliquer
        session.rollback()   # annule
    except:
        session.rollback()
        raise
```
## 10) Bonus : cellule “explorer une table” automatiquement (profil rapide)
``` python
import pandas as pd
from sqlalchemy import text

def profile_table(table, limit=10, schema=None):
    cols = show_columns(table, schema=schema)
    print("TABLE:", table)
    print("COLUMNS:", [c["name"] for c in cols])
    print("PK:", show_pk(table, schema=schema))
    print("FKS:", show_fks(table, schema=schema))
    print("INDEXES:", show_indexes(table, schema=schema))
    with engine.connect() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`")).scalar()
    print("ROWS:", n)
    return pd.read_sql(text(f"SELECT * FROM `{table}` LIMIT {limit}"), engine)

profile_table(show_tables()[0], limit=20)
```
Remarque importante (bonne pratique)

Pour ajouter des colonnes / modifier des types en production, utilise plutôt Alembic (migrations) pour garder l’historique et éviter les surprises. Mais pour explorer/tester dans un notebook, les commandes ci-dessus sont celles que tu utiliseras tout le temps.