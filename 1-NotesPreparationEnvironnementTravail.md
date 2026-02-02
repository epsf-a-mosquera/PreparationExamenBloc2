# Notes de préparation pour l'examen Anderson Fabian MOSQUERA VARELA

# Plan “4 heures” (timing réaliste)
0:00 – 0:15
Lire le sujet, lister livrables, créer l’arborescence, lancer docker compose.
0:15 – 1:05
Notebook exploration JSON + décisions (schéma, cleaning, cible ML).
1:05 – 2:05
Scripts ETL + ORM + ingestion DB. Vérif rapide dans phpMyAdmin.
2:05 – 2:50
ML training + métriques + joblib.
2:50 – 3:25
Kafka producer/consumer “simple” + démonstration (quelques messages).
3:25 – 3:45
Tests pytest + robustesse ingestion.
3:45 – 4:00
README synthèse + zip final + dernier check.

# 1. préparer les dossier de travails et les fichies à générer.
selon l'information donnée pour la préparation de l'examen une archicteture global du dossier du travail peut être

```powershell
ExamenBloc2/
  README.md
  synthese.md
  requirements.txt
  docker-compose.yml
  .env
  notebooks/
    exploration.ipynb
  src/
    __init__.py
    config.py
    extract_transform.py
    db_models.py
    ingest.py
    train.py
    kafka_producer.py
    kafka_consumer.py
    utils.py
  tests/
    test_ingestion.py
  data/            # s'il y a des fichiers data
    raw            # Dossier où on met le JSON brut
    processed      # Dossier où on met les données nettoyées
    features       # features prêtes pour le ML

  artifacts/       # Exemples : le modèle entraîné (model.pkl), un scaler (scaler.pkl), un encoder (onehot_encoder.pkl), un label encoder, des métriques (metrics.json), des hyperparamètres, la version du modèle, des logs d’entraînement, etc.
  outputs/         # figures, rapports, logs
```

Note : 
```bash
Grâce à __init__.py dans un dossier, par exemple src, Python autorise :
from src.config import RAW_JSONL_PATH
Sans __init__.py ❌ :
ModuleNotFoundError: No module named 'src'

Pour lancer depuis bash par exemple le script generate_data qui contient la ligne de code from src.config import RAW_JSONL_PATH
il est necessaire
python3 -m src.generate_data
✔️ Python comprend :
src est un package
generate_data est un module du package src
les imports from src.config import ... fonctionnent
```

Notes : 
```bash
# 1️⃣ Créer un dossier
mkdir mon_dossier
# 2️⃣ Se déplacer dans un dossier
cd mon_dossier
# 3️⃣ Créer un fichier vide
touch fichier.txt
# 4️⃣ Créer un fichier avec du texte
echo "Bonjour" > fichier.txt
# 5️⃣ Ajouter du texte à la fin d'un fichier
echo "Nouvelle ligne" >> fichier.txt
# 6️⃣ Afficher le contenu d'un fichier
cat fichier.txt
# 7️⃣ Lister les fichiers d'un dossier
ls -l
# 8️⃣ Copier un fichier
cp fichier.txt copie.txt
# 9️⃣ Déplacer ou renommer un fichier
mv fichier.txt nouveau_nom.txt
# 🔟 Supprimer un fichier ou un dossier
rm nouveau_nom.txt       # fichier
rm -r mon_dossier        # dossier et son contenu
```

# 2. Préparer l'environnement virtuelle avec les dependances qui seron à utiliser  pendant l'examen
les commandes à savoir pour ce partie sont : 
```bash
# Préparer le fichier requierements.txt
# Partir du fichier requierements.txt fait pendant la préparation de l'examen
```

```bash
# 1️⃣ Mettre à jour le gestionnaire de paquets (optionnel, utile sur Linux)
sudo apt update
# 2️⃣ Créer l'environnement virtuel Python nommé ".venv"
python3 -m venv .venv
# 3️⃣ Activer l'environnement virtuel
source .venv/bin/activate
# 4️⃣ Installer toutes les dépendances listées dans requirements.txt
pip install --upgrade pip  # Mettre pip à jour avant installation
pip install -r requirements.txt
#pour désinstaller les paquets si nécessaire une exemple
pip uninstall -y Kafka kafka kafka-python dotenv
# 5️⃣ Vérifier que les paquets sont bien installés
pip list
# 6️⃣ Désactiver l'environnement virtuel si nécessaire
deactivate
# 7️⃣ Pour creer un nouveau fichier requirements.txt après avoir installé des paquets
pip freeze > requirements.txt
```
```python
#Pour lancer des scripts : Exemple
python3 src/generate_data.py

#si le script est dans un dossier avec un fichier __init__.py, par exemple src avec ce fichier de dans, il est possible de faire
python3 -m src.generate_data (à faire si on a des imports from src.config import ... dans generate_data.py)
```

# 4. Préparer le fichier .env qui va contenir les variables d'environnement qui vont être utilisés par les scripts et par les fichier Docker .env
une proposition à créer unitioalment avec les variables que nous savons déjà qu'allons utiliser dasn le docker-compose-yml sont :

```
# voir le .env
```

# 4. Préparer le template du fichier docker-compose.yml avec les services nécessaires pour l’évaluation.
ici un possible fichier docker-compose.yml avec les informations données pour préparer l'évalution
```yaml
# voir le docker-compose.yml
```

Pour lancer les services docker les commandes sont les suivantes
```bash
# -------------------------------------------------------------
# 1️⃣ Créer les images (si nécessaire) et démarrer les services en arrière-plan
# -------------------------------------------------------------
docker compose up -d
# - "up" : construit les images si elles n'existent pas et démarre les conteneurs
# - "-d" : mode détaché (les conteneurs tournent en arrière-plan)
# -------------------------------------------------------------
# 2️⃣ Voir les conteneurs Docker en cours d'exécution
# -------------------------------------------------------------
docker ps
# Affiche la liste des conteneurs actifs avec leurs ports, noms, et statuts
# -------------------------------------------------------------
# 3️⃣ Stopper un conteneur spécifique
# -------------------------------------------------------------
docker stop <nom_du_conteneur>
# Exemple : docker stop exam-mysql
# -------------------------------------------------------------
# 4️⃣ Stopper tous les conteneurs
# -------------------------------------------------------------
docker stop $(docker ps -q)
# "$(docker ps -q)" récupère tous les IDs des conteneurs en cours
# -------------------------------------------------------------
# 5️⃣ Supprimer un conteneur spécifique
# -------------------------------------------------------------
docker rm <nom_du_conteneur>
# Exemple : docker rm exam-mysql
# Attention : le conteneur doit être arrêté avant de le supprimer
# -------------------------------------------------------------
# 6️⃣ Supprimer tous les conteneurs
# -------------------------------------------------------------
docker rm $(docker ps -a -q)
# "$(docker ps -a -q)" récupère tous les IDs des conteneurs, même arrêtés
# -------------------------------------------------------------
# 7️⃣ Supprimer tous les volumes Docker
# -------------------------------------------------------------
docker volume rm $(docker volume ls -q)
# Attention : supprime toutes les données persistantes
# -------------------------------------------------------------
# 8️⃣ Redémarrer les services (stop + up)
# -------------------------------------------------------------
docker compose down        # Arrête et supprime les conteneurs du compose
docker compose up -d       # Redémarre les services
# -------------------------------------------------------------
# 9️⃣ Afficher les logs d’un service en temps réel
# -------------------------------------------------------------
docker compose logs -f <nom_service>
# Exemple : docker compose logs -f mysql
# "-f" = follow, pour voir les logs en continu
# -------------------------------------------------------------
# 10️ Inspecter l’état détaillé d’un conteneur
# -------------------------------------------------------------
docker inspect <nom_du_conteneur>
# Affiche toutes les informations du conteneur (réseau, volumes, configuration)
# -------------------------------------------------------------
# 11 supprimer tous les conteneurs, images et volumes
# -------------------------------------------------------------
docker system prune -a --volumes
# Attention : supprime toutes les données persistantes et images non utilisées
# -------------------------------------------------------------
# verification que les ports sont bien exposés
docker ps --format "table {{.Names}}\t{{.Ports}}"

# si il n'est pas possible d'acceder aux services despuis le navigateur, utiliser un tunnel SSH avc la commande suivante
ssh -L 8080:localhost:8080 -L 8081:localhost:8081 ubuntu@<IP_VM>
# Remplacer <IP_VM> par l'adresse IP de la VM
# donc si par exemple ubuntu@ip-172-31-40-176 --> pour touver l'ip publique de la vm 
curl ifconfig.me
# exemple de resultat 34.251.2.21

#depuis la machine virtuelle 
ssh -i ~/keys/data_enginering_machine.pem \
  -L 8080:localhost:8080 \
  -L 8081:localhost:8081 \
  ubuntu@34.251.2.21

#à lancer de power shelle del computador
ssh -i "C:\Users\a.mosquera\Dropbox\FormationDataEngineer\data_enginering_machine.pem" `
  -L 18080:127.0.0.1:8080 `
  -L 18081:127.0.0.1:8081 `
  ubuntu@34.251.2.21


ssh -i ~/keys/data_enginering_machine.pem \
  -L 8080:localhost:8080 \
  -L 8081:localhost:8081 \
  ubuntu@34.251.2.21

ssh -i examen-bloc2.pem \
  -L 8080:localhost:8080 \
  -L 8081:localhost:8081 \
  ubuntu@IP_PUBLIQUE

# pour éviter le problème de permission denied for key
chmod 600 ~/ExamenBloc2/keys/data_enginering_machine.pem

```

# installer dnas la VM un navigateur web léger pour pouvoir accéder aux interfaces web des services docker
```bash
sudo apt update
sudo apt install -y firefox
```
# pour lancer firefox depuis la VM
```bash
firefox &
```
# pour verifier les ports exposés par les conteneurs docker
```bash
docker ps --format "table {{.Names}}\t{{.Ports}}"
```

# Pour accéder aux interfaces depuis le navigateur
phpMyAdmin (interface web pour MySQL) → http://localhost:8080
Kafka UI (interface web pour Kafka) → http://localhost:8081

# 5. Préparer le fichier config.py, à partir duquel les script vont lire les variables necessaire pour travailler.
```yaml
# Voir le fichier de config.py de base préparé préalablement
```

# 6. Faire l'analyse de données JSON dans le notebook
Il est nécessaire de s’assurer, avant tout, que le notebook s’exécute correctement avec le kernel de l’environnement virtuel
Avec l'environnement virtuel active, il faut creer le kernel de celui-ci

```bash
#activer l'environnement virtuel 
source .venv/bin/activate

#Para saber con cual python se esta trabajando
which python

# installer ipykernel que normalement est déjà installé à partir du fichier requirements.txt
pip install ipykernel
pip install --upgrade ipykernel
pip install --upgrade jupyter


#Créer un kernel Jupyter lié à l’environnement virtuel
python3 -m ipykernel install --user \
    --name ExamenBloc2 \
    --display-name "Python (.venv ExamenBloc2)"

#voir su le ipykerbel est bien créé
jupyter kernelspec list

#Pour ouvrir la terminar dans le navigateur
jupyter notebook

#supprimer les kernels inutils
jupyter kernelspec remove examenbloc2
Une page web s’ouvre automatiquement (souvent : http://localhost:8888)

#Vérifie avec une cellule pour assurer que j'ai les biblioteques qui sont dans l'enviroment virtuel:
import sys
print(sys.executable)
#Resultat attendu
/home/ubuntu/ExamenBloc2/.venv/bin/python

```
Une fois ce partie realise, il est possible de créer le fichier .ipynb (exploration.ipynb) et selectionner le kernel qu'on vient de créer. 
on commence l'analyse des données JSON et la prise de décisions pour la suite.


# Information sur git 

📌 Information sur Git
1️⃣ Initialiser un dépôt local
Si tu commences un projet depuis zéro :
# Crée un fichier README
echo "# PreparationExamenBloc2" >> README.md
# Initialise le dépôt Git
git init
# Ajouter tous les fichiers du projet au suivi
git add .
# Faire le premier commit
git commit -m "first commit"
# Renommer la branche principale en 'main'
git branch -M main

2️⃣ Connecter le dépôt local à un dépôt distant
# Ajouter l'URL du dépôt distant
git remote add origin https://github.com/epsf-a-mosquera/PreparationExamenBloc2.git
# Envoyer les commits locaux vers le dépôt distant (premier push)
git push -u origin main
Après le premier push, tu pourras utiliser simplement git push pour les futurs commits.
#forcer le push (si nécessaire)
git push origin nom_de_la_branche --force
git push origin main --force

3️⃣ Ajouter/modifier des fichiers et pousser les changements
# Ajouter un nouveau fichier ou les modifications
git add fichier_ou_dossier
# Faire un commit
git commit -m "Message décrivant les changements"
# Envoyer les changements au dépôt distant
git push

4️⃣ Récupérer un dépôt distant sur ta machine (VM ou autre)
Si tu veux travailler sur la VM et récupérer le projet distant :
# Cloner le dépôt distant
git clone https://github.com/epsf-a-mosquera/PreparationExamenBloc2.git
# Se déplacer dans le dossier cloné
cd PreparationExamenBloc2
git clone crée un dossier local avec tous les fichiers et l’historique Git.

5️⃣ Mettre à jour ton dépôt local avec les changements du dépôt distant
# Récupérer les changements depuis le dépôt distant
git fetch
# Fusionner les changements dans la branche courante
git merge origin/main
Ou plus simple (commande courante) :
git pull
git pull = git fetch + git merge

6️⃣ Vérifier l’état du dépôt
# Voir les fichiers modifiés/non suivis
git status
# Voir l’historique des commits
git log --oneline --graph --all

7️⃣ Supprimer des fichiers ou dossiers du suivi
# Retirer un fichier du suivi Git mais le garder sur le disque
git rm --cached fichier.txt
# Retirer un dossier entier du suivi
git rm -r --cached dossier/
Pratique pour .venv/ ou data/ après avoir ajouté un .gitignore.

