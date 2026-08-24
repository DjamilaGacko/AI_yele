# Yélé — Module d'intelligence artificielle

Détection d'anomalies réseau et prévision de la qualité de service, à partir des
mesures collectées par l'application mobile Yélé.

Le module comporte deux volets :

- **La recherche** — notebooks et scripts d'exploration, d'entraînement et
  d'évaluation comparée des modèles.
- **Le service** — une API FastAPI (`service.py`) déployée en production et
  interrogée par le tableau de bord web.

---

## Le projet Yélé en un coup d'œil

| Dépôt | Rôle | Techno |
|---|---|---|
| `mobilefront` | Application mobile : réalise les mesures sur le terrain | Flutter / Dart |
| `mobiletest` | Backend : speedtest, collecte, API | Go + MongoDB |
| `new_web` | Tableau de bord public | HTML/CSS/JS |
| **`ai`** *(ce dépôt)* | Détection d'anomalies et prévision | Python / FastAPI |

```
  [ MongoDB Atlas ]
        ^
        | lit les mesures directement (sans les champs personnels)
        |
   [ CE DÉPÔT ]  service FastAPI, port 8000
        ^
        | proxifié sous /api/ai/*
        |
  [ Backend Go ]  <---  [ Tableau de bord web ]
```

Le service ne reçoit jamais de requête directe du navigateur : il est toujours
relayé par le backend Go.

---

## Prérequis

- **Python >= 3.12**
- Accès en lecture à la base **MongoDB** du projet

## Démarrage rapide

```bash
git clone <url-du-depot>
cd ai

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt

cp .env.example .env            # puis renseigner la chaîne de connexion
```

Lancer le service :

```bash
uvicorn service:app --reload --port 8000
```

Vérifier qu'il répond :

```bash
curl http://localhost:8000/ai/health
```

## Configuration

Les secrets se placent dans un fichier **`.env`**, ignoré par git.
**Ne jamais committer la chaîne de connexion.**

```
SPEEDTEST_DATABASE_CONNECTION_STRING="mongodb+srv://USER:PASS@CLUSTER.mongodb.net/"
SPEEDTEST_DATABASE_NAME="yele_speedtest"
```

Deux fichiers de dépendances, volontairement distincts :

| Fichier | Usage |
|---|---|
| `requirements.txt` | Travail local : inclut Jupyter, matplotlib, outils d'évaluation |
| `requirements-service.txt` | Production : strictement le nécessaire au service |

---

## Les points de terminaison

| Route | Renvoie |
|---|---|
| `GET /ai/health` | État du service, volume de données, opérateurs connus |
| `GET /ai/anomalies?days=30&limit=100` | Anomalies détectées, avec sévérité et explications |
| `GET /ai/forecast?operator=X&hours=48` | Débit descendant prédit, heure par heure |
| `POST /ai/retrain` | Ré-entraîne les modèles sur les données actuelles |

Consommés par le tableau de bord web sous le préfixe `/api/ai/*`.

### Comportement au démarrage

L'entraînement initial se fait **dans un fil d'exécution séparé** : le service
répond immédiatement, sans attendre. Tant que l'entraînement n'est pas terminé,
`/ai/health` renvoie `status: "training"`.

Si MongoDB est injoignable, le service démarre quand même et signale l'erreur
dans `/ai/health` plutôt que de refuser de se lancer. C'est délibéré : une base
indisponible ne doit pas empêcher le diagnostic.

---

## Structure du dépôt

### Le service (déployé)

```
service.py                  API FastAPI : les quatre routes ci-dessus
model.py                    *** Cœur du module *** classe YeleAI
Dockerfile                  Image de production
requirements-service.txt    Dépendances minimales
```

### La recherche (local)

```
prepare_data.py             Extraction MongoDB + nettoyage -> dataset_clean.csv
train.py                    Entraînement + rapport de qualité
evaluate.py                 Évaluation complète, toutes métriques
pyod_detectors.py           Comparaison de détecteurs d'anomalies
comparaison_approches.py    Comparaison des approches essayées
01_nettoyage.ipynb          Notebook : nettoyage pas à pas
02_entrainement.ipynb       Notebook : entraînement et résultats
03_detection_pyod.ipynb     Notebook : détection d'anomalies
explore.ipynb               Notebook : exploration initiale
```

Ordre de lecture conseillé pour comprendre la démarche :
`explore.ipynb` → `01_nettoyage.ipynb` → `02_entrainement.ipynb` →
`03_detection_pyod.ipynb`, puis `model.py`.

### Ce qui n'est pas versionné

Le `.gitignore` exclut délibérément :

- `.env` — secrets ;
- `*.csv`, `*.json` — données extraites, susceptibles de contenir des
  informations sensibles ;
- `*.joblib` — modèles entraînés (le fichier fait environ 44 Mo) ;
- `dataset/` — données externes volumineuses.

Un dépôt fraîchement cloné ne contient donc **ni données ni modèle** : il faut
lancer `python prepare_data.py` puis `python train.py` pour les régénérer.

---

## Comprendre la démarche

### Deux modèles, deux familles de métriques

Le module répond à deux questions distinctes, qu'il ne faut pas confondre :

| | Détection d'anomalies | Prévision du débit |
|---|---|---|
| Nature | Classification non supervisée | Régression |
| Modèle retenu | **ECOD** (PyOD) | **Gradient Boosting** (scikit-learn) |
| Comparé à | Isolation Forest, COPOD, LOF | TabPFN |
| Métriques | Taux de détection, cohérence | MAE, RMSE, R², *skill score* |

Il n'existe ni F1 ni AUC pour une régression : ces métriques mesurent une
classification. L'équivalent du « bon / pas bon » pour la prévision est le
*skill score*, qui compare le modèle à une référence naïve.

Le script `comparaison_approches.py` présente ces comparaisons côte à côte et
documente les choix retenus.

### Dégradation progressive selon le volume de données

Le projet part de très peu de mesures. Les modèles s'adaptent au volume
disponible plutôt que d'échouer :

- **En dessous de `MIN_SAMPLES_ML` (50 mesures)** : seules les méthodes
  statistiques fonctionnent — médiane et IQR par opérateur. `ml_ready` vaut
  `False`, et le tableau de bord l'indique.
- **Au-delà** : les modèles d'apprentissage automatique prennent le relais.

C'est une contrainte réelle d'un projet naissant, pas une limite temporaire à
supprimer.

### Un choix statistique à connaître

Le seuil bas de débit utilise le **percentile 5**, et non la borne classique
`Q1 - 1,5 x IQR`. Cette dernière suppose une distribution symétrique ; sur des
débits réels, fortement asymétriques, elle devenait négative — donc inutilisable.
Le P05 reste toujours une valeur de débit plausible.

### Deux pièges dans les données

**1. `operator` n'est pas l'opérateur mobile.** C'est le fournisseur d'accès
détecté par adresse IP. Pour un test réalisé en WiFi, il vaut « ANPTIC », qui
n'est pas un opérateur mobile. Pour analyser le réseau **mobile**, filtrer
`df[~df['is_wifi']]` et utiliser `cellular_tech` et `sim_operator`.

**2. Les libellés d'opérateur sont en doublon.** « ONATEL (… » et
« ONATEL (…, PTT) » désignent la même entreprise. La table `OPERATOR_ALIASES`
de `model.py` les fusionne. Sans cette normalisation, les statistiques par
opérateur sont fausses. Toute nouvelle variante d'écriture doit y être ajoutée.

### Protection des données personnelles

Les champs personnels (`ip_address`, `isp_info`, `log`, `extra_raw`,
`user_agent`, `language`) sont **exclus dès la requête MongoDB**, via la
projection `PII_PROJECTION` de `model.py`. Ils n'entrent jamais dans un
DataFrame, ne sont jamais écrits dans un CSV et ne peuvent donc pas fuiter par
un notebook partagé.

Cette règle doit être préservée : ne pas retirer de champ de `PII_PROJECTION`.

---

## Utilisation en recherche

```bash
python prepare_data.py    # 1. Extraire et nettoyer -> dataset_clean.csv
python train.py           # 2. Entraîner + rapport de qualité
python evaluate.py        # 3. Évaluation détaillée
python pyod_detectors.py  # 4. Comparer les détecteurs d'anomalies

jupyter lab               # Notebooks d'exploration
```

## Déploiement

Le service est déployé sur Render comme *Web Service* Docker.

```bash
docker build -t yele-ai .
docker run -p 8000:8000 \
  -e SPEEDTEST_DATABASE_CONNECTION_STRING="mongodb+srv://..." \
  yele-ai
```

L'image de production n'embarque ni matplotlib, ni JupyterLab, ni TabPFN.
TabPFN nécessite un environnement graphique indisponible en conteneur : la
prévision retombe donc sur Gradient Boosting en production.

Une fois le service en ligne, renseigner son URL dans le backend Go via
`SPEEDTEST_AI_SERVICE_URL`. Sans cette variable, les routes `/api/ai/*` ne sont
pas exposées et le tableau de bord affiche un état vide explicite.

## Contribuer

Pistes concrètes :

- Enrichir le jeu de données : les modèles restent limités par le volume.
- Prévision multi-variable (latence et gigue, pas seulement le débit).
- Détection tenant compte de la saisonnalité horaire et hebdomadaire.
- Tests automatisés : le module n'en comporte aucun aujourd'hui.

## Licence

À définir avant l'ouverture publique du dépôt.
