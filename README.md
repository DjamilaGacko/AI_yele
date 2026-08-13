# Yélé — Module IA (prétraitement & exploration)

Extraction et prétraitement des mesures réseau depuis MongoDB Atlas, en vue de
la **détection d'anomalies** et de la **prédiction de qualité** du réseau.

## Contenu
| Fichier | Rôle |
|---------|------|
| `prepare_data.py` | Lit MongoDB (sans les champs PII), nettoie, produit `dataset_clean.csv` |
| `explore.ipynb` | Notebook d'exploration (distributions, opérateurs, évolution) |
| `requirements.txt` | Dépendances Python |
| `.env.example` | Modèle de configuration (chaîne de connexion) |

## Installation
```bash
# (recommandé) environnement virtuel
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

## Configuration
Copier `.env.example` en `.env` et y coller la chaîne de connexion Atlas
(récupérée sur Render ou dans Atlas → *Connect*). Le fichier `.env` est ignoré
par git — **ne jamais committer le secret**.

```
SPEEDTEST_DATABASE_CONNECTION_STRING="mongodb+srv://USER:PASS@CLUSTER.mongodb.net/"
SPEEDTEST_DATABASE_NAME="yele_speedtest"
```

## Utilisation
```bash
# 1) Générer le dataset propre
python prepare_data.py

# 2) Explorer
jupyter lab explore.ipynb
```

## Notes
- Les champs personnels (`ip_address`, `isp_info`, `log`, `user_agent`…) sont
  **exclus dès l'extraction**.
- `is_wifi` distingue les tests WiFi des tests mobiles : pour analyser le réseau
  **mobile**, filtrer `df[~df['is_wifi']]` et utiliser `cellular_tech` /
  `sim_operator` (pas `operator`, qui est le FAI de connexion).
- Prochaines étapes : détection d'anomalies (Isolation Forest / IQR), puis
  prédiction (LightGBM), puis exposition via un service FastAPI.
