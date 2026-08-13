"""Option B — Comparaison de détecteurs d'anomalies (autonome).

Compare, sur les mêmes données, plusieurs détecteurs NON supervisés pour
remplacer éventuellement l'Isolation Forest utilisé dans model.py :

  - Isolation Forest  (référence actuelle, scikit-learn)
  - ECOD              (PyOD, sans paramètre)
  - COPOD             (PyOD, sans paramètre)
  - LOF               (PyOD, anomalies locales par voisinage)

Méthode d'évaluation identique à evaluate.py : injection synthétique
d'anomalies (on dégrade volontairement une partie du jeu de test, puis on
mesure si chaque détecteur les retrouve — ROC-AUC, PR-AUC, F1).

Ce fichier est INDÉPENDANT de model.py : il ne modifie rien. Il sert à décider
si ECOD/COPOD valent mieux qu'Isolation Forest AVANT toute intégration.

Usage :
    python pyod_detectors.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.lof import LOF

DATASET = "dataset_clean.csv"
KPI = ["dl", "ul", "ping", "jitter"]
CONTAMINATION = 0.05     # part d'anomalies attendue
INJECTION_RATE = 0.15    # part du jeu de test transformée en anomalie connue
N_REPEATS = 20           # répétitions de l'injection (réduit le hasard)
SEED = 42


# ── Détecteurs à comparer ────────────────────────────────────────────────────

def build_detectors() -> dict:
    """Un dictionnaire nom -> détecteur neuf (non entraîné)."""
    return {
        "Isolation Forest": IsolationForest(
            n_estimators=200, contamination=CONTAMINATION, random_state=SEED),
        "ECOD": ECOD(contamination=CONTAMINATION),
        "COPOD": COPOD(contamination=CONTAMINATION),
        "LOF": LOF(contamination=CONTAMINATION, n_neighbors=15),
    }


def anomaly_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Score où « plus grand = plus anormal », quel que soit le modèle."""
    if isinstance(model, IsolationForest):
        return -model.decision_function(X)   # sklearn : grand = normal -> on inverse
    return model.decision_function(X)        # PyOD : grand = anormal


# ── Injection synthétique d'anomalies (comme evaluate.py) ────────────────────

def inject_anomalies(df: pd.DataFrame, rng: np.random.Generator
                     ) -> tuple[pd.DataFrame, np.ndarray]:
    """Dégrade volontairement une partie des mesures -> étiquettes connues."""
    df = df.copy().reset_index(drop=True)
    y = np.zeros(len(df), dtype=int)
    n = max(int(len(df) * INJECTION_RATE), 1)
    idx = rng.choice(len(df), size=n, replace=False)
    for i in idx:
        panne = rng.integers(0, 3)
        if panne == 0:                                    # effondrement débit
            df.at[i, "dl"] = float(df.at[i, "dl"]) * rng.uniform(0.02, 0.12)
        elif panne == 1:                                  # latence extrême
            df.at[i, "ping"] = float(df.at[i, "ping"]) * rng.uniform(5, 12)
        else:                                             # instabilité
            df.at[i, "dl"] = float(df.at[i, "dl"]) * rng.uniform(0.1, 0.25)
            df.at[i, "jitter"] = float(df.at[i, "jitter"]) * rng.uniform(4, 10)
        y[i] = 1
    return df, y


# ── Comparaison ──────────────────────────────────────────────────────────────

def compare(df: pd.DataFrame, n_repeats: int = N_REPEATS) -> pd.DataFrame:
    """Entraîne chaque détecteur sur des données saines, puis mesure sa
    capacité à retrouver des anomalies injectées. Retourne un tableau trié."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(df) * 0.7)
    train_base = df.iloc[:cut]
    test_base = df.iloc[cut:].reset_index(drop=True)

    med = train_base[KPI].median()
    x_train = train_base[KPI].fillna(med)

    rng = np.random.default_rng(SEED)
    # On fige les mêmes jeux de test injectés pour tous les détecteurs (équité).
    injected = [inject_anomalies(test_base, rng) for _ in range(n_repeats)]

    rows = []
    for name, model in build_detectors().items():
        model.fit(x_train)
        ys, scores = [], []
        for test_df, y in injected:
            s = anomaly_scores(model, test_df[KPI].fillna(med))
            ys.append(y)
            scores.append(s)
        y = np.concatenate(ys)
        s = np.concatenate(scores)

        # Meilleur F1 en balayant le seuil (percentiles du score)
        best_f1 = 0.0
        for q in np.linspace(0.5, 0.99, 50):
            thr = np.quantile(s, q)
            pred = (s >= thr).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f = 2 * p * r / max(p + r, 1e-9)
            best_f1 = max(best_f1, f)

        rows.append({
            "Détecteur": name,
            "ROC-AUC": round(float(roc_auc_score(y, s)), 3),
            "PR-AUC": round(float(average_precision_score(y, s)), 3),
            "F1 (max)": round(best_f1, 3),
        })

    out = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    return out.reset_index(drop=True)


def main() -> None:
    df = pd.read_csv(DATASET, parse_dates=["timestamp"])
    print(f"Dataset : {len(df)} mesures · KPI : {KPI}")
    print(f"Réf. hasard PR-AUC ≈ {INJECTION_RATE:.3f} (taux d'anomalies injectées)\n")
    table = compare(df)
    print(table.to_string(index=False))
    best = table.iloc[0]["Détecteur"]
    iso = table[table["Détecteur"] == "Isolation Forest"].iloc[0]
    print(f"\nMeilleur ROC-AUC : {best}")
    if best != "Isolation Forest":
        gain = table.iloc[0]["ROC-AUC"] - iso["ROC-AUC"]
        print(f"-> {best} dépasse Isolation Forest de {gain:+.3f} en ROC-AUC.")
    else:
        print("-> Isolation Forest reste le meilleur ici (ne pas changer).")
    print("\n⚠️  Anomalies fabriquées : scores = borne supérieure optimiste.")


if __name__ == "__main__":
    main()
