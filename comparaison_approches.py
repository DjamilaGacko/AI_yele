"""Comparaison des approches essayées — trace pour la soutenance.

Montre, côte à côte, les modèles testés à chaque étage, et met en évidence
le choix final retenu. Sert à documenter la démarche (« j'ai essayé plusieurs
approches, voici pourquoi j'ai choisi celle-ci »).

  PRÉDICTION du débit   : TabPFN   vs   Gradient Boosting (retenu)
  DÉTECTION d'anomalies : Isolation Forest / ECOD / COPOD   vs   LOF (retenu)

Méthode d'évaluation identique à evaluate.py (split temporel + injection).
TabPFN n'est installé qu'en local (requirements.txt) : sans lui, seule la
référence et Gradient Boosting sont comparés.

Usage :
    TABPFN_DISABLE_TELEMETRY=1 python comparaison_approches.py
"""

import time

import numpy as np
import pandas as pd

from model import KPI, YeleAI
from pyod_detectors import compare as compare_detectors

DATASET = "dataset_clean.csv"
FOLDS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]


def _line() -> None:
    print("-" * 64)


def make_tabpfn():
    """TabPFN en poids v2 (libres), sur CPU ; None s'il n'est pas installé."""
    try:
        from tabpfn import TabPFNRegressor
    except ImportError:
        return None
    options = dict(ignore_pretraining_limits=True, device="cpu", random_state=42)
    try:
        # TabPFN >= 6 : le modèle par défaut y est la v2.5, à licence restreinte.
        from tabpfn.constants import ModelVersion
        return TabPFNRegressor.create_default_for_version(ModelVersion.V2, **options)
    except ImportError:
        return TabPFNRegressor(**options)


# ── PRÉDICTION : TabPFN vs Gradient Boosting ─────────────────────────────────

def compare_prediction(df: pd.DataFrame) -> None:
    print("\n=== PRÉDICTION DU DÉBIT : Gradient Boosting vs TabPFN ===\n")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Encodage des features identique à celui du service
    enc = YeleAI()
    enc.operators = sorted(df["operator"].unique().tolist())
    enc.networks = sorted(df["network_type"].unique().tolist())
    x, y = enc._features(df), df["dl"].to_numpy()

    from sklearn.ensemble import HistGradientBoostingRegressor
    models = {"Gradient Boosting": lambda: HistGradientBoostingRegressor(random_state=42)}
    if make_tabpfn() is not None:
        models["TabPFN"] = make_tabpfn

    # 1) Découpage 80/20 (tableau du rapport)
    cut = int(len(df) * 0.8)
    mae_base = float(np.mean(np.abs(y[cut:] - np.median(y[:cut]))))
    print(f"Découpage 80/20 ({cut} apprentissage / {len(df) - cut} test)")
    print(f"{'Modèle':22s} {'MAE':>8} {'skill':>8} {'durée':>8}")
    _line()
    print(f"{'Référence (médiane)':22s} {mae_base:8.2f} {0.0:+8.3f}")
    for name, make in models.items():
        t0 = time.time()
        pred = make().fit(x[:cut], y[:cut]).predict(x[cut:])
        mae = float(np.mean(np.abs(y[cut:] - pred)))
        tag = "  <-- RETENU" if name == "Gradient Boosting" else ""
        print(f"{name:22s} {mae:8.2f} {1 - mae / mae_base:+8.3f} "
              f"{time.time() - t0:7.1f}s{tag}")

    # 2) Cinq découpages successifs : le classement tient-il ?
    print("\nCinq découpages chronologiques successifs (skill score)")
    _line()
    skills = {name: [] for name in models}
    for a, b in FOLDS:
        i, j = int(len(df) * a), int(len(df) * b)
        base = float(np.mean(np.abs(y[i:j] - np.median(y[:i]))))
        row = []
        for name, make in models.items():
            mae = float(np.mean(np.abs(y[i:j] - make().fit(x[:i], y[:i]).predict(x[i:j]))))
            skills[name].append(1 - mae / base)
            row.append(f"{name} {skills[name][-1]:+.3f}")
        print(f"  {int(a * 100)}-{int(b * 100)} % : " + " · ".join(row))
    for name, s in skills.items():
        print(f"  {name:20s} skill moyen {np.mean(s):+.3f}")
    print("\nLecture : MAE en Mb/s (plus bas = mieux) ; skill > 0 = mieux que la médiane.")


# ── DÉTECTION : Isolation Forest / ECOD / COPOD vs LOF ───────────────────────

def compare_detection(df: pd.DataFrame) -> None:
    print("\n\n=== DÉTECTION D'ANOMALIES : détecteurs comparés ===\n")
    table = compare_detectors(df)
    print(f"{'Détecteur':20s} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1(max)':>8}   verdict")
    _line()
    for _, r in table.iterrows():
        tag = "  <-- RETENU" if r["Détecteur"] == "LOF" else ""
        print(f"{r['Détecteur']:20s} {r['ROC-AUC']:8.3f} {r['PR-AUC']:8.3f} "
              f"{r['F1 (max)']:8.3f}{tag}")
    print("\nLOF juge les quatre KPI ensemble ; ECOD et COPOD les jugent un par un")
    print("et ratent les combinaisons anormales. Le service explique quand même")
    print("chaque alerte LOF par la variable qui la domine.")


def main() -> None:
    df = pd.read_csv(DATASET)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    print("=" * 64)
    print("  COMPARAISON DES APPROCHES ESSAYÉES — YÉLÉ IA")
    print(f"  Dataset : {len(df)} mesures · KPI : {KPI}")
    print("=" * 64)
    compare_prediction(df)
    compare_detection(df)
    print("\n" + "=" * 64)
    print("  CHOIX FINAUX : prédiction = Gradient Boosting · détection = LOF")
    print("=" * 64)


if __name__ == "__main__":
    main()
