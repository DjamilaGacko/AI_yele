"""Comparaison des approches essayées — trace pour la soutenance.

Montre, côte à côte, les modèles testés à chaque étage, et met en évidence
le choix final retenu. Sert à documenter la démarche (« j'ai essayé plusieurs
approches, voici pourquoi j'ai choisi celle-ci »).

  PRÉDICTION du débit   : Gradient Boosting   vs   TabPFN   (retenu)
  DÉTECTION d'anomalies : Isolation Forest / COPOD / LOF   vs   ECOD (retenu)

Méthode d'évaluation identique à evaluate.py (split temporel + injection).

Usage :
    TABPFN_DISABLE_TELEMETRY=1 python comparaison_approches.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from model import KPI, YeleAI
from pyod_detectors import compare as compare_detectors

DATASET = "dataset_clean.csv"


def _line() -> None:
    print("-" * 64)


# ── PRÉDICTION : Gradient Boosting vs TabPFN ─────────────────────────────────

def compare_prediction(df: pd.DataFrame) -> None:
    print("\n=== PRÉDICTION DU DÉBIT : Gradient Boosting vs TabPFN ===\n")
    df = df.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:cut], df.iloc[cut:]

    # YeleAI entraîne déjà TabPFN (self.reg) et fournit l'encodage des features.
    ai = YeleAI()
    ai.train(train_df)
    if not ai.ml_ready:
        print("ML inactif (trop peu de données).")
        return

    x_tr, x_te = ai._features(train_df), ai._features(test_df)
    y_tr, y_te = train_df["dl"].to_numpy(), test_df["dl"].to_numpy()
    baseline = float(np.median(y_tr))
    mae_base = float(np.mean(np.abs(y_te - baseline)))

    # Approche 1 (ancienne) : Gradient Boosting
    gb = HistGradientBoostingRegressor(random_state=42).fit(x_tr, y_tr)
    pred_gb = gb.predict(x_te)
    # Approche 2 (retenue) : TabPFN, déjà entraîné dans ai.reg
    pred_tp = ai.reg.predict(x_te)

    print(f"{'Modèle':22s} {'MAE':>8} {'skill':>8}   verdict")
    _line()
    for name, pred, retenu in [("Gradient Boosting", pred_gb, False),
                               ("TabPFN", pred_tp, True)]:
        mae = float(np.mean(np.abs(y_te - pred)))
        skill = 1 - mae / mae_base if mae_base else float("nan")
        tag = "  <-- RETENU" if retenu else ""
        print(f"{name:22s} {mae:8.2f} {skill:+8.3f}{tag}")
    print(f"{'Référence (médiane)':22s} {mae_base:8.2f} {0.0:+8.3f}")
    print("\nLecture : MAE en Mb/s (plus bas = mieux) ; skill > 0 = mieux que la médiane.")


# ── DÉTECTION : Isolation Forest / COPOD / LOF vs ECOD ───────────────────────

def compare_detection(df: pd.DataFrame) -> None:
    print("\n\n=== DÉTECTION D'ANOMALIES : détecteurs comparés ===\n")
    table = compare_detectors(df)
    best = table.iloc[0]["Détecteur"]
    print(f"{'Détecteur':20s} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1(max)':>8}   verdict")
    _line()
    for _, r in table.iterrows():
        tag = "  <-- RETENU (explicable)" if r["Détecteur"] == "ECOD" else ""
        print(f"{r['Détecteur']:20s} {r['ROC-AUC']:8.3f} {r['PR-AUC']:8.3f} "
              f"{r['F1 (max)']:8.3f}{tag}")
    print(f"\nMeilleur ROC-AUC brut : {best}. ECOD est retenu pour son explicabilité")
    print("(il dit quelle variable rend la mesure anormale), à score quasi équivalent.")


def main() -> None:
    df = pd.read_csv(DATASET, parse_dates=["timestamp"])
    print("=" * 64)
    print("  COMPARAISON DES APPROCHES ESSAYÉES — YÉLÉ IA")
    print(f"  Dataset : {len(df)} mesures · KPI : {KPI}")
    print("=" * 64)
    compare_prediction(df)
    compare_detection(df)
    print("\n" + "=" * 64)
    print("  CHOIX FINAUX : prédiction = TabPFN · détection = ECOD")
    print("  (⚠️ écarts faibles : le vrai levier reste le volume de données)")
    print("=" * 64)


if __name__ == "__main__":
    main()
