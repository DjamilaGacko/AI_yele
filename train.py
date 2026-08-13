"""Entraînement des modèles IA Yélé + rapport de qualité.

Entraîne les deux modèles (détection d'anomalies, prédiction de débit)
sur les données MongoDB et affiche un rapport : volume, opérateurs,
erreur de prédiction (validation croisée temporelle) et aperçu des
anomalies détectées.

Usage :
    python train.py
"""

import joblib
import numpy as np
from dotenv import load_dotenv

from model import MIN_SAMPLES_ML, YeleAI
from prepare_data import load_clean_dataframe

load_dotenv()

MODEL_FILE = "yele_ai.joblib"


def main() -> None:
    print("=== Entraînement IA Yélé ===\n")

    # 1) Données : export local (yele_speedtest.*.json) si présent, sinon Mongo
    df = load_clean_dataframe()
    if df.empty:
        raise SystemExit("Aucune donnée exploitable dans la base.")
    print(f"[données]   {len(df)} mesures exploitables")
    print(f"[période]   {df['timestamp'].min():%d/%m/%Y} -> {df['timestamp'].max():%d/%m/%Y}")
    print(f"[opérateurs] {df['operator'].nunique()} : "
          f"{', '.join(df['operator'].value_counts().head(5).index)}")

    # 2) Entraînement complet
    ai = YeleAI()
    ai.train(df)
    print(f"\n[modèles]   statistiques IQR : OK ({len(ai.group_stats)} opérateurs)")
    if ai.ml_ready:
        print(f"[modèles]   Détection ({ai.detector_name}) : OK")
        print("[modèles]   Régression (prévision) : OK "
              f"(écart-type des résidus : {ai.resid_std:.1f} Mb/s)")
    else:
        print(f"[modèles]   ML désactivé : {len(df)} mesures < {MIN_SAMPLES_ML} requises")
        print("            (les anomalies utilisent les statistiques IQR uniquement)")

    # 3) Évaluation honnête de la prédiction : split temporel 80/20
    #    (on prédit le futur à partir du passé, jamais l'inverse)
    if ai.ml_ready and len(df) >= 100:
        df_sorted = df.sort_values("timestamp")
        cut = int(len(df_sorted) * 0.8)
        train_df, test_df = df_sorted.iloc[:cut], df_sorted.iloc[cut:]
        eval_ai = YeleAI()
        eval_ai.train(train_df)
        if eval_ai.ml_ready:
            pred = eval_ai.reg.predict(eval_ai._features(test_df))
            mae = float(np.mean(np.abs(test_df["dl"].to_numpy() - pred)))
            baseline = float(np.mean(np.abs(
                test_df["dl"].to_numpy() - train_df["dl"].median())))
            print(f"\n[évaluation] MAE prédiction : {mae:.1f} Mb/s "
                  f"(référence naïve « médiane » : {baseline:.1f} Mb/s)")
            if mae < baseline:
                print("             ✓ le modèle fait mieux que la référence naïve")
            else:
                print("             ⚠ pas encore mieux que la médiane — il faut "
                      "plus de données/variété")

    # 4) Anomalies sur les 30 derniers jours
    import pandas as pd
    recent = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(days=30)]
    found = ai.detect(recent, limit=10)
    print(f"\n[anomalies] {len(found)} détectées sur {len(recent)} mesures récentes")
    for a in found[:5]:
        print(f"  - {a['timestamp'][:16]} {a['operator']} "
              f"({a['download']} Mb/s) sévérité {a['severity']}/100")
        for r in a["reasons"]:
            print(f"      · {r}")

    # 5) Sérialisation : le modèle entraîné devient un fichier réutilisable
    #    (le service web le rechargera sans devoir ré-apprendre).
    joblib.dump(ai, MODEL_FILE)
    print(f"\n[sauvegarde] modèle entraîné -> {MODEL_FILE}")
    print("Entraînement terminé. Lancer le service : uvicorn service:app --port 8000")


if __name__ == "__main__":
    main()
