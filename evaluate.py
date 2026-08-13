"""Évaluation complète des modèles IA Yélé — toutes les métriques.

Deux modèles, deux familles de métriques (on ne peut pas les confondre) :

  A. PRÉDICTION (régression)   -> MAE, RMSE, MedAE, R², MAPE, skill score
     Il n'existe pas de F1/AUC pour une régression : ces métriques mesurent
     une classification. L'équivalent du « bon / pas bon » est le skill score
     face à la référence naïve.

  B. DÉTECTION (non supervisé) -> F1, ROC-AUC, PR-AUC, précision, rappel
     Ces métriques exigent des étiquettes. Comme personne n'a annoté les
     vraies anomalies, on les fabrique par INJECTION SYNTHÉTIQUE : on dégrade
     volontairement des mesures du jeu de test et on vérifie que le modèle
     les retrouve. C'est une borne SUPÉRIEURE optimiste (voir avertissement).

  C. ANNOTATION MANUELLE -> exporte les anomalies à valider par un humain,
     seule façon d'obtenir une précision réelle sur les vraies données.

Usage :
    python evaluate.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)

from model import KPI, YeleAI

DATASET = "dataset_clean.csv"
ANNOT_FILE = "anomalies_a_annoter.csv"
INJECTION_RATE = 0.15   # part du jeu de test transformée en anomalie connue
N_REPEATS = 20          # répétitions de l'injection (réduit le hasard)
SEED = 42


def _titre(txt: str) -> None:
    print(f"\n{'=' * 70}\n{txt}\n{'=' * 70}")


def _sous(txt: str) -> None:
    print(f"\n--- {txt} ---")


# ══════════════════════════════════════════════════════════════════════════
# A. PRÉDICTION — métriques de régression
# ══════════════════════════════════════════════════════════════════════════

def metriques_regression(y_vrai: np.ndarray, y_pred: np.ndarray,
                         baseline: float) -> dict:
    """Toutes les métriques de régression, avec leur interprétation."""
    err = y_vrai - y_pred
    mae = float(np.mean(np.abs(err)))
    mae_base = float(np.mean(np.abs(y_vrai - baseline)))
    return {
        "MAE": mae,
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MedAE": float(np.median(np.abs(err))),
        "R2": float(r2_score(y_vrai, y_pred)) if len(y_vrai) > 1 else float("nan"),
        "MAPE": float(np.mean(np.abs(err / np.maximum(y_vrai, 1e-6))) * 100),
        "biais": float(np.mean(err)),
        "MAE_baseline": mae_base,
        # Skill score : 1 = parfait, 0 = équivalent à la médiane, < 0 = pire
        "skill": float(1 - mae / mae_base) if mae_base > 0 else float("nan"),
    }


def evaluer_prediction(df: pd.DataFrame) -> dict | None:
    _titre("A. PRÉDICTION DU DÉBIT — métriques de régression")
    print("Note : F1 et AUC n'existent pas pour une régression (ce sont des")
    print("       métriques de classification). L'équivalent est le skill score.")

    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── A1. Split temporel simple 80/20 ───────────────────────────────────
    cut = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:cut], df.iloc[cut:]
    ai = YeleAI()
    ai.train(train_df)
    if not ai.ml_ready:
        print(f"\n[!] ML inactif ({len(train_df)} mesures d'entraînement).")
        return None

    pred = ai.reg.predict(ai._features(test_df))
    y = test_df["dl"].to_numpy()
    m = metriques_regression(y, pred, float(train_df["dl"].median()))

    _sous(f"A1. Split temporel 80/20 ({len(train_df)} train / {len(test_df)} test)")
    print(f"  MAE   (erreur moyenne)      : {m['MAE']:7.2f} Mb/s")
    print(f"  RMSE  (pénalise les gros écarts) : {m['RMSE']:7.2f} Mb/s")
    print(f"  MedAE (erreur médiane, robuste)  : {m['MedAE']:7.2f} Mb/s")
    print(f"  MAPE  (erreur relative)     : {m['MAPE']:7.1f} %")
    print(f"  R²    (variance expliquée)  : {m['R2']:7.3f}   "
          f"({'négatif = pire que la moyenne' if m['R2'] < 0 else 'part expliquée'})")
    print(f"  Biais (sur/sous-estimation) : {m['biais']:+7.2f} Mb/s")
    print(f"\n  Référence naïve (médiane)   : {m['MAE_baseline']:7.2f} Mb/s")
    print(f"  SKILL SCORE                 : {m['skill']:+7.3f}")
    if m["skill"] > 0:
        print(f"  -> Le modele fait {m['skill'] * 100:.0f} % mieux que la mediane. OK")
    else:
        print("  -> ECHEC : le modele fait PIRE que predire toujours la mediane.")

    # ── A2. Validation croisée à origine glissante ────────────────────────
    # Un seul split sur 120 lignes est fragile. On répète l'exercice en
    # avançant la coupure : apprendre sur le passé, tester sur la suite.
    _sous("A2. Validation croisée temporelle (origine glissante, 4 plis)")
    print("  Plus fiable qu'un seul split : on répète en avançant la coupure.")
    maes, skills = [], []
    for k, frac in enumerate([0.5, 0.6, 0.7, 0.8], 1):
        c = int(len(df) * frac)
        tr, te = df.iloc[:c], df.iloc[c:c + max(len(df) // 8, 5)]
        if len(te) < 3:
            continue
        a = YeleAI()
        a.train(tr)
        if not a.ml_ready:
            continue
        p = a.reg.predict(a._features(te))
        mm = metriques_regression(te["dl"].to_numpy(), p, float(tr["dl"].median()))
        maes.append(mm["MAE"])
        skills.append(mm["skill"])
        print(f"  Pli {k} : train={len(tr):3d}  test={len(te):3d}  "
              f"MAE={mm['MAE']:6.2f}  baseline={mm['MAE_baseline']:6.2f}  "
              f"skill={mm['skill']:+.3f}")
    if maes:
        print(f"\n  MAE moyenne   : {np.mean(maes):6.2f} Mb/s "
              f"(ecart-type {np.std(maes):.2f})")
        print(f"  Skill moyen   : {np.mean(skills):+6.3f}")
        print(f"  Plis gagnants : {sum(s > 0 for s in skills)}/{len(skills)}")

    # ── A3. Détail par opérateur ──────────────────────────────────────────
    _sous("A3. Erreur par opérateur (où le modèle se trompe-t-il ?)")
    tmp = test_df.copy()
    tmp["err_abs"] = np.abs(y - pred)
    par_op = (tmp.groupby("operator")["err_abs"]
              .agg(["count", "mean"]).round(2)
              .sort_values("mean", ascending=False))
    print(par_op.to_string())

    return m


# ══════════════════════════════════════════════════════════════════════════
# B. DÉTECTION — métriques de classification (F1, AUC)
# ══════════════════════════════════════════════════════════════════════════

def injecter_anomalies(df: pd.DataFrame, rng: np.random.Generator
                       ) -> tuple[pd.DataFrame, np.ndarray]:
    """Dégrade volontairement une partie des mesures -> étiquettes connues.

    Trois pannes réalistes, inspirées de vrais incidents réseau :
      - effondrement du débit (congestion / saturation cellule)
      - latence extrême (routage dégradé, satellite de secours)
      - instabilité (jitter fort + débit divisé)
    """
    df = df.copy().reset_index(drop=True)
    y = np.zeros(len(df), dtype=int)
    n = max(int(len(df) * INJECTION_RATE), 1)
    idx = rng.choice(len(df), size=n, replace=False)

    for i in idx:
        panne = rng.integers(0, 3)
        if panne == 0:                                   # effondrement débit
            df.at[i, "dl"] = float(df.at[i, "dl"]) * rng.uniform(0.02, 0.12)
        elif panne == 1:                                 # latence extrême
            df.at[i, "ping"] = float(df.at[i, "ping"]) * rng.uniform(5, 12)
        else:                                            # instabilité
            df.at[i, "dl"] = float(df.at[i, "dl"]) * rng.uniform(0.1, 0.25)
            df.at[i, "jitter"] = float(df.at[i, "jitter"]) * rng.uniform(4, 10)
        y[i] = 1
    return df, y


def severites(ai: YeleAI, df: pd.DataFrame) -> np.ndarray:
    """Score de sévérité (0-100) pour CHAQUE ligne, via le vrai detect()."""
    trouve = ai.detect(df, limit=len(df))
    par_ts = {a["timestamp"]: a["severity"] for a in trouve}
    return np.array([par_ts.get(t.isoformat(), 0) for t in df["timestamp"]],
                    dtype=float)


def evaluer_detection(df: pd.DataFrame) -> dict | None:
    _titre("B. DÉTECTION D'ANOMALIES — F1, ROC-AUC, PR-AUC")
    print("Méthode : injection synthétique. Le modèle apprend sur des données")
    print("saines, puis on dégrade volontairement 15 % du jeu de test et on")
    print("vérifie qu'il les retrouve. C'est la seule façon d'avoir des")
    print("étiquettes sans annotation humaine.")

    df = df.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(df) * 0.7)
    train_df, test_base = df.iloc[:cut], df.iloc[cut:].reset_index(drop=True)

    ai = YeleAI()
    ai.train(train_df)
    print(f"\n  Entraînement : {len(train_df)} mesures saines "
          f"(ML actif : {ai.ml_ready})")
    print(f"  Test         : {len(test_base)} mesures, "
          f"{N_REPEATS} répétitions d'injection")

    rng = np.random.default_rng(SEED)
    tous_y, tous_s = [], []
    for _ in range(N_REPEATS):
        test_df, y = injecter_anomalies(test_base, rng)
        s = severites(ai, test_df)
        tous_y.append(y)
        tous_s.append(s)
    y = np.concatenate(tous_y)
    s = np.concatenate(tous_s)

    if y.sum() == 0 or y.sum() == len(y):
        print("\n[!] Pas assez de variété pour calculer les métriques.")
        return None

    # ── B1. Métriques indépendantes du seuil ──────────────────────────────
    roc = float(roc_auc_score(y, s))
    pr = float(average_precision_score(y, s))
    taux_base = float(y.mean())

    _sous("B1. Métriques globales (indépendantes du seuil)")
    print(f"  ROC-AUC : {roc:.3f}   (0,5 = hasard · 1,0 = parfait)")
    print(f"  PR-AUC  : {pr:.3f}   (référence hasard = {taux_base:.3f} "
          f"= taux d'anomalies)")
    print(f"  -> {'Bon' if roc >= 0.8 else 'Moyen' if roc >= 0.65 else 'Faible'} "
          f"pouvoir discriminant")

    # ── B2. Seuil optimal et F1 ───────────────────────────────────────────
    prec, rapp, seuils = precision_recall_curve(y, s)
    f1s = 2 * prec * rapp / np.maximum(prec + rapp, 1e-9)
    best = int(np.nanargmax(f1s[:-1])) if len(f1s) > 1 else 0
    seuil_opt = float(seuils[best])

    _sous("B2. F1 et seuil de décision")
    print(f"  {'Seuil':>6} {'Précision':>10} {'Rappel':>8} {'F1':>7}   commentaire")
    for seuil in [10, 20, 30, 40, 50, 60]:
        pred = (s >= seuil).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-9)
        note = "<- seuil actuel du code" if seuil == 30 else ""
        print(f"  {seuil:>6} {p:>10.3f} {r:>8.3f} {f:>7.3f}   {note}")
    print(f"\n  MEILLEUR F1 : {f1s[best]:.3f} au seuil {seuil_opt:.0f} "
          f"(précision {prec[best]:.3f}, rappel {rapp[best]:.3f})")

    # ── B3. Matrice de confusion au seuil optimal ─────────────────────────
    pred = (s >= seuil_opt).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    _sous(f"B3. Matrice de confusion (seuil {seuil_opt:.0f})")
    print(f"                    prédit normal   prédit anomalie")
    print(f"  vraiment normal   {tn:>13d}   {fp:>15d}  <- fausses alertes")
    print(f"  vraie anomalie    {fn:>13d}   {tp:>15d}")
    print(f"\n  Spécificité (vrais normaux bien classés) : "
          f"{tn / max(tn + fp, 1):.3f}")
    print(f"  Taux de fausses alertes                  : "
          f"{fp / max(fp + tn, 1):.3f}")

    # ── B4. Précision@k — ce que voit vraiment l'utilisateur ──────────────
    _sous("B4. Précision@k (l'utilisateur ne regarde que le haut de liste)")
    ordre = np.argsort(-s)
    for k in [5, 10, 20, 50]:
        if k <= len(s):
            print(f"  Précision@{k:<3d} : {y[ordre[:k]].mean():.3f}  "
                  f"({int(y[ordre[:k]].sum())}/{k} vraies anomalies)")

    # ── B5. Contribution de chaque niveau ─────────────────────────────────
    _sous("B5. Apport de chaque niveau de détection")
    test_df, y1 = injecter_anomalies(test_base, np.random.default_rng(SEED))
    trouve = ai.detect(test_df, limit=len(test_df))
    detname = getattr(ai, "detector_name", "détecteur") or "détecteur"
    motifs = {"IQR (statistique)": 0, detname: 0, "Contextuel": 0}
    for a in trouve:
        for r in a["reasons"]:
            if "en dessous de la normale" in r or "médiane" in r:
                motifs["IQR (statistique)"] += 1
            elif "atypique" in r:
                motifs[detname] += 1
            elif "était attendu" in r:
                motifs["Contextuel"] += 1
    total = max(sum(motifs.values()), 1)
    for nom, n in motifs.items():
        print(f"  {nom:22s} : {n:3d} déclenchements ({100 * n / total:4.0f} %)")

    print("\n  [!] AVERTISSEMENT : ces anomalies sont fabriquées, donc plus")
    print("      faciles à détecter que de vraies pannes. Ces scores sont une")
    print("      BORNE SUPÉRIEURE optimiste, pas la performance réelle.")

    return {"roc_auc": roc, "pr_auc": pr, "f1": float(f1s[best]),
            "seuil": seuil_opt}


# ══════════════════════════════════════════════════════════════════════════
# C. Export pour annotation manuelle (seule vérité terrain possible)
# ══════════════════════════════════════════════════════════════════════════

def exporter_annotation(df: pd.DataFrame) -> None:
    _titre("C. EXPORT POUR ANNOTATION MANUELLE")
    ai = YeleAI()
    ai.train(df)
    trouve = ai.detect(df, limit=30)
    if not trouve:
        print("  Aucune anomalie détectée à annoter.")
        return
    out = pd.DataFrame(trouve)
    out["reasons"] = out["reasons"].map(lambda r: " | ".join(r))
    out["vraie_anomalie"] = ""      # colonne à remplir : 1 = oui, 0 = non
    out.to_csv(ANNOT_FILE, index=False)
    print(f"  {len(out)} anomalies -> {ANNOT_FILE}")
    print("\n  Marque 1 (vraie anomalie) ou 0 (fausse alerte) dans la colonne")
    print("  'vraie_anomalie', puis relance : python evaluate.py")
    print("  Tu obtiendras alors une PRÉCISION RÉELLE, sur tes vraies données.")


def lire_annotation() -> None:
    import os
    if not os.path.exists(ANNOT_FILE):
        return
    ann = pd.read_csv(ANNOT_FILE)
    if "vraie_anomalie" not in ann.columns:
        return
    lab = pd.to_numeric(ann["vraie_anomalie"], errors="coerce").dropna()
    if lab.empty:
        return
    _sous("C2. Précision réelle (d'après ton annotation)")
    print(f"  {len(lab)} anomalies annotées sur {len(ann)}")
    print(f"  PRÉCISION RÉELLE : {lab.mean():.3f} "
          f"({int(lab.sum())} vraies / {len(lab)} signalées)")
    print("  (le rappel reste inconnu : on ignore les anomalies non signalées)")


# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    df = pd.read_csv(DATASET, parse_dates=["timestamp"])
    _titre("ÉVALUATION DES MODÈLES IA YÉLÉ")
    print(f"  Dataset  : {len(df)} mesures")
    print(f"  Période  : {df['timestamp'].min():%d/%m/%Y} -> "
          f"{df['timestamp'].max():%d/%m/%Y} "
          f"({df['timestamp'].dt.date.nunique()} jours distincts)")
    print(f"  KPI      : {KPI}")

    reg = evaluer_prediction(df)
    det = evaluer_detection(df)
    exporter_annotation(df)
    lire_annotation()

    _titre("SYNTHÈSE")
    if reg:
        verdict = "OK" if reg["skill"] > 0 else "ECHEC"
        print(f"  Prédiction  MAE {reg['MAE']:.1f} Mb/s · R² {reg['R2']:.2f} · "
              f"skill {reg['skill']:+.3f}  [{verdict}]")
    if det:
        print(f"  Détection   F1 {det['f1']:.3f} · ROC-AUC {det['roc_auc']:.3f} · "
              f"PR-AUC {det['pr_auc']:.3f}")
    print(f"\n  Rappel : {len(df)} mesures sur "
          f"{df['timestamp'].dt.date.nunique()} jours. Toute conclusion reste")
    print("  fragile tant que le dataset n'atteint pas ~500 mesures.")


if __name__ == "__main__":
    main()
