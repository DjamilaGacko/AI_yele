"""Étape 2 (nettoyage) : préparation des mesures Yélé pour l'IA.

Lit l'export Compass local (yele_speedtest.speedtest_results.json ou
raw.json) s'il existe, sinon interroge MongoDB directement. Applique le
nettoyage et produit `dataset_clean.csv`, prêt pour la modélisation.

Chaque transformation est journalisée pour comprendre ce qui est fait :
  1. conversion des types (Compass $date, débits texte -> nombres)
  2. récupération d'infos dans les anciens formats (isp_info, extra_raw)
  3. exclusion des mesures inexploitables (débit vide/0, aberrations)
  4. création des variables temporelles (heure, jour, week-end)
  5. suppression des champs personnels (PII)

Usage :
    python prepare_data.py
"""

import json
import os
import re

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

LOCAL_FILES = ["yele_speedtest.speedtest_results.json", "raw.json"]
OUTPUT = "dataset_clean.csv"
NUMERIC = ["dl", "ul", "ping", "jitter"]
# Champs personnels : jamais conservés dans le dataset final.
PII_COLUMNS = ["_id", "ip_address", "isp_info", "user_agent",
               "language", "log", "extra_raw"]


# ── 1. Chargement ─────────────────────────────────────────────────────────────

def _flatten_extended_json(value):
    """Compass exporte les types Mongo en objets ($date, $oid...)."""
    if isinstance(value, dict):
        if "$date" in value:
            return value["$date"]
        if "$oid" in value:
            return value["$oid"]
        if "$numberDouble" in value:
            return value["$numberDouble"]
        if "$numberLong" in value:
            return value["$numberLong"]
    return value


def load_raw() -> pd.DataFrame:
    for name in LOCAL_FILES:
        if os.path.exists(name):
            with open(name, encoding="utf-8") as f:
                docs = json.load(f)
            docs = [{k: _flatten_extended_json(v) for k, v in d.items()}
                    for d in docs]
            print(f"[chargement] {len(docs)} mesures depuis {name}")
            return pd.DataFrame(docs)

    # Repli : connexion directe MongoDB (nécessite .env)
    from pymongo import MongoClient
    uri = os.environ.get("SPEEDTEST_DATABASE_CONNECTION_STRING")
    if not uri:
        raise SystemExit("Aucun export local trouvé et pas de chaîne de "
                         "connexion dans .env")
    db = os.environ.get("SPEEDTEST_DATABASE_NAME", "yele_speedtest")
    projection = {c: 0 for c in PII_COLUMNS if c != "_id"} | {"_id": 0}
    docs = list(MongoClient(uri)[db]["speedtest_results"].find({}, projection))
    print(f"[chargement] {len(docs)} mesures depuis MongoDB")
    return pd.DataFrame(docs)


# ── 2. Récupération des anciens formats ──────────────────────────────────────

def salvage_legacy_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Les 1ers tests n'avaient pas operator/network_type structurés :
    l'information est récupérable dans isp_info et extra_raw."""

    def op_from_isp(s):
        # "102.180.101.179 - Orange Burkina Faso, BF (40 mi)" -> "Orange Burkina Faso"
        if not isinstance(s, str) or s.startswith("{") or " - " not in s:
            return None
        return s.split(" - ", 1)[1].split(",")[0].strip() or None

    def device_net_from_extra(s):
        # Ancien format texte : "SM-S9010 - 4G" -> (SM-S9010, 4G)
        if not isinstance(s, str) or not s or s.startswith("{"):
            return None, None
        parts = [p.strip() for p in s.split(" - ")]
        return (parts[0] or None,
                parts[1] if len(parts) > 1 and parts[1] else None)

    def country_from_isp(s):
        # "... - Orange Burkina Faso, BF (40 mi)" -> "BF" (code ISO,
        # cohérent avec ce que le backend remplira pour les futurs tests)
        if not isinstance(s, str) or s.startswith("{"):
            return None
        m = re.search(r",\s*([A-Z]{2})\s*\(", s)
        return m.group(1) if m else None

    def _is_number(s):
        try:
            float(s)
            return True
        except (TypeError, ValueError):
            return False

    def _valid_place(s):
        # Un nom de lieu plausible : pas un nombre (coordonnées GPS stockées
        # par erreur), pas un fragment tronqué avec parenthèse ("PTT)").
        return (isinstance(s, str) and len(s) >= 2 and not _is_number(s)
                and "(" not in s and ")" not in s
                and not any(c.isdigit() for c in s))

    def city_country_from_location(s):
        # "Ouagadougou, Burkina Faso" -> ("Ouagadougou", "Burkina Faso")
        # Rejette "12.3063, -1.5122" (GPS) et "PTT), BF" (nom tronqué).
        if not isinstance(s, str) or "," not in s:
            return None, None
        city, _, country = s.partition(",")
        city, country = city.strip(), country.strip()
        return (city if _valid_place(city) else None,
                country if _valid_place(country) else None)

    salvaged_op = salvaged_dev = salvaged_net = 0
    salvaged_city = salvaged_country = 0
    for i in df.index:
        if not df.at[i, "operator"] and "isp_info" in df.columns:
            op = op_from_isp(df.at[i, "isp_info"])
            if op:
                df.at[i, "operator"] = op
                salvaged_op += 1
        if "extra_raw" in df.columns:
            dev, net = device_net_from_extra(df.at[i, "extra_raw"])
            if dev and not df.at[i, "device_model"]:
                df.at[i, "device_model"] = dev
                salvaged_dev += 1
            if net and not df.at[i, "network_type"]:
                df.at[i, "network_type"] = net
                salvaged_net += 1
        # city / country : depuis location ("Ville, Pays"), puis le code
        # pays ISO depuis isp_info en dernier recours.
        city, country = city_country_from_location(df.at[i, "location"]
                                                   if "location" in df.columns
                                                   else None)
        if not df.at[i, "city"] and city:
            df.at[i, "city"] = city
            salvaged_city += 1
        if not df.at[i, "country"]:
            cc = country or (country_from_isp(df.at[i, "isp_info"])
                             if "isp_info" in df.columns else None)
            if cc:
                df.at[i, "country"] = cc
                salvaged_country += 1

    print(f"[récupération] opérateur retrouvé via isp_info : {salvaged_op}")
    print(f"[récupération] appareil/réseau via extra_raw    : "
          f"{salvaged_dev}/{salvaged_net}")
    print(f"[récupération] city/country via location        : "
          f"{salvaged_city}/{salvaged_country}")
    return df


# ── Normalisation des libellés (doublons) ────────────────────────────────────
# Le même acteur apparaît sous plusieurs écritures (noms tronqués, variantes).
# Sans fusion, les stats et l'IA les traitent comme des catégories distinctes.

OPERATOR_ALIASES = [
    # (motif contenu dans le nom, libellé canonique)
    ("onatel", "ONATEL / Telmob"),
    ("office national des telecommunications", "ONATEL / Telmob"),
    ("orange burkina", "Orange Burkina Faso"),
    ("anptic", "ANPTIC"),
    ("agence nationale de promotion des tic", "ANPTIC"),
    ("autorite de regulation des communications", "ARCEP Burkina"),
    ("telecel", "Telecel Faso"),
    ("moov", "Moov Africa"),
]

COUNTRY_TO_ISO = {
    "burkina faso": "BF",
    "cote d'ivoire": "CI",
    "côte d'ivoire": "CI",
    "mali": "ML",
    "niger": "NE",
    "senegal": "SN",
    "sénégal": "SN",
    "france": "FR",
    "pays-bas": "NL",
    "netherlands": "NL",
}


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    def canon_operator(name):
        if not isinstance(name, str) or not name:
            return name
        low = name.lower()
        for pattern, canonical in OPERATOR_ALIASES:
            if pattern in low:
                return canonical
        return name.strip()

    def canon_country(name):
        if not isinstance(name, str) or not name:
            return name
        if len(name.strip()) == 2:          # déjà un code ISO
            return name.strip().upper()
        return COUNTRY_TO_ISO.get(name.strip().lower(), name.strip())

    before_ops = df["operator"].nunique()
    df["operator"] = df["operator"].map(canon_operator)
    df["country"] = df["country"].map(canon_country)
    print(f"[normalisation] opérateurs : {before_ops} -> "
          f"{df['operator'].nunique()} libellés distincts")
    return df


# ── 3-5. Nettoyage principal ─────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    n0 = len(df)

    # Types : débits/latences texte -> nombres ; timestamp -> datetime
    for col in NUMERIC:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    for col in ["latitude", "longitude"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0)
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce",
                                     utc=True, format="mixed")
    df = df.dropna(subset=["timestamp"])

    # Récupération des anciens formats AVANT d'écarter quoi que ce soit
    df = salvage_legacy_fields(df)

    # Exclusions : mesures sans débit exploitable ou aberrantes
    before = len(df)
    df = df.dropna(subset=["dl"])
    df = df[(df["dl"] > 0) & (df["dl"] < 2000)]
    df = df[(df["ping"].isna()) | ((df["ping"] >= 0) & (df["ping"] < 5000))]
    print(f"[exclusion]  {before - len(df)} mesures écartées "
          f"(débit vide/0 ou valeurs impossibles)")

    # Variables temporelles (la charge réseau dépend de l'heure et du jour)
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Catégorielles : vides -> "Inconnu"
    for col in ["operator", "network_type", "sim_operator", "cellular_tech",
                "device_model", "location", "city", "country"]:
        if col in df.columns:
            df[col] = df[col].fillna("Inconnu").replace("", "Inconnu")

    # Fusion des libellés en doublon (ONATEL x2, pays en toutes lettres...)
    df = normalize_labels(df)

    # Colonne location : on écarte les valeurs corrompues (coordonnées GPS,
    # fragments type "PTT), BF") et on la recompose depuis city + country.
    def rebuild_location(row):
        loc = row.get("location")
        bad = (not isinstance(loc, str) or loc == "Inconnu"
               or "(" in loc or ")" in loc
               or any(c.isdigit() for c in loc))
        if not bad:
            return loc
        city, country = row.get("city"), row.get("country")
        parts = [p for p in (city, country) if p and p != "Inconnu"]
        return ", ".join(parts) if parts else "Inconnu"

    fixed = 0
    for i in df.index:
        new_loc = rebuild_location(df.loc[i])
        if new_loc != df.at[i, "location"]:
            df.at[i, "location"] = new_loc
            fixed += 1
    if fixed:
        print(f"[normalisation] location recomposée depuis city/country : {fixed}")

    df["is_wifi"] = (df["network_type"].astype(str).str.upper()
                     .str.contains("WIFI", na=False)).astype(int)

    # Suppression des champs personnels
    df = df.drop(columns=[c for c in PII_COLUMNS if c in df.columns])

    print(f"[bilan]      {len(df)}/{n0} mesures conservées")
    return df.reset_index(drop=True)


def summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("[résumé] dataset vide.")
        return
    print("\n--- Résumé du dataset nettoyé ---")
    print(f"Période  : {df['timestamp'].min():%d/%m/%Y} -> "
          f"{df['timestamp'].max():%d/%m/%Y}")
    print(f"Part WiFi: {df['is_wifi'].mean() * 100:.0f}%")
    print("\nMesures par opérateur :")
    print(df["operator"].value_counts().head(8).to_string())
    print("\nDébit descendant (Mb/s) :")
    print(df["dl"].describe().round(1).to_string())


def load_clean_dataframe() -> pd.DataFrame:
    """Point d'entrée partagé (utilisé aussi par model.py / train.py)."""
    return clean(load_raw())


def main() -> None:
    df = load_clean_dataframe()
    summary(df)
    if not df.empty:
        df.to_csv(OUTPUT, index=False)
        print(f"\n[sauvegarde] {len(df)} lignes -> {OUTPUT}")


if __name__ == "__main__":
    main()
