"""Service IA Yélé (FastAPI).

Endpoints (appelés par le backend Go via le proxy /api/ai/*) :
  GET  /ai/health     -> état du service et des modèles
  GET  /ai/anomalies  -> anomalies détectées sur les mesures récentes
  GET  /ai/forecast   -> débit prédit heure par heure pour un opérateur
  POST /ai/retrain    -> ré-entraîne les modèles sur les données actuelles

Lancement local :
    uvicorn service:app --reload --port 8000
"""

import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, Query

from model import MIN_SAMPLES_ML, YeleAI, load_dataframe

load_dotenv()

_state: dict = {"ai": None, "df": None, "trained_at": None, "error": None}
_lock = threading.Lock()

# Réponses calculées, valables jusqu'au prochain ré-entraînement : l'analyse de
# toute la base n'est faite qu'une fois par question, même sur le CPU modeste
# d'une offre gratuite.
_cache: dict = {}


def _retrain() -> None:
    try:
        df = load_dataframe()
        ai = YeleAI()
        ai.train(df)
        with _lock:
            _state.update(ai=ai, df=df, trained_at=time.time(), error=None)
            _cache.clear()
    except Exception as exc:  # service utilisable même si Mongo est down
        with _lock:
            _state["error"] = str(exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Entraînement initial en arrière-plan (ne bloque pas le démarrage).
    threading.Thread(target=_retrain, daemon=True).start()
    yield


app = FastAPI(title="Yélé AI", lifespan=lifespan)


@app.get("/ai/health")
def health():
    with _lock:
        ai: YeleAI | None = _state["ai"]
        return {
            "status": "ok" if ai else ("error" if _state["error"] else "training"),
            "error": _state["error"],
            "samples": ai.n_samples if ai else 0,
            "mlReady": ai.ml_ready if ai else False,
            "minSamplesForMl": MIN_SAMPLES_ML,
            "trainedAt": _state["trained_at"],
            "detector": ai.detector_name if ai else "",
            "regressor": ai.reg_name if ai else "",
            "operators": ai.operators if ai else [],
        }


@app.get("/ai/anomalies")
def anomalies(limit: int = Query(50, ge=1, le=200),
              days: int = Query(30, ge=1, le=365)):
    with _lock:
        ai, df = _state["ai"], _state["df"]
    if ai is None or df is None or df.empty:
        return {"anomalies": [], "analyzed": 0, "mlReady": False}
    key = ("anomalies", limit, days)
    if key not in _cache:
        cutoff = df["timestamp"].max() - __import__("pandas").Timedelta(days=days)
        recent = df[df["timestamp"] >= cutoff]
        _cache[key] = {
            "anomalies": ai.detect(recent, limit=limit),
            "analyzed": int(len(recent)),
            "mlReady": ai.ml_ready,
        }
    return _cache[key]


@app.get("/ai/forecast")
def forecast(operator: str, hours: int = Query(48, ge=1, le=168)):
    with _lock:
        ai = _state["ai"]
    if ai is None:
        return {"forecast": [], "mlReady": False}
    # La prévision part de l'heure courante : la clé change à chaque heure.
    hour = datetime.now(timezone.utc).strftime("%Y%m%d%H")
    key = ("forecast", operator, hours, hour)
    if key not in _cache:
        for old in [k for k in _cache if k[0] == "forecast" and k[3] != hour]:
            del _cache[old]
        _cache[key] = {
            "operator": operator,
            "forecast": ai.forecast(operator, hours=hours),
            "mlReady": ai.ml_ready,
        }
    return _cache[key]


@app.post("/ai/retrain")
def retrain():
    _retrain()
    return health()
