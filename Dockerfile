# Service IA Yélé — déployable sur Render comme "Web Service" Docker.
# Variables d'environnement requises :
#   SPEEDTEST_DATABASE_CONNECTION_STRING (MongoDB Atlas)
#   SPEEDTEST_DATABASE_NAME (défaut: yele_speedtest)
FROM python:3.12-slim

WORKDIR /app

COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY model.py service.py ./

# Détection : LOF (pyod) ; prédiction : Gradient Boosting (scikit-learn).
# Ni PyTorch ni poids à télécharger : l'image reste légère.
ENV PYTHONUNBUFFERED=1

# Render fournit $PORT ; 8000 par défaut en local.
CMD ["sh", "-c", "uvicorn service:app --host 0.0.0.0 --port ${PORT:-8000}"]
