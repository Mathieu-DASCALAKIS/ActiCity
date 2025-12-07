from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
import tensorflow as tf

app = FastAPI(title="ActiCity Hybrid Recommendation API")

# --------------------------
#  Lazy loading des modèles
# --------------------------
embedder = None
model_dl = None
model_ml = None
tfidf = None
df = None

def load_assets():
    global embedder, model_dl, model_ml, tfidf, df

    if embedder is None:
        embedder = SentenceTransformer("distiluse-base-multilingual-cased-v2")

    if model_dl is None:
        model_dl = tf.keras.models.load_model("model_dl.h5")

    if model_ml is None:
        with open("model_ml.pkl", "rb") as f:
            model_ml = pickle.load(f)

    if tfidf is None:
        with open("tfidf.pkl", "rb") as f:
            tfidf = pickle.load(f)

    if df is None:
        with open("dataframe.pkl", "rb") as f:
            df = pickle.load(f)

# -----------------------------
# ROUTE DE TEST / HEALTHCHECK
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "ActiCity API is running"}

# -----------------------------
# INPUT MODEL
# -----------------------------
class Request(BaseModel):
    text: str
    lat: float
    lon: float

# -----------------------------
# ROUTE DE PRÉDICTION
# -----------------------------
@app.post("/predict")
def predict(req: Request):

    # Charger tout à la première requête
    load_assets()

    # TODO : appeler ici ta fonction recommend_hybrid()
    # Exemple :
    # results = recommend_hybrid(req.text, req.lat, req.lon, df)

    return {
        "status": "ok",
        "message": "Prediction endpoint ready",
        "note": "Intègre maintenant recommend_hybrid() ici"
    }
