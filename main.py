from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
import onnxruntime as ort
import numpy as np

from recommend import recommend_hybrid

app = FastAPI(title="ActiCity Hybrid Recommendation API")

# --------------------------
#  Lazy-loading des modèles
# --------------------------
embedder = None
model_dl = None
model_ml = None
tfidf = None
df = None

def load_assets():
    global embedder, model_dl, model_ml, tfidf, df

    # --- SentenceTransformer ---
    if embedder is None:
        embedder = SentenceTransformer("distiluse-base-multilingual-cased-v2")

    # --- Deep Learning ONNX model ---
    if model_dl is None:
        model_dl = ort.InferenceSession("model_dl.onnx")

    # --- Logistic Regression ---
    if model_ml is None:
        with open("model_ml.pkl", "rb") as f:
            model_ml = pickle.load(f)

    # --- TF-IDF ---
    if tfidf is None:
        with open("tfidf.pkl", "rb") as f:
            tfidf = pickle.load(f)

    # --- DataFrame ---
    if df is None:
        with open("dataframe.pkl", "rb") as f:
            df = pickle.load(f)

# -----------------------------
# ROUTE HEALTHCHECK
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

    load_assets()

    # On calcule les embeddings via SentenceTransformer
    text_vector = embedder.encode(req.text)

    # ONNX : exécution modèle DL
    input_name = model_dl.get_inputs()[0].name
    dl_pred = model_dl.run(None, {input_name: text_vector.astype(np.float32).reshape(1, -1)})[0][0]

    # ML : prédiction TF-IDF → LogisticRegression
    X_tfidf = tfidf.transform([req.text]).toarray()
    ml_pred = model_ml.predict_proba(X_tfidf)[0]

    # Appel modèle hybride
    results = recommend_hybrid(
        text=req.text,
        user_lat=req.lat,
        user_lon=req.lon,
        df=df,
        ml_pred=ml_pred,
        dl_pred=dl_pred
    )

    return {
        "status": "ok",
        "results": results
    }
