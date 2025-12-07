from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import onnxruntime as ort
import pandas as pd
from sentence_transformers import SentenceTransformer

# Import du pipeline hybride
from recommend import recommend_hybrid

app = FastAPI(title="ActiCity Hybrid Recommendation API")

# --------------------------
# Lazy-loading des assets
# --------------------------
embedder = None
model_dl = None
model_ml = None
tfidf = None
df = None

def load_assets():
    global embedder, model_dl, model_ml, tfidf, df

    # ---- 1) SentenceTransformer ----
    if embedder is None:
        embedder = SentenceTransformer("distiluse-base-multilingual-cased-v2")

    # ---- 2) ONNX Runtime ----
    if model_dl is None:
        model_dl = ort.InferenceSession("model_dl.onnx", providers=["CPUExecutionProvider"])

    # ---- 3) LogisticRegression ----
    if model_ml is None:
        with open("model_ml.pkl", "rb") as f:
            model_ml = pickle.load(f)

    # ---- 4) TF-IDF Vectorizer ----
    if tfidf is None:
        with open("tfidf.pkl", "rb") as f:
            tfidf = pickle.load(f)

    # ---- 5) DataFrame ----
    if df is None:
        with open("dataframe.pkl", "rb") as f:
            df = pickle.load(f)


# --------------------------
# Model input
# --------------------------
class Request(BaseModel):
    text: str
    lat: float
    lon: float
    top_k: int = 3


# --------------------------
# Healthcheck
# --------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "ActiCity API is running 🎉"
    }


# --------------------------
# Prediction endpoint
# --------------------------
@app.post("/predict")
def predict(req: Request):

    # Charger les modèles si ce n'est pas déjà fait
    load_assets()

    # Appel du modèle hybride
    results = recommend_hybrid(
        user_text=req.text,
        user_lat=req.lat,
        user_lon=req.lon,
        df=df,
        embedder=embedder,
        model_dl=model_dl,
        top_k=req.top_k
    )

    if results.empty:
        return {
            "status": "ok",
            "message": "Aucun résultat trouvé pour cette requête."
        }

    # Convertir en JSON propre
    results_json = results.to_dict(orient="records")

    return {
        "status": "ok",
        "results": results_json
    }
