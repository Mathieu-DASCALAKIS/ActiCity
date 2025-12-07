from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
from geopy.distance import geodesic
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model

# -------------------------------
# Chargement des ressources
# -------------------------------
app = FastAPI(title="ActiCity Hybrid Recommendation API")

df = pd.read_pickle("dataframe.pkl")
model_dl = load_model("model_dl.h5")

with open("model_ml.pkl", "rb") as f:
    model_ml = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

embedder = SentenceTransformer("distiluse-base-multilingual-cased-v2")

translations = {
    "museum": "musée",
    "art": "art",
    "nightlife": "vie nocturne",
    "drink": "boisson",
    "restaurant": "restaurant",
    "walk": "promenade",
    "nature": "nature",
    "music": "musique",
    "dance": "danse",
    "history": "histoire",
    "sports": "sport"
}
to_french = translations

clean_cols = list(translations.keys())

# -------------------------------
# Fonction distance Haversine
# -------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km


# -------------------------------
# Endpoint de recommandation
# -------------------------------
@app.post("/recommend")
def recommend(user_text: str, user_lat: float, user_lon: float,
              top_k: int = 3, threshold: float = 0.5,
              w_cat: float = 0.3, w_quality: float = 0.8, w_geo: float = 0.5):
    
    # 1️⃣ → Embedding du texte utilisateur
    emb = embedder.encode([user_text], convert_to_numpy=True)

    # 2️⃣ → Prédiction DL des catégories
    dl_pred = model_dl.predict(emb)[0]
    
    predicted = [(clean_cols[i], float(dl_pred[i])) for i in range(len(clean_cols)) if dl_pred[i] >= threshold]
    predicted_dict = dict(predicted)

    if not predicted_dict:
        return {"message": "Aucune catégorie détectée dans la requête."}

    # 3️⃣ → Filtrage des POI matching categories
    mask = df[list(predicted_dict.keys())].sum(axis=1) >= 1
    subset = df[mask].copy()
    
    if subset.empty:
        return {"message": "Aucun POI ne correspond aux catégories détectées."}

    # 4️⃣ → Distance géographique
    subset["distance_km"] = subset.apply(
        lambda row: haversine_distance(user_lat, user_lon, row["lat"], row["lng"]),
        axis=1
    )
    max_dist = subset["distance_km"].max()
    subset["geo_score"] = 1 - (subset["distance_km"] / max_dist)

    # 5️⃣ → Score catégorie
    def compute_cat_score(row):
        score = 0
        for c, p in predicted_dict.items():
            if row[c] == 1:
                score += p
        return score
    subset["cat_score"] = subset.apply(compute_cat_score, axis=1)

    # 6️⃣ → Score qualité (ML)
    subset["quality_score"] = subset["quality_ml"].fillna(0.5)

    # 7️⃣ → Score final
    subset["final_score"] = (
        w_cat * subset["cat_score"] +
        w_quality * subset["quality_score"] +
        w_geo * subset["geo_score"]
    )

    # 8️⃣ → Top-K résultats
    results = subset.sort_values("final_score", ascending=False).head(top_k)

    # 9️⃣ → Format JSON
    output = []
    for _, row in results.iterrows():
        output.append({
            "name": row["name"],
            "address": row["address"],
            "main_category": row["main_category"],
            "distance_km": float(row["distance_km"]),
            "cat_score": float(row["cat_score"]),
            "quality_score": float(row["quality_score"]),
            "geo_score": float(row["geo_score"]),
            "final_score": float(row["final_score"])
        })

    return {"results": output}
