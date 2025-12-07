import numpy as np
import pandas as pd

# -----------------------------
# 1) Haversine distance
# -----------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # km
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))


# -----------------------------
# 2) Catégories FR ↔ EN
# -----------------------------
translations = {
    "nature": "Nature",
    "nightlife": "Vie nocturne",
    "drink": "Bar",
    "music": "Musique",
    "dance": "Danse",
    "history": "Monument",
    "sports": "Sport",
    "art": "Art",
    "museum": "Musée",
    "walk": "Balade",
    "restaurant": "Restaurant",
    "movie": "Cinéma"
}

translations_inv = {v: k for k, v in translations.items()}


# -----------------------------
# 3) Prédiction DL (ONNX)
# -----------------------------
def predict_categories_dl(text, embedder, model_dl, threshold=0.5):
    # Embedding du texte
    vec = embedder.encode([text], convert_to_numpy=True).astype(np.float32)

    # Prédiction ONNX
    probs = model_dl.run(None, {"input": vec})[0][0]  # vecteur probas

    results = []
    for i, (cat, prob) in enumerate(zip(translations.keys(), probs)):
        if prob >= threshold:
            results.append((cat, float(prob)))

    # Si aucune catégorie détectée → garder la top-1
    if len(results) == 0:
        best_idx = np.argmax(probs)
        best_cat = list(translations.keys())[best_idx]
        results = [(best_cat, float(probs[best_idx]))]

    return results


# -----------------------------
# 4) Modèle Hybride
# -----------------------------
def recommend_hybrid(
    user_text,
    user_lat,
    user_lon,
    df,
    embedder,
    model_dl,
    threshold=0.5,
    w_cat=0.5,
    w_quality=0.3,
    w_geo=0.2,
    top_k=5
):
    # ------ DL Prediction ------
    predicted = predict_categories_dl(user_text, embedder, model_dl, threshold)
    predicted_dict = {cat: prob for cat, prob in predicted}

    predicted_cats = list(predicted_dict.keys())

    # ------ Filtrage des POI par catégorie ------
    mask = df[predicted_cats].sum(axis=1) >= 1
    subset = df[mask].copy()

    if subset.empty:
        return pd.DataFrame([])

    # ------ Distance / Score géographique ------
    subset["distance_km"] = subset.apply(
        lambda row: haversine_distance(
            user_lat, user_lon, row["lat"], row["lng"]
        ),
        axis=1
    )

    max_dist = subset["distance_km"].max()
    subset["geo_score"] = 1 - (subset["distance_km"] / max_dist)

    # ------ Score Catégorie DL ------
    def category_score(row):
        score = 0.0
        for cat, prob in predicted_dict.items():
            if row.get(cat, 0) == 1:
                score += prob
        return score

    subset["cat_score"] = subset.apply(category_score, axis=1)

    # ------ Score ML déjà présent dans df ("quality_ml") ------
    if "quality_ml" in subset.columns:
        subset["quality_score"] = subset["quality_ml"]
    else:
        subset["quality_score"] = 0.5  # fallback

    # ------ Score Final ------
    subset["final_score"] = (
        w_cat * subset["cat_score"]
        + w_quality * subset["quality_score"]
        + w_geo * subset["geo_score"]
    )

    cols = [
        "name", "address", "main_category",
        "distance_km", "cat_score",
        "quality_score", "geo_score", "final_score"
    ]

    cols = [c for c in cols if c in subset.columns]

    return subset.sort_values("final_score", ascending=False).head(top_k)[cols]
