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
# 2) Categories FR ↔ EN
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
# 3) DL Prediction (ONNX)
# -----------------------------
def predict_categories_dl(text, embedder, model_dl, threshold=0.5):
    vec = embedder.encode([text], convert_to_numpy=True).astype(np.float32)

    input_name = model_dl.get_inputs()[0].name
    probs = model_dl.run(None, {input_name: vec})[0][0]

    results = []
    for cat, prob in zip(translations.keys(), probs):
        if prob >= threshold:
            results.append((cat, float(prob)))

    # fallback : top-1
    if not results:
        idx = np.argmax(probs)
        best_cat = list(translations.keys())[idx]
        results = [(best_cat, float(probs[idx]))]

    return results


# -----------------------------
# 4) Hybrid recommendation
# -----------------------------
def recommend_hybrid(
    text,
    user_lat,
    user_lon,
    df,
    ml_pred,
    dl_pred,
    w_cat=0.5,
    w_quality=0.3,
    w_geo=0.2,
    top_k=5
):
    # ML returns a vector of probs (one per POI or per label depending on training)
    # DL returns category probs

    dl_dict = {cat: float(prob) for cat, prob in zip(translations.keys(), dl_pred)}

    # Determine categories selected by DL
    predicted_cats = [cat for cat, prob in dl_dict.items() if prob >= 0.5]
    if not predicted_cats:
        predicted_cats = [max(dl_dict, key=dl_dict.get)]

    # Filter dataset by matching categories
    mask = df[predicted_cats].sum(axis=1) >= 1
    subset = df[mask].copy()
    if subset.empty:
        return []

    # Distance score
    subset["distance_km"] = subset.apply(
        lambda row: haversine_distance(user_lat, user_lon, row["lat"], row["lng"]),
        axis=1,
    )

    max_dist = subset["distance_km"].max()
    subset["geo_score"] = 1 - (subset["distance_km"] / max_dist)

    # DL category score
    def score_dl(row):
        return sum(dl_dict.get(cat, 0) for cat in predicted_cats if row.get(cat, 0) == 1)

    subset["cat_score"] = subset.apply(score_dl, axis=1)

    # ML score (already computed outside)
    subset["quality_score"] = ml_pred[: len(subset)]

    # Final score
    subset["final_score"] = (
        w_cat * subset["cat_score"]
        + w_quality * subset["quality_score"]
        + w_geo * subset["geo_score"]
    )

    # Columns to return
    cols = [
        "name", "address", "main_category",
        "distance_km", "cat_score",
        "quality_score", "geo_score", "final_score"
    ]

    cols = [c for c in cols if c in subset.columns]

    return subset.sort_values("final_score", ascending=False).head(top_k)[cols].to_dict(orient="records")
