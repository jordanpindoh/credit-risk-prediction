import json
from pathlib import Path

import joblib
import pandas as pd

from features import create_features


# ============================================================
# CHEMINS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "credit_risk_xgboost.pkl"
)

CONFIG_PATH = (
    BASE_DIR
    / "models"
    / "model_config.json"
)


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# CHARGEMENT DE LA CONFIGURATION
# ============================================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as f:
    config = json.load(f)


# Notre seuil final validé dans Colab
FINAL_THRESHOLD = 0.55


# ============================================================
# PRÉDICTION
# ============================================================

def predict_risk(client_data: dict) -> dict:

    # Données brutes
    df = pd.DataFrame([client_data])

    # Feature Engineering
    df_features = create_features(df)

    # Probabilité de défaut
    probability = model.predict_proba(
        df_features
    )[0, 1]

    # Décision selon le seuil 0.55
    prediction = int(
        probability >= FINAL_THRESHOLD
    )

    # Niveau de risque
    if probability < 0.30:

        risk_level = "Faible"

    elif probability < FINAL_THRESHOLD:

        risk_level = "Modéré"

    else:

        risk_level = "Élevé"

    return {
        "probability": float(probability),
        "prediction": prediction,
        "risk_level": risk_level,
        "threshold": FINAL_THRESHOLD,
    }