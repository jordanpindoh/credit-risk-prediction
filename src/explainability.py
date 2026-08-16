

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

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


# ============================================================
# CHARGEMENT DU PIPELINE
# ============================================================

pipeline = joblib.load(MODEL_PATH)

preprocessing = pipeline.named_steps["preprocessing"]
xgb_model = pipeline.named_steps["model"]


# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(
    xgb_model
)


# ============================================================
# NOMS LISIBLES
# ============================================================

FEATURE_LABELS = {

    "LIMIT_BAL": "Limite de crédit",

    "SEX": "Sexe",

    "EDUCATION": "Niveau d'éducation",

    "MARRIAGE": "Situation familiale",

    "AGE": "Âge",

    "PAY_0": "Retard le plus récent",

    "PAY_2": "Retard précédent",

    "PAY_3": "Retard à M-3",

    "PAY_4": "Retard à M-4",

    "PAY_5": "Retard à M-5",

    "PAY_6": "Retard à M-6",

    "BILL_AMT1": "Facture récente",

    "BILL_AMT2": "Facture M-2",

    "BILL_AMT3": "Facture M-3",

    "BILL_AMT4": "Facture M-4",

    "BILL_AMT5": "Facture M-5",

    "BILL_AMT6": "Facture ancienne",

    "PAY_AMT1": "Paiement récent",

    "PAY_AMT2": "Paiement M-2",

    "PAY_AMT3": "Paiement M-3",

    "PAY_AMT4": "Paiement M-4",

    "PAY_AMT5": "Paiement M-5",

    "PAY_AMT6": "Paiement ancien",

    "nb_retards": "Nombre de retards",

    "retard_max": "Retard maximum",

    "retard_moyen": "Retard moyen",

    "total_factures": "Total des factures",

    "total_paiements": "Total des paiements",

    "ratio_paiement": "Ratio de paiement",

    "facture_moyenne": "Facture moyenne",

    "paiement_moyen": "Paiement moyen",

    "facture_max": "Facture maximale",

    "paiement_max": "Paiement maximal",

    "ratio_paiement_moyen": "Ratio de paiement moyen",

    "ratio_paiement_min": "Ratio de paiement minimum",

    "ratio_paiement_max": "Ratio de paiement maximum",

    "paiement_recent": "Paiement récent moyen",

    "paiement_ancien": "Paiement ancien moyen",

    "evolution_paiement": "Évolution des paiements",

    "nb_paiements_normaux": "Mois sans retard",

    "nb_retards_importants": "Retards importants",

    "retard_recent": "Retard récent",

    "retard_recurrent": "Retards récurrents",

    "nb_mois_sans_retard": "Mois sans retard",
}


# ============================================================
# NETTOYAGE DU NOM DES FEATURES
# ============================================================

def clean_feature_name(name: str) -> str:

    name = str(name)

    if "__" in name:
        name = name.split("__", 1)[1]

    return FEATURE_LABELS.get(
        name,
        name.replace("_", " ").capitalize()
    )


# ============================================================
# EXPLICATION SHAP
# ============================================================

def explain_client(client_data: dict) -> dict:

    # --------------------------------------------------------
    # Données brutes
    # --------------------------------------------------------

    df = pd.DataFrame([client_data])

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    df_features = create_features(df)

    # --------------------------------------------------------
    # Préprocessing
    # --------------------------------------------------------

    X_transformed = preprocessing.transform(
        df_features
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    shap_values = explainer.shap_values(
        X_transformed
    )

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:

        shap_values = shap_values[0, :, 1]

    elif shap_values.ndim == 2:

        shap_values = shap_values[0]

    else:

        shap_values = shap_values.reshape(-1)

    # --------------------------------------------------------
    # Noms des features
    # --------------------------------------------------------

    feature_names = (
        preprocessing
        .get_feature_names_out()
    )

    # --------------------------------------------------------
    # Marge XGBoost
    # --------------------------------------------------------

    margin = float(
        xgb_model.predict(
            X_transformed,
            output_margin=True
        )[0]
    )

    # --------------------------------------------------------
    # Probabilité
    # --------------------------------------------------------

    probability = float(
        pipeline.predict_proba(
            df_features
        )[0, 1]
    )

    # --------------------------------------------------------
    # Contributions
    # --------------------------------------------------------

    contributions = []

    for name, value in zip(
        feature_names,
        shap_values
    ):

        contributions.append(
            {
                "feature": str(name),
                "label": clean_feature_name(name),
                "shap_value": float(value),
                "abs_shap": float(abs(value)),
            }
        )

    # --------------------------------------------------------
    # Tri
    # --------------------------------------------------------

    contributions.sort(
        key=lambda x: x["abs_shap"],
        reverse=True
    )

    # --------------------------------------------------------
    # Facteurs aggravants
    # --------------------------------------------------------

    risk_factors = [
        x
        for x in contributions
        if x["shap_value"] > 0
    ]

    # --------------------------------------------------------
    # Facteurs protecteurs
    # --------------------------------------------------------

    protective_factors = [
        x
        for x in contributions
        if x["shap_value"] < 0
    ]

    return {
        "probability": probability,
        "margin": margin,
        "contributions": contributions,
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "feature_count": len(feature_names),
    }