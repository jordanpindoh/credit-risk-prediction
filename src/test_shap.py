import sys
from pathlib import Path

import joblib
import shap
import pandas as pd

from features import create_features


# ============================================================
# CHEMINS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "credit_risk_xgboost.pkl"


# ============================================================
# CLIENT DE TEST
# ============================================================

client = {
    "LIMIT_BAL": 120000,
    "SEX": 1,
    "EDUCATION": 2,
    "MARRIAGE": 2,
    "AGE": 35,

    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,

    "BILL_AMT1": 30000,
    "BILL_AMT2": 30000,
    "BILL_AMT3": 30000,
    "BILL_AMT4": 30000,
    "BILL_AMT5": 30000,
    "BILL_AMT6": 30000,

    "PAY_AMT1": 3000,
    "PAY_AMT2": 3000,
    "PAY_AMT3": 3000,
    "PAY_AMT4": 3000,
    "PAY_AMT5": 3000,
    "PAY_AMT6": 3000,
}


# ============================================================
# CHARGEMENT
# ============================================================

print("=" * 70)
print("              SHAP EXPLAINABILITY TEST")
print("=" * 70)

pipeline = joblib.load(MODEL_PATH)

preprocessing = pipeline.named_steps["preprocessing"]
xgb_model = pipeline.named_steps["model"]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df = pd.DataFrame([client])

df_features = create_features(df)


# ============================================================
# TRANSFORMATION
# ============================================================

X_transformed = preprocessing.transform(
    df_features
)


print()
print("Features après feature engineering :", df_features.shape)
print("Features après preprocessing :", X_transformed.shape)


# ============================================================
# NOMS DES FEATURES TRANSFORMÉES
# ============================================================

feature_names = preprocessing.get_feature_names_out()

print()
print("Nombre de features SHAP :", len(feature_names))

print()
print("Premières features :")

for i, name in enumerate(feature_names[:20]):
    print(i, ":", name)


# ============================================================
# EXPLAINER SHAP
# ============================================================

explainer = shap.TreeExplainer(
    xgb_model
)


# ============================================================
# SHAP VALUES
# ============================================================

shap_values = explainer.shap_values(
    X_transformed
)


print()
print("Shape SHAP :", shap_values.shape)


# ============================================================
# TOP FEATURES
# ============================================================

values = shap_values[0]

importance = sorted(
    zip(feature_names, values),
    key=lambda x: abs(x[1]),
    reverse=True,
)


print()
print("=" * 70)
print("TOP 10 FACTEURS EXPLIQUANT LA PREDICTION")
print("=" * 70)

for name, value in importance[:10]:

    direction = (
        "AUGMENTE le risque"
        if value > 0
        else "DIMINUE le risque"
    )

    print(
        f"{name:40s} "
        f"{value:+.6f}  →  {direction}"
    )


print()
print("=" * 70)
print("SHAP TEST TERMINÉ")
print("=" * 70)
