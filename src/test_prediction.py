
from prediction import predict_risk


# ============================================================
# CLIENT DE TEST
# ============================================================

client = {

    "LIMIT_BAL": 120000,

    "SEX": 2,

    "EDUCATION": 2,

    "MARRIAGE": 2,

    "AGE": 34,

    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,

    "BILL_AMT1": 30000,
    "BILL_AMT2": 28000,
    "BILL_AMT3": 26000,
    "BILL_AMT4": 24000,
    "BILL_AMT5": 22000,
    "BILL_AMT6": 20000,

    "PAY_AMT1": 3000,
    "PAY_AMT2": 3000,
    "PAY_AMT3": 3000,
    "PAY_AMT4": 3000,
    "PAY_AMT5": 3000,
    "PAY_AMT6": 3000,
}


# ============================================================
# PRÉDICTION
# ============================================================

result = predict_risk(client)


# ============================================================
# AFFICHAGE
# ============================================================

print()
print("=" * 50)
print("       CREDIT RISK PREDICTION TEST")
print("=" * 50)

print(
    f"Probabilité de défaut : "
    f"{result['probability'] * 100:.2f}%"
)

print(
    f"Décision : "
    f"{result['prediction']}"
)

print(
    f"Niveau de risque : "
    f"{result['risk_level']}"
)

print(
    f"Seuil de décision : "
    f"{result['threshold']:.2f}"
)

print("=" * 50)