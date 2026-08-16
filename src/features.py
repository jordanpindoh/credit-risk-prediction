import numpy as np
import pandas as pd


# ============================================================
# COLONNES DU DATASET ORIGINAL
# ============================================================

PAY_COLS = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]

BILL_COLS = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
]

PAY_AMT_COLS = [
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # ========================================================
    # 1. PAYMENT BEHAVIOR
    # ========================================================

    # Nombre de mois avec retard
    df["nb_retards"] = (
        df[PAY_COLS] > 0
    ).sum(axis=1)

    # Nombre de mois sans retard
    df["nb_mois_sans_retard"] = (
        df[PAY_COLS] <= 0
    ).sum(axis=1)

    # Nombre de retards importants >= 2
    df["nb_retards_importants"] = (
        df[PAY_COLS] >= 2
    ).sum(axis=1)

    # Retard maximum
    df["retard_max"] = (
        df[PAY_COLS]
        .clip(lower=0)
        .max(axis=1)
    )

    # Retard moyen uniquement sur les mois avec retard
    df["retard_moyen"] = (
        df[PAY_COLS]
        .clip(lower=0)
        .replace(0, np.nan)
        .mean(axis=1)
        .fillna(0)
    )

    # Retard récent
    df["retard_recent"] = (
        df["PAY_0"] > 0
    ).astype(int)

    # Retards récurrents
    df["retard_recurrent"] = (
        df["nb_retards"] >= 3
    ).astype(int)

    # ========================================================
    # 2. FINANCIAL BEHAVIOR
    # ========================================================

    # Total des factures sur 6 mois
    df["total_factures"] = (
        df[BILL_COLS].sum(axis=1)
    )

    # Total des paiements sur 6 mois
    df["total_paiements"] = (
        df[PAY_AMT_COLS].sum(axis=1)
    )

    # Facture moyenne
    df["facture_moyenne"] = (
        df[BILL_COLS].mean(axis=1)
    )

    # Paiement moyen
    df["paiement_moyen"] = (
        df[PAY_AMT_COLS].mean(axis=1)
    )

    # Facture maximale
    df["facture_max"] = (
        df[BILL_COLS].max(axis=1)
    )

    # Paiement maximal
    df["paiement_max"] = (
        df[PAY_AMT_COLS].max(axis=1)
    )

    # ========================================================
    # 3. MONTHLY PAYMENT COVERAGE
    # ========================================================

    monthly_ratios = []

    for bill_col, pay_col in zip(
        BILL_COLS,
        PAY_AMT_COLS
    ):

        ratio = np.where(
            df[bill_col].abs() > 0,
            df[pay_col] / df[bill_col].abs(),
            0
        )

        monthly_ratios.append(ratio)

    monthly_ratios = np.array(
        monthly_ratios
    ).T

    # Limitation des ratios extrêmes
    monthly_ratios = np.clip(
        monthly_ratios,
        0,
        2
    )

    df["ratio_paiement_moyen"] = (
        monthly_ratios.mean(axis=1)
    )

    df["ratio_paiement_min"] = (
        monthly_ratios.min(axis=1)
    )

    df["ratio_paiement_max"] = (
        monthly_ratios.max(axis=1)
    )

    # ========================================================
    # 4. PAYMENT TREND
    # ========================================================

    # Paiements récents
    df["paiement_recent"] = (
        df["PAY_AMT1"] +
        df["PAY_AMT2"]
    ) / 2

    # Paiements anciens
    df["paiement_ancien"] = (
        df["PAY_AMT5"] +
        df["PAY_AMT6"]
    ) / 2

    # Évolution des paiements
    df["evolution_paiement"] = (
        df["paiement_recent"] -
        df["paiement_ancien"]
    )

    return df
