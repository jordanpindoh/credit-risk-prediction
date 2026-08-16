

# ============================================================
# CREDIT RISK INTELLIGENCE
# APPLICATION STREAMLIT COMPLÈTE
# ============================================================

import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CHEMINS DU PROJET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))




    # ============================================================
# IMPORTS DU PROJET
# ============================================================

try:
    from prediction import predict_risk
except Exception as e:
    predict_risk = None
    PREDICTION_ERROR = str(e)


try:
    from explainability import explain_client
except Exception as e:
    explain_client = None
    EXPLAINABILITY_ERROR = str(e)


try:
    from features import create_features
except Exception as e:
    create_features = None
    FEATURES_ERROR = str(e)


# ============================================================
# MODÈLE
# ============================================================

MODEL_PATH = MODELS_DIR / "credit_risk_xgboost.pkl"
CONFIG_PATH = MODELS_DIR / "model_config.json"

MODEL_AVAILABLE = False
model = None

try:
    model = joblib.load(MODEL_PATH)
    MODEL_AVAILABLE = True
except Exception as e:
    MODEL_LOAD_ERROR = str(e)


# ============================================================
# CONFIGURATION DU MODÈLE
# ============================================================

MODEL_CONFIG = {
    "model": "XGBoost",
    "threshold": 0.55,
    "target": "default payment next month",
    "random_state": 42,
}

try:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            MODEL_CONFIG.update(json.load(f))
except Exception:
    pass


MODEL_NAME = MODEL_CONFIG.get("model", "XGBoost")
FINAL_THRESHOLD = float(
    MODEL_CONFIG.get("threshold", 0.55)
)


# ============================================================
# COLONNES DU DATASET
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
# VALEURS PAR DÉFAUT
# ============================================================

DEFAULT_CLIENT = {
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
# FONCTIONS UTILITAIRES
# ============================================================

def get_risk_color(probability):
    if probability < 0.30:
        return "🟢"
    elif probability < FINAL_THRESHOLD:
        return "🟠"
    elif probability < 0.75:
        return "🔴"
    return "🚨"


def get_risk_label(probability):
    if probability < 0.30:
        return "Faible"
    elif probability < FINAL_THRESHOLD:
        return "Modéré"
    elif probability < 0.75:
        return "Élevé"
    return "Très élevé"


def get_decision(probability):
    if probability >= FINAL_THRESHOLD:
        return "Défaut potentiel"
    return "Risque acceptable"


def predict_client(client_data):
    if predict_risk is None:
        return None

    try:
        return predict_risk(client_data)
    except Exception as e:
        st.error("Erreur pendant la prédiction.")
        st.exception(e)
        return None


def format_probability(value):
    return f"{float(value) * 100:.2f} %"


def build_client_from_form():
    return {
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,

        "PAY_0": pay_0,
        "PAY_2": pay_2,
        "PAY_3": pay_3,
        "PAY_4": pay_4,
        "PAY_5": pay_5,
        "PAY_6": pay_6,

        "BILL_AMT1": bill_amt1,
        "BILL_AMT2": bill_amt2,
        "BILL_AMT3": bill_amt3,
        "BILL_AMT4": bill_amt4,
        "BILL_AMT5": bill_amt5,
        "BILL_AMT6": bill_amt6,

        "PAY_AMT1": pay_amt1,
        "PAY_AMT2": pay_amt2,
        "PAY_AMT3": pay_amt3,
        "PAY_AMT4": pay_amt4,
        "PAY_AMT5": pay_amt5,
        "PAY_AMT6": pay_amt6,
    }


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 35px;
    }

    .section-title {
        font-size: 30px;
        font-weight: 750;
        margin-top: 10px;
        margin-bottom: 25px;
    }

    .kpi-card {
        background: #20212a;
        border-radius: 14px;
        padding: 22px;
        min-height: 130px;
        border: 1px solid #30313b;
    }

    .kpi-label {
        color: #a7a9b5;
        font-size: 15px;
    }

    .kpi-value {
        font-size: 31px;
        font-weight: 750;
        margin-top: 8px;
    }

    .kpi-description {
        color: #8f929e;
        font-size: 13px;
        margin-top: 6px;
    }

    .risk-card {
        padding: 22px;
        border-radius: 14px;
        background: #20212a;
        border: 1px solid #30313b;
        margin-top: 15px;
    }

    .risk-value {
        font-size: 40px;
        font-weight: 800;
    }

    .risk-label {
        font-size: 18px;
        color: #a7a9b5;
    }

    .info-card {
        background: #20212a;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #30313b;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        # 💳 Credit Risk
        ## Intelligence

        AI-powered credit risk analytics platform
        """
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "👤 Analyse client",
            "🔮 Simulation",
            "🧠 Explainable AI",
            "📊 Portefeuille",
            "📈 Monitoring",
        ],
    )

    st.divider()

    st.markdown("### 🤖 Modèle")

    st.caption(
        f"Algorithm : {MODEL_NAME}"
    )

    st.caption(
        "Features : 48"
    )

    st.caption(
        f"Decision threshold : {FINAL_THRESHOLD:.0%}"
    )

    if MODEL_AVAILABLE:
        st.success("● Modèle disponible")
    else:
        st.error("● Modèle indisponible")


# ============================================================
# HEADER GLOBAL
# ============================================================

st.markdown(
    '<div class="main-title">Credit Risk Intelligence</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Plateforme intelligente d'analyse, de prédiction,
    de simulation et de monitoring du risque de crédit.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD
# ============================================================


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">📊 Vue d’ensemble</div>',
        unsafe_allow_html=True,
    )

    # ========================================================
    # INDICATEURS CLÉS DU MODÈLE
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Clients de référence</div>
                <div class="kpi-value">30 000</div>
                <div class="kpi-description">
                    Dataset utilisé pour le modèle
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Taux de défaut de référence</div>
                <div class="kpi-value">22,12 %</div>
                <div class="kpi-description">
                    Population du dataset d'entraînement
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Seuil de décision</div>
                <div class="kpi-value">
                    {FINAL_THRESHOLD:.0%}
                </div>
                <div class="kpi-description">
                    Seuil validé du modèle
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Algorithme</div>
                <div class="kpi-value">XGBoost</div>
                <div class="kpi-description">
                    Classification du risque
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ========================================================
    # ARCHITECTURE DE LA PLATEFORME
    # ========================================================

    st.markdown(
        "### 🎯 Architecture de la plateforme"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            """
            **🔍 Analyse**

            Évaluation individuelle du profil
            et estimation de la probabilité
            de défaut.
            """
        )

    with col2:
        st.info(
            """
            **🔮 Simulation**

            Modification des caractéristiques
            d'un client et mesure de l'impact
            sur le risque.
            """
        )

    with col3:
        st.info(
            """
            **🧠 Explainable AI**

            Explication des prédictions
            individuelles avec SHAP.
            """
        )

    st.divider()

    # ========================================================
    # NIVEAUX DE RISQUE
    # ========================================================

    st.markdown(
        "### 📈 Niveaux de risque"
    )

    risk_data = pd.DataFrame(
        {
            "Niveau": [
                "Faible",
                "Modéré",
                "Élevé",
                "Très élevé",
            ],
            "Probabilité": [
                30,
                55,
                75,
                100,
            ],
        }
    )

    fig = px.bar(
        risk_data,
        x="Niveau",
        y="Probabilité",
        text="Probabilité",
        title="Zones indicatives de risque",
    )

    fig.update_layout(
        height=430,
        yaxis_title="Probabilité maximale (%)",
        xaxis_title="",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )


# ============================================================
# ANALYSE CLIENT
# ============================================================



# ============================================================
# ANALYSE CLIENT
# ============================================================

elif page == "👤 Analyse client":

    st.markdown(
        '<div class="section-title">👤 Analyse d’un client</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        Saisissez les caractéristiques financières et
        comportementales du client afin d'estimer sa
        probabilité de défaut.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # PROFIL
    # --------------------------------------------------------

    st.markdown("### 👤 Profil du client")

    c1, c2, c3 = st.columns(3)

    with c1:
        limit_bal = st.number_input(
            "Limite de crédit",
            min_value=0,
            max_value=1000000,
            value=DEFAULT_CLIENT["LIMIT_BAL"],
            step=1000,
        )

    with c2:
        age = st.number_input(
            "Âge",
            min_value=18,
            max_value=100,
            value=DEFAULT_CLIENT["AGE"],
        )

    with c3:
        sex = st.selectbox(
            "Sexe",
            [1, 2],
            index=DEFAULT_CLIENT["SEX"] - 1,
        )

    c1, c2 = st.columns(2)

    with c1:
        education = st.selectbox(
            "Éducation",
            [1, 2, 3, 4],
            index=DEFAULT_CLIENT["EDUCATION"] - 1,
        )

    with c2:
        marriage = st.selectbox(
            "Situation matrimoniale",
            [1, 2, 3],
            index=DEFAULT_CLIENT["MARRIAGE"] - 1,
        )

    # --------------------------------------------------------
    # RETARDS
    # --------------------------------------------------------

    st.markdown("### ⚠️ Historique des retards")

    pay_values = {}

    cols = st.columns(6)

    for i, name in enumerate(PAY_COLS):

        with cols[i]:

            pay_values[name] = st.number_input(
                name,
                min_value=-2,
                max_value=8,
                value=DEFAULT_CLIENT[name],
                step=1,
            )

    # --------------------------------------------------------
    # FACTURES
    # --------------------------------------------------------

    st.markdown("### 💳 Factures mensuelles")

    bill_values = {}

    cols = st.columns(6)

    for i, name in enumerate(BILL_COLS):

        with cols[i]:

            bill_values[name] = st.number_input(
                name,
                min_value=-100000,
                max_value=1000000,
                value=DEFAULT_CLIENT[name],
                step=1000,
            )

    # --------------------------------------------------------
    # PAIEMENTS
    # --------------------------------------------------------

    st.markdown("### 💰 Paiements mensuels")

    payment_values = {}

    cols = st.columns(6)

    for i, name in enumerate(PAY_AMT_COLS):

        with cols[i]:

            payment_values[name] = st.number_input(
                name,
                min_value=0,
                max_value=1000000,
                value=DEFAULT_CLIENT[name],
                step=500,
            )

    client_data = {
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        **pay_values,
        **bill_values,
        **payment_values,
    }

    st.divider()

    if st.button(
        "🚀 ANALYSER LE RISQUE",
        type="primary",
    ):

        result = predict_client(client_data)

        if result is not None:

            probability = result["probability"]

            risk_level = get_risk_label(
                probability
            )

            decision = get_decision(
                probability
            )

            st.session_state[
                "last_client"
            ] = client_data

            st.session_state[
                "last_prediction"
            ] = result

            st.markdown(
                "## 📊 Résultat de l’analyse"
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Probabilité de défaut",
                    format_probability(
                        probability
                    ),
                )

            with c2:
                st.metric(
                    "Seuil",
                    f"{FINAL_THRESHOLD:.0%}",
                )

            with c3:
                st.metric(
                    "Décision",
                    decision,
                )

            if risk_level == "Faible":
                st.success(
                    f"🟢 Niveau de risque : {risk_level}"
                )

            elif risk_level == "Modéré":
                st.warning(
                    f"🟠 Niveau de risque : {risk_level}"
                )

            else:
                st.error(
                    f"🔴 Niveau de risque : {risk_level}"
                )

            # ------------------------------------------------
            # JAUGE
            # ------------------------------------------------

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    title={
                        "text": "Probabilité de défaut (%)"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },
                        "threshold": {
                            "line": {
                                "color": "red",
                                "width": 4,
                            },
                            "value": FINAL_THRESHOLD * 100,
                        },
                    },
                )
            )

            fig.update_layout(
                height=430
            )

            st.plotly_chart(fig)

            # ------------------------------------------------
            # ANALYSE COMPORTEMENTALE
            # ------------------------------------------------

            st.markdown(
                "### 🔎 Analyse comportementale"
            )

            nb_retards = sum(
                1
                for x in pay_values.values()
                if x > 0
            )

            retard_max = max(
                [max(0, x) for x in pay_values.values()]
            )

            nb_retards_importants = sum(
                1
                for x in pay_values.values()
                if x >= 2
            )

            nb_mois_sans_retard = sum(
                1
                for x in pay_values.values()
                if x <= 0
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Nombre de retards",
                    nb_retards,
                )

            with c2:
                st.metric(
                    "Retard maximum",
                    retard_max,
                )

            with c3:
                st.metric(
                    "Retards importants",
                    nb_retards_importants,
                )

            with c4:
                st.metric(
                    "Mois sans retard",
                    nb_mois_sans_retard,
                )


                # ============================================================
# SIMULATION WHAT-IF
# ============================================================

elif page == "🔮 Simulation":

    st.markdown(
        '<div class="section-title">🔮 Simulation What-if</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        Modifiez les caractéristiques du client et mesurez
        l'impact de chaque scénario sur la probabilité de défaut.
        """
    )

    st.divider()

    st.markdown("### 👤 Profil du client")

    c1, c2, c3 = st.columns(3)

    with c1:
        sim_limit = st.number_input(
            "Limite de crédit",
            min_value=0,
            max_value=1000000,
            value=DEFAULT_CLIENT["LIMIT_BAL"],
            step=1000,
            key="sim_limit",
        )

    with c2:
        sim_age = st.number_input(
            "Âge",
            min_value=18,
            max_value=100,
            value=DEFAULT_CLIENT["AGE"],
            key="sim_age",
        )

    with c3:
        sim_sex = st.selectbox(
            "Sexe",
            [1, 2],
            index=DEFAULT_CLIENT["SEX"] - 1,
            key="sim_sex",
        )

    c1, c2 = st.columns(2)

    with c1:
        sim_education = st.selectbox(
            "Éducation",
            [1, 2, 3, 4],
            index=DEFAULT_CLIENT["EDUCATION"] - 1,
            key="sim_education",
        )

    with c2:
        sim_marriage = st.selectbox(
            "Situation matrimoniale",
            [1, 2, 3],
            index=DEFAULT_CLIENT["MARRIAGE"] - 1,
            key="sim_marriage",
        )

    # --------------------------------------------------------
    # RETARDS
    # --------------------------------------------------------

    st.markdown("### ⚠️ Historique des retards")

    sim_pay = {}

    cols = st.columns(6)

    for i, name in enumerate(PAY_COLS):

        with cols[i]:

            sim_pay[name] = st.number_input(
                name,
                min_value=-2,
                max_value=8,
                value=DEFAULT_CLIENT[name],
                step=1,
                key=f"sim_{name}",
            )

    # --------------------------------------------------------
    # FACTURES
    # --------------------------------------------------------

    st.markdown("### 💳 Factures mensuelles")

    sim_bills = {}

    cols = st.columns(6)

    for i, name in enumerate(BILL_COLS):

        with cols[i]:

            sim_bills[name] = st.number_input(
                name,
                min_value=-100000,
                max_value=1000000,
                value=DEFAULT_CLIENT[name],
                step=1000,
                key=f"sim_{name}",
            )

    # --------------------------------------------------------
    # PAIEMENTS
    # --------------------------------------------------------

    st.markdown("### 💰 Paiements mensuels")

    sim_payments = {}

    cols = st.columns(6)

    for i, name in enumerate(PAY_AMT_COLS):

        with cols[i]:

            sim_payments[name] = st.number_input(
                name,
                min_value=0,
                max_value=1000000,
                value=DEFAULT_CLIENT[name],
                step=500,
                key=f"sim_{name}",
            )

    simulated_client = {
        "LIMIT_BAL": sim_limit,
        "SEX": sim_sex,
        "EDUCATION": sim_education,
        "MARRIAGE": sim_marriage,
        "AGE": sim_age,
        **sim_pay,
        **sim_bills,
        **sim_payments,
    }

    st.divider()

    if st.button(
        "🚀 CALCULER LE NOUVEAU RISQUE",
        type="primary",
    ):

        result = predict_client(
            simulated_client
        )

        if result is not None:

            probability = result["probability"]

            st.session_state[
                "simulation_result"
            ] = result

            st.session_state[
                "simulation_client"
            ] = simulated_client

    if "simulation_result" in st.session_state:

        result = st.session_state[
            "simulation_result"
        ]

        probability = result["probability"]

        st.markdown(
            "## 📊 Résultat de la simulation"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Probabilité de défaut",
                format_probability(
                    probability
                ),
            )

        with c2:
            st.metric(
                "Seuil",
                f"{FINAL_THRESHOLD:.0%}",
            )

        with c3:
            st.metric(
                "Décision",
                get_decision(probability),
            )

        risk_level = get_risk_label(
            probability
        )

        if risk_level == "Faible":
            st.success(
                f"🟢 Niveau de risque : {risk_level}"
            )

        elif risk_level == "Modéré":
            st.warning(
                f"🟠 Niveau de risque : {risk_level}"
            )

        else:
            st.error(
                f"🔴 Niveau de risque : {risk_level}"
            )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={
                    "text": "Probabilité de défaut (%)"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "threshold": {
                        "line": {
                            "color": "red",
                            "width": 4,
                        },
                        "value": FINAL_THRESHOLD * 100,
                    },
                },
            )
        )

        fig.update_layout(
            height=450
        )

        st.plotly_chart(fig)

        st.markdown(
            "### 🔎 Analyse comportementale"
        )

        client = st.session_state[
            "simulation_client"
        ]

        nb_retards = sum(
            1
            for x in PAY_COLS
            if client[x] > 0
        )

        retard_max = max(
            [
                max(0, client[x])
                for x in PAY_COLS
            ]
        )

        nb_retards_importants = sum(
            1
            for x in PAY_COLS
            if client[x] >= 2
        )

        nb_mois_sans_retard = sum(
            1
            for x in PAY_COLS
            if client[x] <= 0
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Nombre de retards",
                nb_retards,
            )

        with c2:
            st.metric(
                "Retard maximum",
                retard_max,
            )

        with c3:
            st.metric(
                "Retards importants",
                nb_retards_importants,
            )

        with c4:
            st.metric(
                "Mois sans retard",
                nb_mois_sans_retard,
            )

        st.markdown(
            "### 📍 Positionnement du risque"
        )

        risk_zones = pd.DataFrame(
            {
                "Niveau": [
                    "Faible",
                    "Modéré",
                    "Élevé",
                    "Très élevé",
                ],
                "Probabilité": [
                    30,
                    55,
                    75,
                    100,
                ],
            }
        )

        fig = px.bar(
            risk_zones,
            x="Niveau",
            y="Probabilité",
            text="Probabilité",
            title="Zones de risque",
        )

        fig.add_hline(
            y=probability * 100,
            line_dash="dash",
            annotation_text=(
                f"Client : {probability * 100:.2f}%"
            ),
        )

        fig.update_layout(
            height=450,
            yaxis_title="Probabilité maximale",
        )

        st.plotly_chart(fig)


# ============================================================
# EXPLAINABLE AI
# ============================================================

elif page == "🧠 Explainable AI":

    st.markdown(
        '<div class="section-title">🧠 Explainable AI</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        Cette section explique les prédictions individuelles
        du modèle XGBoost grâce à SHAP.
        """
    )

    if explain_client is None:

        st.error(
            "Le module Explainable AI n'est pas disponible."
        )

        if "EXPLAINABILITY_ERROR" in globals():
            st.code(
                EXPLAINABILITY_ERROR
            )

    else:

        client = st.session_state.get(
            "last_client",
            DEFAULT_CLIENT.copy(),
        )

        if st.button(
            "🧠 EXPLIQUER LA PRÉDICTION",
            type="primary",
        ):

            try:

                explanation = explain_client(
                    client
                )

                st.session_state[
                    "shap_explanation"
                ] = explanation

            except Exception as e:

                st.error(
                    "Erreur pendant l'explication SHAP."
                )

                st.exception(e)

        if "shap_explanation" in st.session_state:

            explanation = st.session_state[
                "shap_explanation"
            ]

            probability = explanation[
                "probability"
            ]

            st.markdown(
                "### 📊 Résumé de la prédiction"
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Probabilité",
                    format_probability(
                        probability
                    ),
                )

            with c2:
                st.metric(
                    "Features SHAP",
                    explanation[
                        "feature_count"
                    ],
                )

            with c3:
                st.metric(
                    "Marge XGBoost",
                    f"{explanation['margin']:.4f}",
                )

            st.divider()

            # ------------------------------------------------
            # FACTEURS DE RISQUE
            # ------------------------------------------------

            st.markdown(
                "### 🔴 Facteurs augmentant le risque"
            )

            risk_factors = explanation[
                "risk_factors"
            ][:10]

            if risk_factors:

                risk_df = pd.DataFrame(
                    {
                        "Facteur": [
                            x["label"]
                            for x in risk_factors
                        ],
                        "Contribution SHAP": [
                            x["shap_value"]
                            for x in risk_factors
                        ],
                    }
                )

                fig = px.bar(
                    risk_df,
                    x="Contribution SHAP",
                    y="Facteur",
                    orientation="h",
                    text_auto=".3f",
                    title=(
                        "Facteurs aggravants"
                    ),
                )

                st.plotly_chart(fig)

            else:

                st.success(
                    "Aucun facteur aggravant significatif."
                )

            # ------------------------------------------------
            # FACTEURS PROTECTEURS
            # ------------------------------------------------

            st.markdown(
                "### 🟢 Facteurs réduisant le risque"
            )

            protective = explanation[
                "protective_factors"
            ][:10]

            if protective:

                protective_df = pd.DataFrame(
                    {
                        "Facteur": [
                            x["label"]
                            for x in protective
                        ],
                        "Contribution SHAP": [
                            x["shap_value"]
                            for x in protective
                        ],
                    }
                )

                fig = px.bar(
                    protective_df,
                    x="Contribution SHAP",
                    y="Facteur",
                    orientation="h",
                    text_auto=".3f",
                    title=(
                        "Facteurs protecteurs"
                    ),
                )

                st.plotly_chart(fig)

            else:

                st.info(
                    "Aucun facteur protecteur significatif."
                )

            # ------------------------------------------------
            # TOP FEATURES
            # ------------------------------------------------

            st.markdown(
                "### 📌 Principaux facteurs de la prédiction"
            )

            contributions = explanation[
                "contributions"
            ][:15]

            contribution_df = pd.DataFrame(
                {
                    "Facteur": [
                        x["label"]
                        for x in contributions
                    ],
                    "Contribution": [
                        x["shap_value"]
                        for x in contributions
                    ],
                }
            )

            fig = px.bar(
                contribution_df,
                x="Contribution",
                y="Facteur",
                orientation="h",
                text_auto=".3f",
                title="Top 15 contributions SHAP",
            )

            st.plotly_chart(fig)


# ============================================================
# PORTEFEUILLE
# ============================================================

# ============================================================
# PORTEFEUILLE
# ============================================================

elif page == "📊 Portefeuille":

    st.markdown(
        '<div class="section-title">📊 Analyse du portefeuille</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Chargez un fichier CSV contenant plusieurs clients afin
        d'appliquer le modèle XGBoost à l'ensemble du portefeuille.
        """
    )

    # ========================================================
    # COLONNES ATTENDUES
    # ========================================================

    required_columns = [
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]

    # ========================================================
    # UPLOAD CSV
    # ========================================================

    uploaded_file = st.file_uploader(
        "📂 Charger un portefeuille CSV",
        type=["csv"],
        help="Le fichier doit contenir les variables utilisées par le modèle.",
    )

    # ========================================================
    # TRAITEMENT DU FICHIER
    # ========================================================

    if uploaded_file is not None:

        try:

            portfolio_df = pd.read_csv(
                uploaded_file
            )

            # ------------------------------------------------
            # Vérification des colonnes
            # ------------------------------------------------

            missing_columns = [
                col
                for col in required_columns
                if col not in portfolio_df.columns
            ]

            if missing_columns:

                st.error(
                    "❌ Colonnes obligatoires manquantes : "
                    + ", ".join(missing_columns)
                )

                st.stop()

            # ------------------------------------------------
            # Sélection des variables du modèle
            # ------------------------------------------------

            portfolio_input = portfolio_df[
                required_columns
            ].copy()

            # ------------------------------------------------
            # Conversion numérique
            # ------------------------------------------------

            for column in required_columns:

                portfolio_input[column] = pd.to_numeric(
                    portfolio_input[column],
                    errors="coerce",
                )

            # ------------------------------------------------
            # Vérification des valeurs manquantes
            # ------------------------------------------------

            missing_values = (
                portfolio_input
                .isna()
                .sum()
                .sum()
            )

            if missing_values > 0:

                st.warning(
                    f"⚠️ {missing_values} valeur(s) manquante(s) "
                    "ont été détectées dans le fichier."
                )

                portfolio_input = (
                    portfolio_input
                    .fillna(0)
                )

            # ------------------------------------------------
            # Aperçu du portefeuille
            # ------------------------------------------------

            st.markdown(
                "### 👥 Aperçu du portefeuille"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Clients chargés",
                    f"{len(portfolio_input):,}".replace(",", " "),
                )

            with col2:

                st.metric(
                    "Variables",
                    len(required_columns),
                )

            with col3:

                st.metric(
                    "Modèle",
                    "XGBoost",
                )

            st.dataframe(
                portfolio_input.head(10),
                width="stretch",
                hide_index=True,
            )

            st.divider()

            # =================================================
            # BOUTON D'ANALYSE
            # =================================================

            analyse_portfolio = st.button(
                "🚀 ANALYSER LE PORTEFEUILLE",
                type="primary",
            )

            if analyse_portfolio:

                with st.spinner(
                    "Analyse du portefeuille en cours..."
                ):

                    # -----------------------------------------
                    # Feature engineering
                    # -----------------------------------------

                    portfolio_features = create_features(
                        portfolio_input
                    )

                    # -----------------------------------------
                    # Prédiction
                    # -----------------------------------------

                    probabilities = model.predict_proba(
                        portfolio_features
                    )[:, 1]

                    predictions = (
                        probabilities >= 0.55
                    ).astype(int)

                    # -----------------------------------------
                    # Résultats
                    # -----------------------------------------

                    results = portfolio_input.copy()

                    results[
                        "probabilite_defaut"
                    ] = probabilities

                    results[
                        "prediction"
                    ] = predictions

                    # -----------------------------------------
                    # Niveau de risque
                    # -----------------------------------------

                    results[
                        "niveau_risque"
                    ] = np.select(
                        [
                            probabilities < 0.30,
                            probabilities < 0.55,
                            probabilities < 0.75,
                        ],
                        [
                            "Faible",
                            "Modéré",
                            "Élevé",
                        ],
                        default="Très élevé",
                    )

                    # -----------------------------------------
                    # Probabilité en pourcentage
                    # -----------------------------------------

                    results[
                        "probabilite_defaut_pct"
                    ] = (
                        results[
                            "probabilite_defaut"
                        ] * 100
                    )

                    # -----------------------------------------
                    # Sauvegarde session
                    # -----------------------------------------

                    st.session_state[
                        "portfolio_results"
                    ] = results.copy()

                    st.session_state[
                        "portfolio_analyzed"
                    ] = True

            # =================================================
            # AFFICHAGE DES RÉSULTATS
            # =================================================

            if st.session_state.get(
                "portfolio_analyzed",
                False,
            ):

                results = st.session_state[
                    "portfolio_results"
                ]

                st.divider()

                st.markdown(
                    "## 📊 Résultats de l'analyse"
                )

                # =================================================
                # KPIs
                # =================================================

                total_clients = len(results)

                default_clients = int(
                    (
                        results["prediction"] == 1
                    ).sum()
                )

                default_rate = (
                    default_clients
                    / total_clients
                    if total_clients > 0
                    else 0
                )

                high_risk_clients = int(
                    results[
                        "niveau_risque"
                    ].isin(
                        [
                            "Élevé",
                            "Très élevé",
                        ]
                    ).sum()
                )

                average_probability = (
                    results[
                        "probabilite_defaut"
                    ].mean()
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Clients analysés",
                        f"{total_clients:,}".replace(
                            ",", " "
                        ),
                    )

                with col2:

                    st.metric(
                        "Défauts potentiels",
                        f"{default_clients:,}".replace(
                            ",", " "
                        ),
                    )

                with col3:

                    st.metric(
                        "Taux de défaut prédit",
                        f"{default_rate * 100:.2f} %",
                    )

                with col4:

                    st.metric(
                        "Probabilité moyenne",
                        f"{average_probability * 100:.2f} %",
                    )

                st.divider()

                # =================================================
                # DISTRIBUTION DES RISQUES
                # =================================================

                st.markdown(
                    "### 📈 Distribution des niveaux de risque"
                )

                risk_order = [
                    "Faible",
                    "Modéré",
                    "Élevé",
                    "Très élevé",
                ]

                risk_counts = (
                    results[
                        "niveau_risque"
                    ]
                    .value_counts()
                    .reindex(
                        risk_order,
                        fill_value=0,
                    )
                )

                risk_chart_data = pd.DataFrame(
                    {
                        "Niveau de risque": risk_counts.index,
                        "Nombre de clients": risk_counts.values,
                    }
                )

                fig_risk = px.bar(
                    risk_chart_data,
                    x="Niveau de risque",
                    y="Nombre de clients",
                    title="Répartition du portefeuille par niveau de risque",
                    text="Nombre de clients",
                )

                fig_risk.update_layout(
                    height=450,
                    xaxis_title="",
                    yaxis_title="Nombre de clients",
                )

                st.plotly_chart(
                    fig_risk,
                    width="stretch",
                )

                # =================================================
                # PROBABILITÉ DE DÉFAUT
                # =================================================

                st.markdown(
                    "### 🎯 Distribution des probabilités de défaut"
                )

                fig_probability = px.histogram(
                    results,
                    x="probabilite_defaut_pct",
                    nbins=20,
                    title="Distribution des probabilités de défaut",
                )

                fig_probability.update_layout(
                    height=400,
                    xaxis_title="Probabilité de défaut (%)",
                    yaxis_title="Nombre de clients",
                )

                fig_probability.add_vline(
                    x=55,
                    line_dash="dash",
                    annotation_text="Seuil 55 %",
                )

                st.plotly_chart(
                    fig_probability,
                    width="stretch",
                )

                # =================================================
                # CLIENTS À RISQUE
                # =================================================

                st.markdown(
                    "### 🚨 Clients présentant le risque le plus élevé"
                )

                high_risk = (
                    results[
                        results[
                            "prediction"
                        ] == 1
                    ]
                    .sort_values(
                        "probabilite_defaut",
                        ascending=False,
                    )
                    .copy()
                )

                display_columns = [
                    "LIMIT_BAL",
                    "AGE",
                    "SEX",
                    "EDUCATION",
                    "MARRIAGE",
                    "probabilite_defaut_pct",
                    "niveau_risque",
                ]

                if len(high_risk) > 0:

                    high_risk_display = (
                        high_risk[
                            display_columns
                        ]
                        .head(20)
                        .copy()
                    )

                    high_risk_display[
                        "probabilite_defaut_pct"
                    ] = high_risk_display[
                        "probabilite_defaut_pct"
                    ].round(2)

                    high_risk_display = (
                        high_risk_display.rename(
                            columns={
                                "LIMIT_BAL": "Limite de crédit",
                                "AGE": "Âge",
                                "SEX": "Sexe",
                                "EDUCATION": "Éducation",
                                "MARRIAGE": "Situation familiale",
                                "probabilite_defaut_pct": "Probabilité de défaut (%)",
                                "niveau_risque": "Niveau de risque",
                            }
                        )
                    )

                    st.dataframe(
                        high_risk_display,
                        width="stretch",
                        hide_index=True,
                    )

                else:

                    st.success(
                        "✅ Aucun client ne dépasse le seuil de décision de 55 %."
                    )

                # =================================================
                # TABLEAU COMPLET
                # =================================================

                st.markdown(
                    "### 📋 Résultats complets"
                )

                complete_display = results.copy()

                complete_display[
                    "probabilite_defaut_pct"
                ] = complete_display[
                    "probabilite_defaut_pct"
                ].round(2)

                complete_display = (
                    complete_display.rename(
                        columns={
                            "probabilite_defaut": "Probabilité brute",
                            "probabilite_defaut_pct": "Probabilité défaut (%)",
                            "prediction": "Défaut potentiel",
                            "niveau_risque": "Niveau de risque",
                        }
                    )
                )

                st.dataframe(
                    complete_display,
                    width="stretch",
                    hide_index=True,
                )

                # =================================================
                # TÉLÉCHARGEMENT
                # =================================================

                st.markdown(
                    "### 💾 Export des résultats"
                )

                csv_results = results.to_csv(
                    index=False
                ).encode(
                    "utf-8"
                )

                st.download_button(
                    label="⬇️ Télécharger les résultats CSV",
                    data=csv_results,
                    file_name="credit_risk_portfolio_results.csv",
                    mime="text/csv",
                )

                st.success(
                    "✅ Analyse du portefeuille terminée. "
                    "Les résultats sont également disponibles "
                    "dans la section 📈 Monitoring."
                )

        except Exception as e:

            st.error(
                "❌ Une erreur est survenue pendant "
                "l'analyse du portefeuille."
            )

            st.exception(e)


# ============================================================
# FIN PORTEFEUILLE
# ============================================================

# ============================================================
# MONITORING
# ============================================================

# ============================================================
# MONITORING DU RISQUE
# ============================================================

elif page == "📈 Monitoring":

    st.markdown(
        '<div class="section-title">📈 Monitoring du risque</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
        Suivi des indicateurs de risque du portefeuille analysé.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # VÉRIFICATION DU PORTEFEUILLE
    # --------------------------------------------------------

    if (
        "portfolio_results" not in st.session_state
        or st.session_state["portfolio_results"] is None
        or len(st.session_state["portfolio_results"]) == 0
    ):

        st.info(
            """
            Aucun portefeuille analysé pour le moment.

            Rendez-vous dans 📊 **Portefeuille**, chargez un fichier CSV
            puis lancez l'analyse afin d'alimenter automatiquement
            le monitoring.
            """
        )

    else:

        # ----------------------------------------------------
        # RÉCUPÉRATION DES RÉSULTATS
        # ----------------------------------------------------

        portfolio_results = st.session_state[
            "portfolio_results"
        ].copy()

        # ----------------------------------------------------
        # NORMALISATION DES COLONNES
        # ----------------------------------------------------

        probability_col = None
        prediction_col = None
        risk_col = None

        for col in portfolio_results.columns:

            col_lower = str(col).lower()

            if (
                probability_col is None
                and (
                    "probability" in col_lower
                    or "probabilité" in col_lower
                    or "probabilite" in col_lower
                )
            ):
                probability_col = col

            if (
                prediction_col is None
                and (
                    "prediction" in col_lower
                    or "prédiction" in col_lower
                )
            ):
                prediction_col = col

            if (
                risk_col is None
                and (
                    "risk_level" in col_lower
                    or "niveau" in col_lower
                    or "risque" in col_lower
                )
            ):
                risk_col = col

        # ----------------------------------------------------
        # SI LES COLONNES ATTENDUES EXISTENT
        # ----------------------------------------------------

        if probability_col is not None:

            probabilities = pd.to_numeric(
                portfolio_results[probability_col],
                errors="coerce",
            )

            # Conversion éventuelle 0-100 → 0-1
            if probabilities.max() > 1:

                probabilities = probabilities / 100

            probabilities = probabilities.clip(
                lower=0,
                upper=1,
            )

        else:

            probabilities = pd.Series(
                dtype=float
            )

        # ----------------------------------------------------
        # KPI
        # ----------------------------------------------------

        total_clients = len(
            portfolio_results
        )

        if len(probabilities) > 0:

            mean_probability = (
                probabilities.mean() * 100
            )

            high_risk_rate = (
                (probabilities >= 0.55).mean()
                * 100
            )

            very_high_risk_rate = (
                (probabilities >= 0.75).mean()
                * 100
            )

        else:

            mean_probability = 0
            high_risk_rate = 0
            very_high_risk_rate = 0

        # ----------------------------------------------------
        # TITRE
        # ----------------------------------------------------

        st.markdown(
            "### 📊 Indicateurs clés"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Clients analysés",
                f"{total_clients:,}".replace(
                    ",", " "
                ),
            )

        with col2:

            st.metric(
                "Probabilité moyenne",
                f"{mean_probability:.2f} %",
            )

        with col3:

            st.metric(
                "Risque ≥ 55 %",
                f"{high_risk_rate:.2f} %",
            )

        with col4:

            st.metric(
                "Risque ≥ 75 %",
                f"{very_high_risk_rate:.2f} %",
            )

        st.divider()

        # ----------------------------------------------------
        # DISTRIBUTION DES PROBABILITÉS
        # ----------------------------------------------------

        st.markdown(
            "### 📈 Distribution du risque"
        )

        if len(probabilities) > 0:

            risk_distribution = pd.DataFrame(
                {
                    "Client": range(
                        1,
                        len(probabilities) + 1,
                    ),
                    "Probabilité de défaut (%)":
                        probabilities * 100,
                }
            )

            fig_distribution = px.histogram(
                risk_distribution,
                x="Probabilité de défaut (%)",
                nbins=20,
                title="Distribution des probabilités de défaut",
            )

            fig_distribution.add_vline(
                x=55,
                line_dash="dash",
                annotation_text="Seuil 55 %",
            )

            fig_distribution.update_layout(
                height=450,
                xaxis_title="Probabilité de défaut (%)",
                yaxis_title="Nombre de clients",
            )

            st.plotly_chart(
                fig_distribution,
                width="stretch",
            )

        else:

            st.warning(
                "Aucune probabilité de défaut disponible."
            )

        st.divider()

        # ----------------------------------------------------
        # SEGMENTATION DU PORTEFEUILLE
        # ----------------------------------------------------

        st.markdown(
            "### 🎯 Segmentation du portefeuille"
        )

        if len(probabilities) > 0:

            risk_levels = pd.cut(
                probabilities,
                bins=[
                    -float("inf"),
                    0.30,
                    0.55,
                    0.75,
                    float("inf"),
                ],
                labels=[
                    "Faible",
                    "Modéré",
                    "Élevé",
                    "Très élevé",
                ],
            )

            risk_counts = (
                risk_levels
                .value_counts()
                .reindex(
                    [
                        "Faible",
                        "Modéré",
                        "Élevé",
                        "Très élevé",
                    ],
                    fill_value=0,
                )
            )

            risk_df = pd.DataFrame(
                {
                    "Niveau": risk_counts.index,
                    "Clients": risk_counts.values,
                }
            )

            col1, col2 = st.columns(2)

            with col1:

                fig_bar = px.bar(
                    risk_df,
                    x="Niveau",
                    y="Clients",
                    title="Nombre de clients par niveau de risque",
                    text="Clients",
                )

                fig_bar.update_layout(
                    height=420,
                    xaxis_title="Niveau de risque",
                    yaxis_title="Nombre de clients",
                )

                st.plotly_chart(
                    fig_bar,
                    width="stretch",
                )

            with col2:

                fig_pie = px.pie(
                    risk_df,
                    names="Niveau",
                    values="Clients",
                    title="Répartition du portefeuille",
                    hole=0.45,
                )

                fig_pie.update_layout(
                    height=420,
                )

                st.plotly_chart(
                    fig_pie,
                    width="stretch",
                )

        else:

            st.warning(
                "Impossible de calculer la segmentation."
            )

        st.divider()

        # ----------------------------------------------------
        # TABLEAU DE SYNTHÈSE
        # ----------------------------------------------------

        st.markdown(
            "### 📋 Synthèse du portefeuille"
        )

        if len(probabilities) > 0:

            summary_df = pd.DataFrame(
                {
                    "Indicateur": [
                        "Clients analysés",
                        "Probabilité moyenne",
                        "Clients ≥ 55 %",
                        "Clients ≥ 75 %",
                    ],
                    "Valeur": [
                        total_clients,
                        f"{mean_probability:.2f} %",
                        f"{int((probabilities >= 0.55).sum())}",
                        f"{int((probabilities >= 0.75).sum())}",
                    ],
                }
            )

            st.dataframe(
                summary_df,
                width="stretch",
                hide_index=True,
            )

        st.divider()

        # ----------------------------------------------------
        # PIED DE PAGE
        # ----------------------------------------------------

        st.caption(
            "Credit Risk Intelligence • XGBoost • "
            "48 features • Seuil de décision 55 %"
        )