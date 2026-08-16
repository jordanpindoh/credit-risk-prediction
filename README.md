

# 💳 Credit Risk Intelligence

### Explainable Credit Risk Prediction & Portfolio Analytics

Une plateforme de Machine Learning dédiée à l'analyse et à la prédiction du risque de défaut de crédit.

Le projet combine **XGBoost**, **feature engineering**, **Explainable AI avec SHAP** et **Streamlit** afin de transformer un modèle de classification en une application complète d'analyse du risque.

---

## 🎯 Objectif du projet

L'objectif de **Credit Risk Intelligence** est d'estimer la probabilité de défaut d'un client à partir de ses caractéristiques financières et comportementales.

La plateforme permet également d'interpréter les prédictions du modèle et d'analyser un portefeuille complet de clients.

Le projet couvre plusieurs niveaux d'analyse :

- 👤 Analyse individuelle d'un client
- 🔮 Simulation de profils
- 🧠 Explainable AI avec SHAP
- 📊 Analyse de portefeuille
- 📈 Monitoring du risque
- 💾 Export des résultats

---

## 🚀 Fonctionnalités principales

### 👤 Analyse client

Évaluation individuelle du profil d'un client et estimation de sa probabilité de défaut.

L'application fournit notamment :

- la probabilité de défaut ;
- le niveau de risque ;
- la décision associée au seuil du modèle.

### 🔮 Simulation

Modification des caractéristiques d'un profil client afin d'observer l'évolution de sa probabilité de défaut.

### 🧠 Explainable AI

Interprétation des prédictions individuelles avec **SHAP (SHapley Additive exPlanations)**.

La plateforme identifie notamment :

- les facteurs contribuant à augmenter le risque ;
- les facteurs contribuant à réduire le risque ;
- les contributions des variables à la prédiction.

### 📊 Analyse de portefeuille

Chargement d'un portefeuille CSV et application du modèle à plusieurs clients.

### 📈 Monitoring

Visualisation de la distribution du risque sur le portefeuille analysé.

### 💾 Export

Export des résultats de l'analyse de portefeuille au format CSV.




---

## 🤖 Machine Learning

### Modèle

Le modèle principal utilisé dans le projet est un **XGBoost Classifier**.

Le modèle est intégré dans un pipeline comprenant :

1. préparation des données ;
2. feature engineering ;
3. preprocessing ;
4. classification avec XGBoost ;
5. estimation de la probabilité de défaut.

### 🎯 Seuil de décision

Le seuil opérationnel utilisé par l'application est fixé à :

**55 %**

Ainsi :

- probabilité < 55 % → **Risque acceptable**
- probabilité ≥ 55 % → **Défaut potentiel**

Ce seuil est utilisé dans les différents modules de l'application afin de conserver une logique de décision cohérente.

### 🔢 Features

Le pipeline produit actuellement :

**48 features après preprocessing**

Les variables exploitées couvrent notamment :

- la limite de crédit ;
- l'âge ;
- les caractéristiques démographiques ;
- l'historique des retards ;
- les montants des factures ;
- les montants des paiements ;
- les ratios de paiement ;
- différents indicateurs comportementaux issus du feature engineering.

---

## 🧠 Explainable AI — SHAP

L'interprétabilité constitue une composante centrale de **Credit Risk Intelligence**.

Le projet utilise **SHAP** afin d'expliquer les prédictions individuelles produites par le modèle XGBoost.

Pour chaque client analysé, le système calcule les contributions des différentes variables à la prédiction.

### Facteurs aggravants

Les variables présentant une contribution SHAP positive augmentent la sortie du modèle vers le risque de défaut.

### Facteurs protecteurs

Les variables présentant une contribution SHAP négative diminuent la sortie du modèle vers le risque de défaut.

### Exemple

Pour un profil de test, les principaux facteurs protecteurs identifiés par le modèle comprennent notamment :

- **Ratio de paiement minimum**
- **Nombre de retards**
- **Retard le plus récent**
- **Retard moyen**
- **Facture récente**

Les contributions SHAP permettent ainsi de passer d'une simple prédiction à une **explication de la décision du modèle**.

---

## 🔍 Exemple de sortie SHAP

Le système produit notamment :

```text
Nombre de features SHAP : 48

Ratio de paiement minimum    → contribution négative
Nombre de retards            → contribution négative
Retard le plus récent        → contribution négative
Retard moyen                 → contribution négative
Facture récente              → contribution négative



---

## 🏗️ Architecture du projet

L'application suit une architecture séparant la logique métier, le modèle de Machine Learning et l'interface utilisateur.

```text
                         Données client / CSV
                                  │
                                  ▼
                         Feature Engineering
                                  │
                                  ▼
                            Preprocessing
                                  │
                                  ▼
                            XGBoost Model
                                  │
                     ┌────────────┴────────────┐
                     │                         │
                     ▼                         ▼
              Probabilité                  SHAP
                     │                         │
                     ▼                         ▼
             Niveau de risque           Explication
                     │
                     ▼
                  Streamlit
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
     Analyse      Simulation   Portefeuille
        │            │            │
        └────────────┼────────────┘
                     ▼
                 Monitoring


                 ---

## ⚙️ Installation

### 1. Cloner le projet

```bash
git clone <URL_DU_REPOSITORY>
cd credit-risk-prediction


---

## 📊 Résultats du projet

Le projet fournit une chaîne complète allant de la préparation des données jusqu'à l'interprétation des prédictions.

Les principaux résultats disponibles dans l'application sont :

- probabilité individuelle de défaut ;
- classification du niveau de risque ;
- décision selon le seuil de 55 % ;
- contributions SHAP individuelles ;
- segmentation d'un portefeuille ;
- distribution des probabilités de défaut ;
- identification des profils présentant le risque le plus élevé ;
- export des résultats au format CSV.

L'application permet ainsi de passer d'un modèle de Machine Learning isolé à une **plateforme interactive d'analyse du risque de crédit**.

---

## ⚠️ Limites du projet

Ce projet est conçu comme un projet de **Data Science / Machine Learning et de portfolio professionnel**.

Les prédictions produites ne constituent pas une décision financière réelle.

Une utilisation dans un environnement bancaire ou financier réel nécessiterait notamment :

- une validation approfondie du modèle ;
- une analyse des biais et de l'équité ;
- une validation métier ;
- une gouvernance du modèle ;
- un suivi de la dérive des données ;
- un monitoring des performances ;
- une validation réglementaire ;
- une gestion appropriée des données personnelles ;
- une stratégie de réentraînement du modèle.

Le modèle doit donc être considéré comme un outil d'aide à l'analyse et non comme un système autonome de décision financière.

---

## 🔭 Perspectives d'amélioration

Plusieurs évolutions pourraient être ajoutées au projet :

### Machine Learning

- comparaison avec d'autres algorithmes ;
- optimisation avancée des hyperparamètres ;
- calibration des probabilités ;
- analyse plus approfondie du compromis précision / rappel ;
- validation croisée renforcée.

### Explainable AI

- visualisations SHAP globales ;
- analyse de l'importance des variables ;
- comparaison des explications entre différents profils ;
- génération automatique de rapports d'explication.

### MLOps

- suivi des versions du modèle ;
- automatisation du pipeline d'entraînement ;
- monitoring de la dérive des données ;
- suivi des performances en production ;
- pipeline CI/CD ;
- tests automatisés.

### Application

- authentification des utilisateurs ;
- gestion de plusieurs portefeuilles ;
- historique des analyses ;
- tableaux de bord plus avancés ;
- déploiement cloud ;
- API de scoring.

---

## 🧪 Validation technique

Le projet comprend également des scripts dédiés à la validation des fonctionnalités principales :

```text
src/test_prediction.py
src/test_shap.py
