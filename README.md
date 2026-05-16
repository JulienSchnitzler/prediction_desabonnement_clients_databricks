# Prédiction du Désabonnement Clients (Churn) — PySpark & Databricks

> Projet étudiant — MS Expert Big Data Engineer, Promo 2026, UTT Troyes

---

## Auteurs

| Nom | Filière |
|-----|---------|
| Guy Martial GADJEU KAMENI | MS Expert Big Data Engineer — Promo 2026 |
| Julien Schnitzler | MS Expert Big Data Engineer — Promo 2026 |

---

## Contexte & Objectif

Une entreprise de télécommunications souhaite **identifier proactivement les clients susceptibles de se désabonner (churn)** afin d'engager des actions de fidélisation ciblées avant qu'ils ne partent.

**Problème :** Classification binaire — `Churn = Yes (1)` / `Churn = No (0)`

**Pipeline complet :**
1. Chargement et exploration des données avec **PySpark** sur Databricks
2. Analyse exploratoire approfondie (EDA) avec visualisations
3. Feature engineering en **PySpark natif**
4. Modélisation avec **scikit-learn + XGBoost** (3 modèles comparés)
5. Évaluation détaillée et sélection du meilleur modèle
6. Recommandations métier et stratégie de déploiement

---

## Jeu de données

**Telco Customer Churn** — [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Caractéristique | Valeur |
|-----------------|--------|
| Lignes | 7 043 clients |
| Colonnes | 21 variables |
| Cible | `Churn` (Yes / No) |
| Déséquilibre | ~73.5 % Non-Churn / 26.5 % Churn |

**Variables incluses :** données démographiques, type de contrat, services souscrits, mode de paiement, montant des factures, ancienneté.

---

## Stack technique

| Composant | Technologie |
|-----------|------------|
| Plateforme | Databricks Community Edition |
| Big Data | PySpark 3.5 |
| Machine Learning | scikit-learn, XGBoost |
| Visualisation | matplotlib, seaborn |
| Data manipulation | pandas, numpy |

> **Note :** La bibliothèque **Spark MLlib** (`pyspark.ml`) n'est pas disponible sur Databricks Community Edition. Le feature engineering utilise PySpark natif (`when`, `col`) et la modélisation utilise scikit-learn + XGBoost, qui sont des équivalents Python de `StringIndexer`/`VectorAssembler` et `GBTClassifier` (MLlib).

---

## Structure du notebook

Le notebook [`prediction_desabonnement_clients.ipynb`](prediction_desabonnement_clients.ipynb) est organisé en 8 sections (87 cellules) :

```
A  Chargement des librairies
│
B  Importation des données & Exploration initiale
│   ├─ B.1  Chargement du CSV
│   ├─ B.2  Affichage du dataset (7 043 × 21)
│   ├─ B.3  Schéma + correction type TotalCharges
│   └─ B.4  Résumé statistique commenté
│
C  Analyse Exploratoire des Données (EDA) — Complète
│   ├─ C.1  Valeurs manquantes + imputation intelligente
│   │         (TotalCharges = tenure × MonthlyCharges, justifié)
│   ├─ C.2  Valeurs uniques des variables catégorielles
│   ├─ C.3  Distributions (Churn, numériques, catégorielles)
│   ├─ C.4  Heatmap des corrélations
│   ├─ C.5  Analyse bivariée — taux de Churn par variable
│   ├─ C.6  Détection des outliers (IQR + boxplots)
│   └─ C.7  Synthèse des insights clés
│
D  Préparation des données & Feature Engineering
│   ├─ D.1  Encodage binaire Yes/No → 0/1 (PySpark natif)
│   ├─ D.2  Suppression colonnes inutiles
│   ├─ D.3  One-Hot Encoding des variables multi-catégorielles (pandas)
│   └─ D.4  StandardScaler + découpage stratifié Train/Test 80/20
│
E  Modélisation — 3 algorithmes comparés
│   ├─ E.1  Fonction helper train_evaluate()
│   ├─ E.2  Régression Logistique (baseline)
│   ├─ E.3  Random Forest
│   ├─ E.4  XGBoost (Gradient Boosting)
│   ├─ E.5  Tableau comparatif des 5 métriques
│   └─ E.6  Visualisation barres groupées
│
F  Évaluation Détaillée
│   ├─ F.1  Matrices de confusion (3 modèles)
│   ├─ F.2  Courbes ROC + Precision-Recall
│   ├─ F.3  Analyse FP / FN (coût métier)
│   ├─ F.4  Importance des features (RF vs XGBoost)
│   └─ F.5  Recommandations métier + stratégie de seuillage
│
G  Conclusions
│   ├─ G.1  Modèle retenu + justification
│   ├─ G.2  Performance vs baseline naïf
│   ├─ G.3  Limitations & biais détectés
│   └─ G.4  Recommandations pour la production
│
H  [Optionnel] Déploiement
    ├─ H.1  Sauvegarde du modèle (joblib)
    └─ H.2  Pipeline de scoring batch PySpark
```

---

## Résultats obtenus

| Modèle | AUC-ROC | F1-Score | Recall | Accuracy |
|--------|:-------:|:--------:|:------:|:--------:|
| Baseline (Non-Churn) | 0.500 | 0.000 | 0.000 | 0.735 |
| Régression Logistique | 0.841 | 0.616 | **0.786** | 0.740 |
| **Random Forest** ✅ | **0.843** | **0.633** | 0.751 | **0.769** |
| XGBoost | 0.834 | 0.602 | 0.671 | 0.764 |

**Modèle retenu : Random Forest**
- Meilleur AUC-ROC (0.843) et F1-Score (0.633)
- Robuste au déséquilibre de classes (`class_weight='balanced'`)

**Features les plus importantes (consensus RF + XGBoost) :**
`Contract_Two year`, `Contract_One year`, `tenure`, `InternetService_Fiber optic`, `PaymentMethod_Electronic check`

---

## Lancer le notebook

### Sur Databricks Community Edition (recommandé)

1. Importer le notebook dans votre workspace Databricks
2. Uploader `WA_Fn-UseC_-Telco-Customer-Churn.csv` dans `/data/raw/`
3. Mettre à jour le `chemin_csv` (Section B.1) avec votre chemin workspace
4. **Run All**

### En local (VS Code / Jupyter)

**Prérequis :**
- Python ≥ 3.11
- Java JDK 11 ou 17 (ex: [Eclipse Adoptium](https://adoptium.net/))
- `JAVA_HOME` configuré **ou** Java installé dans un chemin standard

```bash
# Installation des dépendances
pip install pyspark scikit-learn xgboost matplotlib seaborn pandas numpy

# Lancer Jupyter
jupyter lab
```

> La cellule SparkSession (Section A) **détecte automatiquement l'environnement** :
> - Sur Databricks → utilise la session existante
> - En local → configure `JAVA_HOME` et crée la session

---

## Structure du repo

```
prediction_desabonnement_clients_databricks/
├── prediction_desabonnement_clients.ipynb   # Notebook principal (87 cellules)
├── data/
│   └── raw/                                  # Dossier pour le CSV (non versionné)
├── main.py                                   # Script Python autonome
├── pyproject.toml                            # Dépendances du projet
├── uv.lock                                   # Lock file uv
└── README.md
```

> Le dataset CSV n'est pas inclus dans le repo (taille). Le télécharger depuis [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

---

## Imputation des valeurs manquantes

11 valeurs manquantes dans `TotalCharges` — toutes correspondant à `tenure = 0`.

**Approche retenue** (justifiée statistiquement) :

```
TotalCharges = tenure × MonthlyCharges
```

Corrélation empirique entre cette formule et les valeurs existantes : **r > 0.99**.  
Pour `tenure = 0` → `TotalCharges = 0` (aucune charge cumulée, logique métier).

---

*Projet réalisé dans le cadre de l'UE "Autre langage de programmation — Spark", UTT Troyes, 2025-2026.*
