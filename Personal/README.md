# 🫀 Predicting 10-Year Coronary Heart Disease Risk
### Framingham Heart Study — Binary Classification Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Gradio](https://img.shields.io/badge/Gradio-deployed-brightgreen?logo=gradio)](https://gradio.app/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)](https://jupyter.org/)

---

## Overview

This project applies machine learning to predict a patient's **10-year risk of developing coronary heart disease (CHD)** using structured clinical data from the Framingham Heart Study — one of the most influential longitudinal cardiovascular studies ever conducted.

The full workflow — from raw data to an interactive web application — is implemented in a single, self-contained Jupyter Notebook (`Week7_chd_prediction.ipynb`).

---

## Table of Contents

1. [Dataset](#dataset)
2. [Project Structure](#project-structure)
3. [Workflow Summary](#workflow-summary)
   - [Data Loading & Description](#1-data-loading--description)
   - [Exploratory Data Analysis](#2-exploratory-data-analysis)
   - [Preprocessing](#3-preprocessing)
   - [Feature Engineering](#4-feature-engineering)
   - [Modelling](#5-modelling)
   - [Best Model Deep-Dive](#6-best-model-deep-dive--logistic-regression)
   - [Hyperparameter Tuning](#7-hyperparameter-tuning)
   - [Gradio Deployment](#8-gradio-deployment)
4. [Results](#results)
5. [Key Findings](#key-findings)
6. [Installation & Usage](#installation--usage)
7. [Dependencies](#dependencies)
8. [Limitations](#limitations)

---

## Dataset

| Property | Detail |
|---|---|
| **Name** | Framingham Heart Study |
| **File** | `train.csv` |
| **Rows** | ~4,200 patients |
| **Features** | 15 clinical & demographic variables |
| **Target** | `TenYearCHD` — binary (1 = CHD within 10 years, 0 = No CHD) |
| **Class balance** | ~85% No CHD / ~15% CHD (imbalanced) |

### Feature Reference

| Category | Feature | Description |
|---|---|---|
| **Demographic** | `age` | Patient age (years) |
| | `sex` | `'M'` / `'F'` |
| | `education` | Highest education level (1–4) — *dropped* |
| **Behavioural** | `is_smoking` | Current smoker — *dropped (redundant)* |
| | `cigsPerDay` | Average cigarettes per day |
| **Medical (history)** | `BPMeds` | On blood pressure medication (0/1) |
| | `prevalentStroke` | Prior stroke (0/1) |
| | `prevalentHyp` | Hypertensive (0/1) |
| | `diabetes` | Diabetic (0/1) — *dropped (redundant with glucose)* |
| **Medical (current)** | `totChol` | Total cholesterol (mg/dL) |
| | `sysBP` | Systolic blood pressure (mmHg) |
| | `diaBP` | Diastolic blood pressure (mmHg) — *dropped (collinear)* |
| | `BMI` | Body Mass Index (kg/m²) |
| | `heartRate` | Resting heart rate (bpm) |
| | `glucose` | Blood glucose (mg/dL) |
| **Target** | `TenYearCHD` | 10-year CHD risk — **1 = Yes**, **0 = No** |

---

## Project Structure

```
.
├── Week7_chd_prediction.ipynb      # Main notebook (full pipeline)
├── train.csv                       # Raw dataset (required)
└── README.md                       # This file
```

---

## Workflow Summary

### 1. Data Loading & Description

- Dataset loaded with `pandas` and thoroughly inspected: shape, dtypes, summary statistics, and unique value counts per column.
- The `id` column (row identifier, no predictive value) is removed immediately.
- A formatted feature reference table documents every variable's category, name, and clinical meaning.

---

### 2. Exploratory Data Analysis

#### Missing Values
Six features contained missing data. Visualised with a **missingness heatmap** and a **horizontal bar chart** of percentage missing per feature.

| Feature | Missing % | Strategy |
|---|---|---|
| `education` | ~2.5% | Dropped — not a clinical predictor |
| `glucose` | ~9.1% | MICE imputation |
| `BPMeds` | ~1.3% | Filled with mode (0) |
| `cigsPerDay`, `totChol`, `BMI`, `heartRate` | <2% | MICE imputation |

#### Target Distribution
The dataset is significantly **imbalanced** (~85 : 15). Visualised with a count bar chart and pie chart. All models use `class_weight='balanced'` to counteract this.

#### Univariate Analysis
Distribution plots (histogram + KDE + mean line) for all 8 numerical features: `age`, `cigsPerDay`, `totChol`, `sysBP`, `diaBP`, `BMI`, `heartRate`, `glucose`.

#### Bivariate Analysis
- **Numerical vs target:** box plots comparing distributions of each numerical feature across CHD / No-CHD groups.
- **Categorical vs target:** count plots for `sex`, `is_smoking`, `BPMeds`, `prevalentStroke`, `prevalentHyp`, `diabetes`.

#### Correlation Analysis
A masked lower-triangle heatmap reveals:
- `sysBP` ↔ `diaBP` : r ≈ 0.79 → keep `sysBP`, drop `diaBP`
- `diabetes` ↔ `glucose` : high overlap → keep `glucose`, drop `diabetes`
- `is_smoking` ↔ `cigsPerDay` : redundant → drop `is_smoking`

---

### 3. Preprocessing

| Step | Detail |
|---|---|
| Drop `education` | Not clinically meaningful; too many missing values |
| MICE imputation | `IterativeImputer(max_iter=10)` on 5 numerical columns |
| Mode-fill `BPMeds` | Missing → 0 (the dominant class) |
| Duplicate check | No duplicates found |
| Drop correlated columns | `diaBP`, `diabetes`, `is_smoking` removed |

---

### 4. Feature Engineering

On top of the cleaned original features, three clinically motivated derived features were created:

| Feature | Formula / Logic | Clinical Meaning |
|---|---|---|
| `pulse_pressure` | `sysBP − diaBP` (proxied) | Marker of arterial stiffness |
| `heavy_smoker` | `cigsPerDay > 20` → 1/0 | High-load smoking flag |
| `hyp_on_meds` | `prevalentHyp == 1 AND BPMeds == 1` | Controlled hypertension indicator |

`sex` was binary-encoded: Male = 1, Female = 0.

**Final feature count: 14** (after encoding and engineering).

---

### 5. Modelling

#### Train / Test Split
- 80 / 20 stratified split (`random_state=42`)
- CHD rate preserved in both splits

#### Model Suite (7 classifiers)

| Model | Notes |
|---|---|
| **Dummy (Baseline)** | Always predicts majority class |
| **Logistic Regression** | L1/L2 regularisation, `class_weight='balanced'` |
| **Decision Tree** | `max_depth=5`, balanced |
| **Random Forest** | 300 trees, balanced |
| **HistGradientBoosting** | `learning_rate=0.05`, `max_depth=5` |
| **SVC** | RBF kernel, probability calibration, balanced |
| **KNN** | k=15, scaled inputs |

#### Evaluation Metrics
Given class imbalance, accuracy alone is insufficient. All models are evaluated on:

- **ROC-AUC** — overall discrimination ability
- **PR-AUC** (Precision-Recall) — performance on the minority (CHD) class
- **F1 Score (CHD class)** — harmonic mean of precision and recall for positives
- **Recall (CHD class)** — how many true CHD cases are caught
- **Accuracy** — reported for reference only
- **McFadden R²** — goodness-of-fit for probabilistic classifiers

---

### 6. Best Model Deep-Dive — Logistic Regression

Logistic Regression achieved the best overall performance and was selected as the final model due to its combination of **discriminative power** and **clinical interpretability**.

#### Performance (test set)

| Metric | Score |
|---|---|
| ROC-AUC | ~0.75 |
| PR-AUC | ~0.39 |
| F1 (CHD) | competitive |
| McFadden R² | ~0.88 |

#### Diagnostic Plots
- **Confusion matrix** — TP/FP/TN/FN breakdown
- **ROC curve** — discrimination across all thresholds

#### Feature Coefficients & Odds Ratios

Coefficients are on standardised features, so odds ratios reflect the impact of a 1-SD increase in each variable:

| Feature | Odds Ratio | Interpretation |
|---|---|---|
| `age` | ~1.63 | Strongest predictor — 63% higher CHD odds per SD |
| `sex` (Male) | ~1.50 | Males face meaningfully higher risk |
| `sysBP` | ~1.40 | Higher systolic BP substantially raises risk |
| `cigsPerDay` | ~1.25 | More cigarettes compounds risk |
| `totChol` | ~1.10 | Elevated cholesterol modestly increases risk |

---

### 7. Hyperparameter Tuning

5-fold `GridSearchCV` over:

```python
param_grid = {
    'model__C':       np.logspace(-3, 2, 10),   # regularisation strength
    'model__penalty': ['l1', 'l2'],
    'model__solver':  ['liblinear']
}
scoring = 'roc_auc'
```

A **regularisation curve** plots mean CV AUC ± 1 std against C (log scale), with the optimal C marked. The tuned model is re-evaluated on the hold-out test set with the full classification report.

---

### 8. Gradio Deployment

The tuned Logistic Regression pipeline is deployed as an interactive web app using **Gradio**.

#### Features
- Three-column layout: Demographic & Behavioural | Medical History | Clinical Measurements
- Sliders for all continuous inputs (age, BP, cholesterol, BMI, glucose, etc.)
- Checkboxes for binary flags (BPMeds, stroke, hypertension, heavy smoker, hyp+meds)
- Instant prediction with probability percentage and colour-coded risk tier:

| Probability | Label |
|---|---|
| < 10% | 🟢 Low Risk |
| 10–20% | 🟡 Moderate Risk |
| 20–35% | 🟠 High Risk |
| > 35% | 🔴 Very High Risk |

- Three pre-filled **example patients** (low-risk, high-risk, very high-risk)
- `share=True` generates a **public URL** valid for 72 hours — no server or account needed

#### How to launch
Simply run all cells in the notebook. The final Gradio cell will print:
```
Running on public URL: https://xxxx.gradio.live
```

---

## Results

| Model | ROC-AUC | PR-AUC | F1 (CHD) | Accuracy |
|---|---|---|---|---|
| **Logistic Regression** ✅ | **~0.75** | **~0.39** | competitive | ~0.65 |
| Random Forest | ~0.73 | ~0.35 | moderate | ~0.75 |
| HistGradientBoosting | ~0.72 | ~0.34 | moderate | ~0.80 |
| SVC | ~0.71 | ~0.33 | moderate | ~0.68 |
| Decision Tree | ~0.65 | ~0.28 | lower | ~0.67 |
| KNN | ~0.63 | ~0.26 | lower | ~0.72 |
| Dummy (Baseline) | 0.50 | ~0.15 | 0.00 | ~0.85 |

> Accuracy is **intentionally de-emphasised** — the Dummy classifier scores 85% accuracy while providing zero clinical value.

---

## Key Findings

- **Age** is the single strongest predictor of 10-year CHD risk.
- **Systolic blood pressure** and **daily cigarette use** are the most impactful modifiable risk factors identified by the model.
- **Male sex** is independently associated with elevated CHD odds.
- **Accuracy is a misleading metric** with imbalanced health data; ROC-AUC and PR-AUC are more appropriate.
- **Logistic Regression outperforms** more complex ensemble methods in this setting, and its probabilistic outputs map directly to clinical risk communication.

---

## Installation & Usage

### 1. Clone / download the project
```bash
git clone <your-repo-url>
cd framingham-chd
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
Or install manually:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn gradio jupyter
```

### 3. Add the dataset
Place `train.csv` in the same directory as the notebook.

### 4. Run the notebook
```bash
jupyter notebook framingham_chd_analysis.ipynb
```
Run all cells top-to-bottom. The final cell launches the Gradio app and prints a shareable URL.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical computation |
| `pandas` | Data manipulation |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualisation |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `gradio` | Interactive web deployment |
| `jupyter` | Notebook environment |

Python 3.10+ recommended.

---

## Limitations

- **Population bias:** The Framingham cohort is predominantly white and American (late 1940s–1960s). Generalisability to modern, diverse populations is limited.
- **Missing features:** LDL/HDL cholesterol ratio, family history, C-reactive protein, and lifestyle factors are not captured.
- **Threshold fixed at 0.5:** Optimal decision thresholds should be set based on the clinical cost trade-off between false negatives (missed CHD) and false positives (unnecessary intervention).
- **Static model:** No retraining or drift detection — the model reflects the training distribution only.
- **Not validated externally:** Model performance on independent cohorts has not been assessed.

---

## Disclaimer

> This project is for **research and educational purposes only**. The model and application do not constitute medical advice and should not be used for clinical decision-making.

---

*Framingham Heart Study data · scikit-learn · Gradio · Python*
