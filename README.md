# Nutrition ML — Child Malnutrition Assessment

A machine learning pipeline and web application for assessing child malnutrition (ages 6–59 months) using anthropometric measurements and facial recognition for patient identification.

## Features

- **ML-based assessment** — Predicts wasting (WHZ-based) and stunting (HAZ-based) using Random Forest classifiers trained on CNNS (Comprehensive National Nutrition Survey) data
- **Clinical rule engine** — MUAC-based acute malnutrition screening (SAM < 115mm, MAM < 125mm) and hemoglobin-based anemia detection (Hb < 11 g/dL)
- **Risk stratification** — Four-tier system (critical/high/moderate/low) combining ML predictions with clinical rules
- **Facial recognition** — InsightFace (ArcFace) for identifying returning children without manual data entry
- **Live recognition** — Real-time webcam feed with face detection, automatic profile lookup, and dietary recommendations
- **Admin dashboard** — Profile management, high-risk child tracking, assessment history with trend analysis
- **SQLite database** — Persistent child profiles with photo storage, face encodings, and assessment history

## Architecture

```
src/
  generate_data.py    # Synthetic data generation (WHO growth curves)
  preprocess.py       # Data preprocessing, StandardScaler, train/test split
  train_acute.py      # Wasting model training (Random Forest)
  train_stunting.py   # Stunting model training (Random Forest)
  predict.py          # CLI-based prediction
  recommend.py        # Risk stratification & dietary recommendations
  rule_engine.py      # MUAC and anemia clinical rules
  database.py         # SQLite CRUD operations
  face.py             # InsightFace face encoding & matching
webapp.py             # NiceGUI web application (4 pages)
```

## Web Application Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | **Assess** | Upload photo for face matching or select existing profile, enter measurements, get risk assessment with detailed dietary recommendations |
| `/admin` | **Admin** | Dashboard with risk distribution stats, high-risk children list, searchable profile directory |
| `/admin/child/{id}` | **Child Detail** | Full profile with assessment history table, trend analysis (weight/height changes, risk trajectory), edit/delete |
| `/live` | **Live** | Real-time webcam recognition with bounding box, 2-second polling, camera flip, instant dietary recommendations for matched children |

## Risk Stratification

| Level | Criteria | Action |
|-------|----------|--------|
| **Critical** | MUAC SAM (< 115mm), or wasted + another condition | Immediate clinical referral |
| **High** | Wasted, MUAC MAM (< 125mm), or stunted + anemic | Urgent nutritional intervention |
| **Moderate** | Stunted or anemic alone | Monitor and follow up |
| **Low** | No conditions detected | Routine monitoring |

## Setup

### Requirements

- Python 3.10+
- Webcam (optional, for live recognition)

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

InsightFace downloads the ArcFace model (~300MB) on first run to `~/.insightface/models/`.

### Train Models

```bash
python -m src.preprocess
python -m src.train_acute
python -m src.train_stunting
```

### Run

**Web application:**
```bash
python webapp.py
# Opens at http://localhost:8081
```

**CLI prediction:**
```bash
python main.py
```

## Data

Training data: `data/synthetic_cnns_multistate_1to4_from_factsheets.dta` — 13,622 synthetic records across 24 Indian states, generated from CNNS state factsheet prevalence distributions.

**Input features:** age (months), sex, weight (kg), height (cm), MUAC (mm), hemoglobin (g/dL), BMI (derived)

**Prediction targets:**
- Wasting — `wasted_proxy` (WHZ < -2), 15% prevalence
- Stunting — `stunted_proxy` (HAZ < -2), 31% prevalence
- Anemia — rule-based (Hb < 11 g/dL), 34% prevalence
- MUAC — rule-based (< 125mm MAM, < 115mm SAM)

## Model Performance

| Model | Accuracy | Minority-class F1 |
|-------|----------|-------------------|
| Wasting (Random Forest, 300 trees) | 96% | 0.84 |
| Stunting (Random Forest, 300 trees) | 95% | 0.92 |

## Tech Stack

- **ML:** scikit-learn (Random Forest), pandas, joblib
- **Face recognition:** InsightFace (ArcFace/buffalo_l), onnxruntime
- **Web:** NiceGUI, FastAPI, Quasar (Vue.js)
- **Database:** SQLite
- **Frontend:** Custom CSS theme (light/dark), Geist font (optional)
