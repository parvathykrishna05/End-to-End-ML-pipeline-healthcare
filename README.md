# Healthcare Readmission Prediction – End-to-End ML Pipeline

End-to-end machine learning project to predict 30-day hospital readmission risk using the UCI diabetes dataset. Covers full lifecycle: data ingestion, preprocessing, model training/tuning, experiment tracking, explainability, and FastAPI deployment.

## Project Overview

- **Goal**: Predict 30-day hospital readmission risk from demographics, hospital stays, diagnoses, medications
- **Dataset**: Diabetes 130-US hospitals (Kaggle/UCI) via Azure Blob Storage
- **Stack**: Python, pandas, scikit-learn, MLflow, SHAP, FastAPI, Uvicorn

## Features

- **Data Pipeline**: Azure Blob → cleaning → categorical encoding → feature engineering
- **Modeling**: Tuned RandomForestClassifier (RandomizedSearchCV) + baseline comparison
- **Tracking**: MLflow for metrics/parameters/artifacts
- **API**: FastAPI service with health checks, single/batch prediction, CSV logging

## Project Structure

├─ app.py # FastAPI service (/health, /predict, /predict-batch)

├─ healthcare-readmission-analysis.ipynb # EDA + training + SHAP notebook

└─ .gitignore # Ignores models/mlruns/logs


## Quick Start

### 1. Setup
pip install -r requirements.txt

### 2. Train Model
Run `healthcare-readmission-analysis.ipynb` → generates `models/` artifacts

### 3. Start API
uvicorn app:app --reload

### 4. Test API
- `http://127.0.0.1:8000/docs` → interactive Swagger UI
- `GET /health` → `{"status": "ok"}`
- `POST /predict` → single patient prediction

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/predict` | POST | Single patient prediction |
| `/predict-batch` | POST | Multiple patients |

## Sample Request
{
"race": "1", "gender": "1", "age": "5",
"time_in_hospital": 5, "num_lab_procedures": 40,
"num_procedures": 1, "num_medications": 13,
"number_outpatient": 0, "number_emergency": 0,
"number_inpatient": 1, "number_diagnoses": 7,
"change": "1", "diabetesMed": "1"
}

**Response**: `{"prediction": 1, "probability_readmitted": 0.57}`

## Results

- **Best Model**: RandomForest (tuned: 300 estimators, max_depth=20)
- **Metrics**: ~0.65 accuracy, ~0.70 F1-score (<30day class)
- **Top Features**: `num_lab_procedures`, `time_in_hospital`, `num_medications`

## Deployment History

- Originally deployed on **Azure Databricks + MLflow**
- Model artifacts saved via `joblib`
- API runs locally (container-ready for cloud)

## Next Steps

- Docker containerization
- Azure Container Apps deployment
- Real-time monitoring dashboard
- A/B testing with logistic regression baseline

---
**Built by Parvathy Krishna M** | Aspiring Data Scientist/ML Engineer/Data Engineer | [LinkedIn](https://www.linkedin.com/in/parvathy-krishna-726a5227b/)
