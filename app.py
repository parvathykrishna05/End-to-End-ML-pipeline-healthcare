<<<<<<< HEAD
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import sklearn
import csv
from typing import List
from datetime import datetime
import os


# Just to be explicit – we will pass arrays to scaler, so name check won’t matter
sklearn.set_config(transform_output="default")

# ---------- Load artifacts ----------
model = joblib.load("models/best_rf_model.joblib")
scaler = joblib.load("models/scaler.joblib")
feature_cols = joblib.load("models/scaled_feature_columns.joblib")

app = FastAPI(title="Healthcare Readmission API")

LOG_FILE = "prediction_logs.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "race",
            "gender",
            "age",
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "number_diagnoses",
            "change",
            "diabetesMed",
            "prediction",
            "probability_readmitted"
        ])

# ---------- Input schema ----------
class PatientFeatures(BaseModel):
    race: str
    gender: str
    age: str
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    change: str
    diabetesMed: str
    # If you used more columns when training, add them here with correct types

class PatientBatch(BaseModel):
    patients: List[PatientFeatures]


# ---------- Preprocessing ----------
def preprocess_input(data: PatientFeatures) -> np.ndarray:
    # Convert request body to DataFrame
    df = pd.DataFrame([data.dict()])

    # Ensure all columns used at training time exist
    for col in feature_cols:
        if col not in df.columns:
            # For now, fill missing training columns with 0 (or a sensible default)
            df[col] = 0

    # Drop any extra columns and reorder exactly like during training
    df = df[feature_cols]

    # IMPORTANT: in the notebook you label-encoded all object columns.
    # Here we assume the incoming values are already in the same numeric form.
    # If not, you should eventually load and apply the same LabelEncoders.
    # Temporary workaround: try to cast to float.
    df = df.astype(float)

    # Use NumPy array so sklearn doesn't check feature names
    X_array = df.to_numpy()

    # Apply the same scaler as training
    X_scaled = scaler.transform(X_array)
    return X_scaled


# ---------- Endpoints ----------
@app.get("/")
def read_root():
    return {"message": "Healthcare Readmission Prediction API is running."}


@app.post("/predict")
def predict_readmission(features: PatientFeatures):
    X_scaled = preprocess_input(features)
    proba = float(model.predict_proba(X_scaled)[0, 1])
    pred = int(proba >= 0.5)

    # ----- logging to CSV -----
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            features.race,
            features.gender,
            features.age,
            features.time_in_hospital,
            features.num_lab_procedures,
            features.num_procedures,
            features.num_medications,
            features.number_outpatient,
            features.number_emergency,
            features.number_inpatient,
            features.number_diagnoses,
            features.change,
            features.diabetesMed,
            pred,
            proba
        ])
    # ---------------------------

    return {
        "prediction": pred,
        "probability_readmitted": proba
    }

@app.post("/predict-batch")
def predict_batch(batch: PatientBatch):
    # Convert list of PatientFeatures to DataFrame
    rows = [p.dict() for p in batch.patients]
    df = pd.DataFrame(rows)

    # Ensure all training columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Reorder and cast to float (to match training + scaler)
    df = df[feature_cols].astype(float)

    # Use same scaler and model as /predict
    X_scaled = scaler.transform(df.to_numpy())
    probas = model.predict_proba(X_scaled)[:, 1]
    preds = (probas >= 0.5).astype(int).tolist()

    return {
        "predictions": preds,
        "probabilities": probas.tolist()
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
=======
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import sklearn
import csv
from typing import List
from datetime import datetime
import os


# Just to be explicit – we will pass arrays to scaler, so name check won’t matter
sklearn.set_config(transform_output="default")

# ---------- Load artifacts ----------
model = joblib.load("models/best_rf_model.joblib")
scaler = joblib.load("models/scaler.joblib")
feature_cols = joblib.load("models/scaled_feature_columns.joblib")

app = FastAPI(title="Healthcare Readmission API")

LOG_FILE = "prediction_logs.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "race",
            "gender",
            "age",
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "number_diagnoses",
            "change",
            "diabetesMed",
            "prediction",
            "probability_readmitted"
        ])

# ---------- Input schema ----------
class PatientFeatures(BaseModel):
    race: str
    gender: str
    age: str
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    change: str
    diabetesMed: str
    # If you used more columns when training, add them here with correct types

class PatientBatch(BaseModel):
    patients: List[PatientFeatures]


# ---------- Preprocessing ----------
def preprocess_input(data: PatientFeatures) -> np.ndarray:
    # Convert request body to DataFrame
    df = pd.DataFrame([data.dict()])

    # Ensure all columns used at training time exist
    for col in feature_cols:
        if col not in df.columns:
            # For now, fill missing training columns with 0 (or a sensible default)
            df[col] = 0

    # Drop any extra columns and reorder exactly like during training
    df = df[feature_cols]

    # IMPORTANT: in the notebook you label-encoded all object columns.
    # Here we assume the incoming values are already in the same numeric form.
    # If not, you should eventually load and apply the same LabelEncoders.
    # Temporary workaround: try to cast to float.
    df = df.astype(float)

    # Use NumPy array so sklearn doesn't check feature names
    X_array = df.to_numpy()

    # Apply the same scaler as training
    X_scaled = scaler.transform(X_array)
    return X_scaled


# ---------- Endpoints ----------
@app.get("/")
def read_root():
    return {"message": "Healthcare Readmission Prediction API is running."}


@app.post("/predict")
def predict_readmission(features: PatientFeatures):
    X_scaled = preprocess_input(features)
    proba = float(model.predict_proba(X_scaled)[0, 1])
    pred = int(proba >= 0.5)

    # ----- logging to CSV -----
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            features.race,
            features.gender,
            features.age,
            features.time_in_hospital,
            features.num_lab_procedures,
            features.num_procedures,
            features.num_medications,
            features.number_outpatient,
            features.number_emergency,
            features.number_inpatient,
            features.number_diagnoses,
            features.change,
            features.diabetesMed,
            pred,
            proba
        ])
    # ---------------------------

    return {
        "prediction": pred,
        "probability_readmitted": proba
    }

@app.post("/predict-batch")
def predict_batch(batch: PatientBatch):
    # Convert list of PatientFeatures to DataFrame
    rows = [p.dict() for p in batch.patients]
    df = pd.DataFrame(rows)

    # Ensure all training columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Reorder and cast to float (to match training + scaler)
    df = df[feature_cols].astype(float)

    # Use same scaler and model as /predict
    X_scaled = scaler.transform(df.to_numpy())
    probas = model.predict_proba(X_scaled)[:, 1]
    preds = (probas >= 0.5).astype(int).tolist()

    return {
        "predictions": preds,
        "probabilities": probas.tolist()
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
>>>>>>> d52d767 (initial commit)
