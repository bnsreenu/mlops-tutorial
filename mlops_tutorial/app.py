"""
app.py — FastAPI prediction service for the Heart Disease classifier.
Loads the trained model and scaler at startup.
Exposes /health and /predict endpoints.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from pathlib import Path

# Ensure paths resolve relative to the project root
# app.py lives in mlops_tutorial/, so one level up is the root
ROOT = Path(__file__).parent.parent

# Create the FastAPI application
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts presence of heart disease from clinical features.",
    version="1.0.0"
)

# Module-level variables to hold the loaded model and scaler
model = None
scaler = None

@app.on_event("startup")
def load_model():
    """Load model and scaler when the server starts."""
    global model, scaler

    model_path  = ROOT / "models" / "model.pkl"
    scaler_path = ROOT / "data" / "processed" / "scaler.pkl"

    if not model_path.exists():
        raise RuntimeError(
            f"Model not found at {model_path}. "
            "Run dvc repro to generate the model."
        )
    if not scaler_path.exists():
        raise RuntimeError(
            f"Scaler not found at {scaler_path}. "
            "Run dvc repro to generate the scaler."
        )

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"Model loaded from:  {model_path}")
    print(f"Scaler loaded from: {scaler_path}")


class PatientFeatures(BaseModel):
    """
    Clinical features for heart disease prediction.
    All 13 features from the Cleveland Heart Disease dataset.
    """
    age:      float = Field(..., description="Age in years")
    sex:      float = Field(..., description="Sex: 1=male, 0=female")
    cp:       float = Field(..., description="Chest pain type: 1-4")
    trestbps: float = Field(..., description="Resting blood pressure (mm Hg)")
    chol:     float = Field(..., description="Serum cholesterol (mg/dl)")
    fbs:      float = Field(..., description="Fasting blood sugar > 120: 1=true, 0=false")
    restecg:  float = Field(..., description="Resting ECG results: 0-2")
    thalach:  float = Field(..., description="Maximum heart rate achieved")
    exang:    float = Field(..., description="Exercise induced angina: 1=yes, 0=no")
    oldpeak:  float = Field(..., description="ST depression induced by exercise")
    slope:    float = Field(..., description="Slope of peak exercise ST segment: 1-3")
    ca:       float = Field(..., description="Number of major vessels colored by fluoroscopy: 0-3")
    thal:     float = Field(..., description="Thalassemia: 3=normal, 6=fixed defect, 7=reversible defect")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
                "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
                "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
            }
        }


class PredictionResponse(BaseModel):
    """Prediction result returned by the /predict endpoint."""
    prediction:       int   = Field(..., description="0 = no heart disease, 1 = heart disease")
    confidence:       float = Field(..., description="Model confidence for the predicted class")
    predicted_label:  str   = Field(..., description="Human-readable prediction label")


@app.get("/health")
def health():
    """Health check — confirms the API is running and the model is loaded."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

# Feature order must match exactly what the model was trained on
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):
    """
    Predict heart disease presence from clinical features.
    Returns prediction (0/1), confidence score, and human-readable label.
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server startup logs."
        )

    # Build feature array in the correct column order
    features = np.array([[
        getattr(patient, name) for name in FEATURE_NAMES
    ]])

    # Scale using the fitted scaler from the pipeline
    features_scaled = scaler.transform(features)

    # Predict
    prediction  = int(model.predict(features_scaled)[0])
    confidence  = float(model.predict_proba(features_scaled)[0][prediction])

    label = "Heart disease detected" if prediction == 1 else "No heart disease detected"

    return PredictionResponse(
        prediction=prediction,
        confidence=round(confidence, 4),
        predicted_label=label
    )

