import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from app.schemas import CustomerDataInput, PredictionOutput

app = FastAPI(
    title="Customer Segmentation Machine Learning API",
    description="Production REST API deploying K-Means Clustering for retail customer profiling.",
    version="1.0.0"
)

# Load model and scaler at startup
try:
    model = joblib.load("models/kmeans_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    model = None
    scaler = None

CLUSTER_NAMES = {
    0: "Moderate Income / Moderate Spenders",
    1: "Low Income / High Spenders",
    2: "High Income / High Spenders (Target Group)"
}

@app.get("/")
def home():
    return {"status": "Active", "message": "Customer Segmentation API is running."}

@app.post("/predict", response_model=PredictionOutput)
def predict_cluster(data: CustomerDataInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model assets not loaded.")
    
    raw_input = np.array([[data.annual_income, data.spending_score]])
    scaled_input = scaler.transform(raw_input)
    cluster = int(model.predict(scaled_input)[0])
    
    return PredictionOutput(
        cluster_id=cluster,
        cluster_label=CLUSTER_NAMES.get(cluster, f"Cluster {cluster}"),
        annual_income_k=data.annual_income,
        spending_score=data.spending_score
    )
