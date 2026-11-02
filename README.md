# End-to-End Customer Segmentation & AI Deployment Capstone

## Overview
This repository contains a production-grade machine learning pipeline and API service for Customer Segmentation and Prediction. It demonstrates end-to-end ML workflows: data preprocessing, feature engineering, clustering (K-Means), model evaluation, serialization, and deployment via FastAPI.

## Project Structure
```
ml_capstone_project/
├── app/
│   ├── main.py              # FastAPI Web Application & Prediction Endpoints
│   ├── schemas.py           # Pydantic Input/Output Validation Schemas
├── models/
│   ├── kmeans_model.pkl     # Serialized K-Means Model
│   └── scaler.pkl           # Standardized Feature Scaler
├── src/
│   ├── data_pipeline.py     # Preprocessing & Scaling Logic
│   └── train.py             # Model Training & Serialization Script
├── data/
│   └── customer_data.csv    # Raw & Processed Dataset
├── requirements.txt         # Project Dependencies
├── Dockerfile               # Containerization Setup
└── README.md                # Project Documentation
```

## Setup & Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/customer-segmentation-capstone.git
   cd customer-segmentation-capstone
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train and Serialize the Model:**
   ```bash
   python src/train.py
   ```

5. **Run the FastAPI Server:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   Access the interactive API docs at: `http://localhost:8000/docs`

## Docker Deployment
```bash
docker build -t customer-segmentation-api .
docker run -p 8000:8000 customer-segmentation-api
```
